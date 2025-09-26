# frap_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from conf.frap_config import Config


class FRAPModel2(nn.Module):
    def __init__(self):
        super().__init__()
        # 直接学习每个phase的Q值，忽略输入
        self.q_values = nn.Parameter(torch.randn(Config.PHASE_NUM))
    
    def forward(self, movement_feat, cur_phase_onehot=None):
        batch_size = movement_feat.shape[0]
        assert not torch.all(movement_feat == 0), 'invalid feature!!!'
        # 扩展为batch size
        return self.q_values.unsqueeze(0).expand(batch_size, -1)


class FRAPModel(nn.Module):
    """
    Simplified FRAP implementation for 4-phase single intersection.
    Input assumptions:
      - movement_feat: tensor (batch, MOVEMENT_NUM, FEAT_DIM), feature per movement
      - cur_phase_onehot: tensor (batch, PHASE_NUM) one-hot (or can be derived)
    Output:
      - q_values: tensor (batch, PHASE_NUM)
    """
    def __init__(self):
        super().__init__()

        # per-movement encoders (h_v and h_s then merge -> d_i)
        #self.v_fc = nn.Linear(1, Config.MOVEMENT_HIDDEN)   # from vehicle count (scalar)
        
        self.v_fc = nn.Sequential(
            nn.Linear(Config.FEAT_DIM, Config.MOVEMENT_HIDDEN),
            nn.LayerNorm(Config.MOVEMENT_HIDDEN),
        )

        self.s_fc = nn.Linear(1, Config.MOVEMENT_HIDDEN)   # from current-signal-bit (scalar)

        self.merge_fc = nn.Linear(Config.MOVEMENT_HIDDEN*2, Config.MOVEMENT_EMB)

        # relation embedding table for phase pairs (P x P x RELATION_DIM)
        self.relation_table = nn.Parameter(
            torch.randn(Config.PHASE_NUM, Config.PHASE_NUM, Config.RELATION_DIM) * 0.1
        )

        # pair-demand -> embedded vector
        pair_input_dim = Config.MOVEMENT_EMB * 2  # concat d(p), d(q)
        pair_dim = Config.RELATION_DIM

        # K layers for Hd (demand stream)
        hd_layers = []
        in_dim = pair_input_dim
        for k in range(Config.K_LAYERS):
            hd_layers.append(nn.Linear(in_dim, Config.PAIR_HIDDEN))
            hd_layers.append(nn.LeakyReLU())
            in_dim = Config.PAIR_HIDDEN
        self.hd_net = nn.Sequential(*hd_layers)

        # K layers for Hr (relation stream)
        hr_layers = []
        in_dim = pair_dim
        for k in range(Config.K_LAYERS):
            hr_layers.append(nn.Linear(in_dim, Config.PAIR_HIDDEN))
            hr_layers.append(nn.LeakyReLU())
            in_dim = Config.PAIR_HIDDEN
        self.hr_net = nn.Sequential(*hr_layers)

        # after element-wise multiplication, project to scalar competition score per (p,q)
        self.comp_fc = nn.Linear(Config.PAIR_HIDDEN, 1)

        self._init_weights() # seed不为42的时候，这个也可以不用调用

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
        # 特别初始化relation_table
        nn.init.normal_(self.relation_table, mean=0, std=0.1)

    def forward(self, movement_feat, cur_phase_onehot):
        """
        movement_feat: tensor (batch, MOVEMENT_NUM, FEAT_DIM) -- floats (vehicle counts)
        cur_phase_onehot: tensor (batch, PHASE_NUM) or None. If None, we treat s_i as 0 for all movements.
        Returns q_values: (batch, PHASE_NUM)
        """
        B = movement_feat.shape[0]
        P = Config.PHASE_NUM
        M = Config.MOVEMENT_NUM

        device = movement_feat.device

        # build per-movement signal bits: we need for each movement whether it's currently green (1) or red (0)
        # The env typically reports current phase index; but in network we take per-movement s_i bit.
        # If cur_phase_onehot is None, assume all zeros.
        s_bits = cur_phase_onehot  # (B, P)

        # Convert phase-onehot -> movement-wise s_i (movement count vector length M)
        # We'll map each movement to its phase (the phase that would turn it green). For our PHASE_MOVEMENTS mapping,
        # movements indices appear in a phase pair. Build movement->phase indicator:
        # For simplicity, for each movement we check which phase contains it (there might be exactly one).
        movement_to_phase = torch.zeros((M, P), device=device)  # (M,P) boolean-like
        for p_idx, moves in enumerate(Config.PHASE_MOVEMENTS):
            for mv in moves:
                movement_to_phase[mv, p_idx] = 1.0

        # s_i = sum_p movement_to_phase[mv,p] * s_p
        # s_bits: (B, P) -> (B, 1, P) multiply (M,P) -> result (B, M)
        s_bits_mov = torch.matmul(s_bits, movement_to_phase.t())  # (B, M) 指示当前相位是否绿灯/放行，1表示放行，0表示禁行

        # per-movement encoding (process per movement separately)
        # movement_feat: (B,M, FEAT_DIM) -> flatten to (B*M, FEAT_DIM)
        mv_feat_flat = movement_feat.reshape(-1, Config.FEAT_DIM)
        s_bits_flat = s_bits_mov.reshape(-1, 1)

        h_v = F.leaky_relu(self.v_fc(mv_feat_flat))   # (B*M, Hv)
        h_s = F.leaky_relu(self.s_fc(s_bits_flat))      # (B*M, Hv)
        merged = torch.cat([h_v, h_s], dim=1)     # (B*M, 2*Hv)
        d_i = F.leaky_relu(self.merge_fc(merged))       # (B*M, D)
        d_i = d_i.reshape(B, M, -1)               # (B, M, D)

        ##----------- 至此，完成了每个movement的demand的计算-----------------##

        # build phase demand d(p) by summing the two movement demands in that phase
        # PHASE_MOVEMENTS lists movement indices per phase
        # 一个Phase p由两个不冲突的movement组成，把这两个mv的demand直接按元素相加，作为p的demand
        # d_p就是每个Phase的Demand
        d_p_list = []
        for p_idx in range(P):
            mv_idx = Config.PHASE_MOVEMENTS[p_idx]
            d_sum = d_i[:, mv_idx[0], :] + d_i[:, mv_idx[1], :]  # (B,D)
            d_p_list.append(d_sum.unsqueeze(1))  # (B,1,D)
        d_p = torch.cat(d_p_list, dim=1)  # (B,P,D)

        # 对固定的phase p，枚举它的所有对手phase q,形成一个pair
        # p和q的demand拼接起来作为pair的demand
        # p和q的relation（竞争与否）是查一张表，该表也是待学习参数
        pair_demands = []   # will hold [d(p), d(q)] for all (p,q)
        pair_relations = [] # will hold relation embedding e(p,q)

        for p in range(P):
            for q in range(P):
                if p == q:
                    continue
                # demand part: concat d(p), d(q)
                d_p_vec = d_p[:, p, :]   # (B, D)
                d_q_vec = d_p[:, q, :]   # (B, D)
                pair_demand = torch.cat([d_p_vec, d_q_vec], dim=1)  # (B, 2D)
                pair_demands.append(pair_demand)

                # relation embedding part: look up table entry (p,q)
                # self.relation_table: (P, P, REL)
                relation_vec = self.relation_table[p, q]  # (REL,)
                # expand to batch
                # 这 B 个样本对应的 relation embedding 是一样的（因为相位关系不依赖具体样本）。
                # 所以我们要把这个（REL,)向量复制B份，得到（B，REL）形状的向量
                relation_vec = relation_vec.unsqueeze(0).expand(B, -1)  # (B, REL)
                pair_relations.append(relation_vec)

        #至此，pair_demands中有 PxP个张量，每个张量的尺寸是(B,2D)。列表中依次为第一个相位的相位对样本、第二个相位的相位对样本...
        # torch.cat(pair_demands, dim=0)会沿着这些张量的B的维度做拼接，也就是保持第二个维度为2D不变，依次拼接第一个维度，得到BXPXP的维度
        # 数据组织遵循"批次优先"的原则，即结果张量的前 B 行​​：第一个相位对 (p0,q0)的所有批次样本，依次类推

        # stack into big tensors, 这里直接用了cat而不是stack，相当于先stack再reshape
        D_flat = torch.cat(pair_demands, dim=0)   # (B*P*P, 2D)
        E_flat = torch.cat(pair_relations, dim=0) # (B*P*P, REL)

        #-------- 至此，完成了E 和 D两个立方体的构造，维度还多了一个B --------#

        '''
        这里 nn.Linear 和 nn.Conv2d(kernel=1) 在数学意义上是 等价的：
        nn.Conv2d(in_channels=C_in, out_channels=C_out, kernel_size=1)
        就是对每个位置上的输入通道做一个线性变换，等价于 nn.Linear(C_in, C_out)。
        所以很多实现为了简洁，会用 Linear 代替 1×1 conv。

        还是很直观的：因为D_flat的形状是(B*P*P, 2D)，而 Linear 的运算就是对dim=1第二维
        进行加权求和得到一个值，相当于对(B,P,P, 2D)的第四维做卷积，也就是保持PxP的关系不变
        不在PXP的维度上做交叉卷积
        '''
        Hd = self.hd_net(D_flat)      # (B*P*P, PAIR_HIDDEN)
        Hr = self.hr_net(E_flat)      # (B*P*P, PAIR_HIDDEN)

        # element-wise multiplication
        Hc = Hd * Hr                  # (B*P*P, PAIR_HIDDEN)
        C_flat = F.leaky_relu(self.comp_fc(Hc))  # (B*P*P, 1) 又是用Linear替代了1x1卷积
        C = C_flat.reshape(B, P, P-1)        # (B, P, P) competition score of p against q

        # for each phase p, sum over opponents q (including itself) to get final phase score
        q_values = torch.sum(C, dim=2)  # (B, P)

        return q_values
