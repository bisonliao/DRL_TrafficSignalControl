# frap_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from conf.frap_config import Config


class FRAPModel(nn.Module):
    """
    Simplified FRAP implementation for 4-phase single intersection.
    Input assumptions:
      - movement_counts: tensor (batch, MOVEMENT_NUM), scalar counts per movement
      - cur_phase_onehot: tensor (batch, PHASE_NUM) one-hot (or can be derived)
    Output:
      - q_values: tensor (batch, PHASE_NUM)
    """
    def __init__(self):
        super().__init__()
      

        # per-movement encoders (h_v and h_s then merge -> d_i)
        self.v_fc = nn.Linear(1, Config.MOVEMENT_HIDDEN)   # from vehicle count (scalar)
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
            hd_layers.append(nn.ReLU())
            in_dim = Config.PAIR_HIDDEN
        self.hd_net = nn.Sequential(*hd_layers)

        # K layers for Hr (relation stream)
        hr_layers = []
        in_dim = pair_dim
        for k in range(Config.K_LAYERS):
            hr_layers.append(nn.Linear(in_dim, Config.PAIR_HIDDEN))
            hr_layers.append(nn.ReLU())
            in_dim = Config.PAIR_HIDDEN
        self.hr_net = nn.Sequential(*hr_layers)

        # after element-wise multiplication, project to scalar competition score per (p,q)
        self.comp_fc = nn.Linear(Config.PAIR_HIDDEN, 1)

    def forward(self, movement_counts, cur_phase_onehot=None):
        """
        movement_counts: tensor (batch, MOVEMENT_NUM) -- floats (vehicle counts)
        cur_phase_onehot: tensor (batch, PHASE_NUM) or None. If None, we treat s_i as 0 for all movements.
        Returns q_values: (batch, PHASE_NUM)
        """
        B = movement_counts.shape[0]
        P = Config.PHASE_NUM
        M = Config.MOVEMENT_NUM

        device = movement_counts.device

        # build per-movement signal bits: we need for each movement whether it's currently green (1) or red (0)
        # The env typically reports current phase index; but in network we take per-movement s_i bit.
        # If cur_phase_onehot is None, assume all zeros.
        if cur_phase_onehot is None:
            s_bits = torch.zeros((B, P), device=device)
        else:
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
        # movement_counts: (B,M) -> flatten to (B*M,1)
        mv_counts_flat = movement_counts.reshape(-1, 1)
        s_bits_flat = s_bits_mov.reshape(-1, 1)

        h_v = F.relu(self.v_fc(mv_counts_flat))   # (B*M, Hv)
        h_s = F.relu(self.s_fc(s_bits_flat))      # (B*M, Hv)
        merged = torch.cat([h_v, h_s], dim=1)     # (B*M, 2*Hv)
        d_i = F.relu(self.merge_fc(merged))       # (B*M, D)
        d_i = d_i.reshape(B, M, -1)               # (B, M, D)

        # build phase demand d(p) by summing the two movement demands in that phase
        # PHASE_MOVEMENTS lists movement indices per phase
        d_p_list = []
        for p_idx in range(P):
            mv_idx = Config.PHASE_MOVEMENTS[p_idx]
            d_sum = d_i[:, mv_idx[0], :] + d_i[:, mv_idx[1], :]  # (B,D)
            d_p_list.append(d_sum.unsqueeze(1))  # (B,1,D)
        d_p = torch.cat(d_p_list, dim=1)  # (B,P,D)

        pair_demands = []   # will hold [d(p), d(q)] for all (p,q)
        pair_relations = [] # will hold relation embedding e(p,q)

        for p in range(P):
            for q in range(P):
                # demand part: concat d(p), d(q)
                d_p_vec = d_p[:, p, :]   # (B, D)
                d_q_vec = d_p[:, q, :]   # (B, D)
                pair_demand = torch.cat([d_p_vec, d_q_vec], dim=1)  # (B, 2D)
                pair_demands.append(pair_demand)

                # relation embedding part: look up table entry (p,q)
                # self.relation_table: (P, P, REL)
                relation_vec = self.relation_table[p, q]  # (REL,)
                # expand to batch
                relation_vec = relation_vec.unsqueeze(0).expand(B, -1)  # (B, REL)
                pair_relations.append(relation_vec)


        # stack into big tensors
        D_flat = torch.cat(pair_demands, dim=0)   # (B*P*P, 2D)
        E_flat = torch.cat(pair_relations, dim=0) # (B*P*P, REL)

        '''
        这里 nn.Linear 和 nn.Conv2d(kernel=1) 在数学意义上是 等价的：
        nn.Conv2d(in_channels=C_in, out_channels=C_out, kernel_size=1)
        就是对每个位置上的输入通道做一个线性变换，等价于 nn.Linear(C_in, C_out)。
        所以很多实现为了简洁，会用 Linear 代替 1×1 conv。
        '''
        Hd = self.hd_net(D_flat)      # (B*P*P, PAIR_HIDDEN)
        Hr = self.hr_net(E_flat)      # (B*P*P, PAIR_HIDDEN)

        # element-wise multiplication
        Hc = Hd * Hr                  # (B*P*P, PAIR_HIDDEN)
        C_flat = F.relu(self.comp_fc(Hc))  # (B*P*P, 1) 又是用Linear替代了1x1卷积
        C = C_flat.reshape(B, P, P)        # (B, P, P) competition score of p against q

        # for each phase p, sum over opponents q (including itself) to get final phase score
        q_values = torch.sum(C, dim=2)  # (B, P)

        return q_values
