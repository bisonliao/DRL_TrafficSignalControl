# frap_agent.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from conf.frap_config import Config
from model.frap_model import FRAPModel
from model.frap_model2 import Network
from env.TscEnv import TscEnv
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import random
import gymnasium as gym  # expected gymnasium interface
from collections import deque, Counter
from tools.common import *



Transition = namedtuple('Transition', ['obs', 'action', 'reward', 'next_obs', 'done'])

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # convert to arrays
        obs = [b.obs for b in batch]
        actions = np.array([b.action for b in batch], dtype=np.int64)
        rewards = np.array([b.reward for b in batch], dtype=np.float32)
        next_obs = [b.next_obs for b in batch]
        dones = np.array([b.done for b in batch], dtype=np.float32)
        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return len(self.buffer)
    

class FRAPAgent:

    
    def __init__(self,   seed: int = None):

        self.writer = SummaryWriter(f'logs/frap_{datetime.datetime.now().strftime("%y%m%d_%H%M%S")}')
        self.env = TscEnv(self.writer)
     
        
        if seed is None:
            seed = Config.SEED
        set_seed(seed)


        phase2movement = np.zeros((Config.PHASE_NUM, Config.MOVEMENT_NUM), dtype=np.int32)
        for p in range(len(Config.PHASE_MOVEMENTS)):
            for m in Config.PHASE_MOVEMENTS[p]:
                phase2movement[p][m] = 1

       

        # networks
        #self.model = FRAPModel().to(Config.DEVICE)
        #self.target_model = FRAPModel().to(Config.DEVICE)
        self.model = Network(Config.MOVEMENT_NUM, Config.PHASE_NUM, torch.tensor(phase2movement)).to(Config.DEVICE)
        self.target_model = Network(Config.MOVEMENT_NUM, Config.PHASE_NUM, torch.tensor(phase2movement)).to(Config.DEVICE)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LR)


        # replay
        self.replay = ReplayBuffer(Config.BUFFER_SIZE)
        self.epsilon_decay = Config.EPS_DECAY_RATE  # 探索率衰减

        # training counters
        self.total_steps = 0
        self.update_count = 0
        self.action_history = deque(maxlen=100) # rollout的时候的动作历史

    def observation_process(self):
        self.env.calc_vehicle_info()

        mv_cnt = np.zeros((Config.MOVEMENT_NUM,), dtype=np.int32) # 车辆数目
        mv_dense = np.zeros((Config.MOVEMENT_NUM,), dtype=np.float32) # 车辆密度
        


        curr_phase_onehot = np.zeros((Config.PHASE_NUM,), dtype=np.float32)
        curr_phase_onehot[self.env.current_phase-1] = self.env.phase_duration / Config.MAX_PHASE_DUR # todo：这里放当前相位持续了多长时间试试
        
        vehicle_ids = self.env.eng.get_vehicles(True)
        for vid in vehicle_ids:
            try:
                info = self.env.eng.get_vehicle_info(vid)
                if info.get('running', "0") != "1": #whether the vehicle is running
                    continue

                #  The next intersection if the vehicle is running on a lane
                if info.get('intersection', "")!= 'intersection_1_1':
                    continue
                
                speed = info['speed']       
                route = info['route'] # A string contains ids of following roads in the vehicle’s route which are separated by space
                movement_id = self.env._route2movement(route)
                mv_cnt[movement_id] += 1
                
                # 离停止线远的车辆（比如还在 upstream 很远的地方）对当前信号相位的即时效果没有直接影响。
                # 所以只统计停止线附近 30m 范围内的车辆数
                # 可以直接调用CityFlow的get_lane_vehicle_count()函数的
                drivable = info['drivable']
                vehicle_distance = info["distance"]  # 已行驶距离
                to_stop_line = Config.LANE_LEN - float(vehicle_distance)
                if to_stop_line > (Config.CELL_LEN *Config.OBSERV_CELL_NUM) or to_stop_line < 0:
                    continue
                
                mv_dense[movement_id] += 1/(Config.OBSERV_CELL_NUM)
                

            except Exception as e:
                print(f"Error retrieving info for vehicle {vid}: {e}")
        
        mv_feat = np.stack([mv_cnt, mv_dense], axis=0).transpose() # shape:(8,2)
        #assert mv_feat.shape[0] == Config.MOVEMENT_NUM and mv_feat.shape[1] == Config.FEAT_DIM, "invalid mv feature"
        #return mv_feat, curr_phase_onehot
        return np.array([mv_cnt]).transpose() / Config.MAX_QLEN, curr_phase_onehot
        
    
    def reward_shape1(self):
        vehicle_ids = self.env.eng.get_vehicles(True)
        delay_sum = 0
        for vid in vehicle_ids:
            try:
                info = self.env.eng.get_vehicle_info(vid)
                if info.get('running', "0") != "1": #whether the vehicle is running
                    continue

                #  The next intersection if the vehicle is running on a lane
                if info.get('intersection', "")!= 'intersection_1_1':
                    continue
                

                speed = info['speed']       
                route = info['route'] # A string contains ids of following roads in the vehicle’s route which are separated by space
                movement_id = self.env._route2movement(route)

                delay = max(0, 1 - float(speed) / 11.11) # 11.11是flow.json文件里普遍用到的最大车速，单位 m/s
                delay_sum += delay
                
            except Exception as e:
                print(f"Error retrieving info for vehicle {vid}: {e}")
        return -delay_sum
    
    def reward_shape(self):
        # 某些方向的等待车辆堆积超过阈值,
        if np.any(self.env.waiting_qlen >= Config.MAX_QLEN):
            self.long_q_duration += 1
        else:
            self.long_q_duration = 0

        if self.long_q_duration > 20:
            #print(f'too long queue')
            return -5
        
        # 某个相位持续时间超过 30s
        if self.env.phase_duration >= Config.MAX_PHASE_DUR:  
            #print(f'too long phase')
            return -5
        
        vehicle_ids = self.env.eng.get_vehicles(True)
        q_lens = torch.zeros((Config.MOVEMENT_NUM,), dtype=torch.float32)
        for vid in vehicle_ids:
            try:
                info = self.env.eng.get_vehicle_info(vid)
                if info.get('running', "0") != "1": #whether the vehicle is running
                    continue

                #  The next intersection if the vehicle is running on a lane
                if info.get('intersection', "")!= 'intersection_1_1':
                    continue
                

                speed = info['speed']       
                if float(speed) > 0.1:
                    continue
                route = info['route'] # A string contains ids of following roads in the vehicle’s route which are separated by space
                movement_id = self.env._route2movement(route)
                q_lens[movement_id] += 1
                
            except Exception as e:
                print(f"Error retrieving info for vehicle {vid}: {e}")
        reward = torch.mean(q_lens).item()
        reward = reward**0.25

        return -reward
        
    
    def reward_shape2(self):
        # 某些方向的等待车辆堆积超过阈值,
        if np.any(self.env.waiting_qlen >= Config.MAX_QLEN):
            self.long_q_duration += 1
        else:
            self.long_q_duration = 0

        if self.long_q_duration > 20:
            #print(f'too long queue')
            return -10
        
        # 某个相位持续时间超过 30s
        if self.env.phase_duration >= Config.MAX_PHASE_DUR:  
            #print(f'too long phase')
            return -10
        
        
        vehicle_cnt = np.sum(self.env.waiting_qlen)

        if self.env.phase_duration == 1: #刚经历了切换，有3s红灯时间成本，所以-1处罚
            reward = -(vehicle_cnt**0.25) - 1
        else:
            reward = -(vehicle_cnt**0.25)
        
        return reward

    def select_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randrange(Config.PHASE_NUM)
        
        mv_counts, phase_onehot = obs
        mv_tensor = torch.tensor(mv_counts, dtype=torch.float32, device=Config.DEVICE).unsqueeze(0)
        phase_tensor = torch.tensor(phase_onehot, dtype=torch.float32, device=Config.DEVICE).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.model(mv_tensor, phase_tensor)  # (1,P)
            q_vals = q_vals.cpu().numpy().squeeze(0)
        action = int(np.argmax(q_vals))
        self.action_history.append(action)
        
        return action
    


    def _obs_to_tensors(self, obs_list):
        """Convert list of obs to tensors for batch training"""
        B = len(obs_list)
        M = Config.MOVEMENT_NUM
        P = Config.PHASE_NUM
        mv_array = np.zeros((B, M, Config.FEAT_DIM), dtype=np.float32)
        ph_array = np.zeros((B, P), dtype=np.float32)
        for i, ob in enumerate(obs_list):
            mv, ph = ob
            mv_array[i] = mv
            ph_array[i] = ph
        mv_tensor = torch.tensor(mv_array, dtype=torch.float32, device=Config.DEVICE)
        ph_tensor = torch.tensor(ph_array, dtype=torch.float32, device=Config.DEVICE)

        assert mv_tensor.shape == (B, M, Config.FEAT_DIM), "invalid mv_tensor"
        return mv_tensor, ph_tensor
    

    def train_step(self):
        if len(self.replay) < Config.MIN_REPLAY_SIZE:
            return None
        
        obs_batch, actions, rewards, next_obs_batch, dones = self.replay.sample(Config.BATCH_SIZE)
        mv_batch, ph_batch = self._obs_to_tensors(obs_batch)
        mv_next_batch, ph_next_batch = self._obs_to_tensors(next_obs_batch)

        # compute Q(s,a)
        q_vals = self.model(mv_batch, ph_batch)   # (B,P)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=Config.DEVICE)
        q_s_a = q_vals.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)  # (B,)

        # compute target r + gamma * max_a' Q_target(s',a') * (1 - done)
        with torch.no_grad():
            q_next = self.target_model(mv_next_batch, ph_next_batch)  # (B,P)
            q_next_max = q_next.max(dim=1)[0]
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=Config.DEVICE)
            dones_tensor = torch.tensor(dones, dtype=torch.float32, device=Config.DEVICE)
            target = rewards_tensor + (1.0 - dones_tensor) * (Config.GAMMA * q_next_max)

        loss = nn.MSELoss()(q_s_a, target)
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.total_steps % 907 == 7:
            log_network_statistics(self.model, self.writer,self.total_steps)
            check_grad_norm(self.model, self.writer, self.total_steps)


        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.update_count += 1
        return loss.item()
    
    def save_checkpoint(self, id):
        os.makedirs('./checkpoints/', exist_ok=True)
        path=f"./checkpoints/frap_checkpoint_{id}.pth"
        torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved to {path}")


    def load_checkpoint(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=Config.DEVICE))
            print(f"Checkpoint loaded from {path}")
        else:
            print("No checkpoint found, starting from scratch.")
    def train(self, num_episodes=500, max_steps_per_episode=2000):
        eps_start = Config.EPS_START
        eps_end = Config.EPS_END
        total_steps = 0
        epsilon = eps_start
        losses = deque(maxlen=100)
        episode_start = 1


        # 预加载继续训练
        '''
        self.load_checkpoint('./checkpoints/frap_checkpoint_176_-9339.384.pth')
        epsilon = 0.02
        episode_start = 200
        total_steps = 2000 * episode_start
        self.env.total_step = total_steps
        '''

        for ep in range(episode_start, num_episodes + 1):
            self.env.reset()
            self.reset()
            obs = self.observation_process()
            done = False
            ep_reward = 0.0
            steps = 0
            while not done:
                action = self.select_action(obs, epsilon)
                _,_, terminated, truncated, info = self.env.step(action)
                next_obs = self.observation_process()
                reward = self.reward_shape()
                done  = terminated or truncated
                if self.total_steps % 117 == 7:
                    self.writer.add_scalar('train/step_reward', reward, self.total_steps)
                self.replay.push(obs, action, reward, next_obs, float(done))
                obs = next_obs
                ep_reward += float(reward)
                steps += 1
                total_steps += 1
                self.total_steps = total_steps

                # train
                if total_steps % Config.TRAIN_FREQ == 0 and len(self.replay) >= Config.MIN_REPLAY_SIZE:
                    loss = self.train_step()
                    if loss is not None:
                        losses.append(loss)

                # update target network
                if total_steps % Config.TARGET_UPDATE_FREQ == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

                if done:
                    break

            epsilon = max(eps_end, epsilon * self.epsilon_decay)

            if ep % Config.PRINT_EVERY == 0 or ep == 1:
                avg_loss = np.mean(losses) if losses else 0.0
                print(f"{datetime.datetime.now().strftime('%H:%M:%S')} [EP {ep:4d}] ep_reward={ep_reward:.3f} total_steps={total_steps} epsilon={epsilon:.3f} avg_loss={avg_loss:.4f}")
                att, thrput = self.env.get_biz_metrics()
                self.writer.add_scalar('train/loss', avg_loss, ep)
                self.writer.add_scalar('train/episode_reward', ep_reward, ep)
                self.writer.add_scalar('train/epsilon', epsilon, ep)
                self.writer.add_scalar('train/att', att, ep)
                self.writer.add_scalar('train/thrput', thrput, ep)
                self.writer.add_scalar('train/action_entropy', check_action_entropy(self.action_history), ep)
            if ep % 25 == 1 and ep > 25:
                self.save_checkpoint(f'{ep}_{ep_reward:.3f}')

        # final copy
        self.target_model.load_state_dict(self.model.state_dict())
        print("Training finished.")

    def reset(self):
        self.long_q_duration = 0

    def eval(self, filename):
        self.load_checkpoint(filename)
        self.model.eval()

        self.env.reset(save_replay=True)
        state = self.observation_process()

        while True:
            action = self.select_action(state, 0)
            if random.random() < 0.1: print(f'action={action}')
            _, _, terminated, truncated, _ = self.env.step(action)
            next_state = self.observation_process()
            done = terminated or truncated
            state = next_state
                    
            if done:
                break
        att,thrput = self.env.get_biz_metrics()
        print(f'att={att}')
    

def check_computation_graph_connectivity(model:FRAPModel):
    """检查计算图是否断开"""
    print("\n=== Computation Graph Connectivity Check ===")
    
    # 重新进行一次前向传播以便检查计算图
    with torch.no_grad():
        test_input = torch.randn(1, Config.MOVEMENT_NUM, Config.FEAT_DIM).to(Config.DEVICE)
        test_phase = torch.zeros(1, Config.PHASE_NUM).to(Config.DEVICE)
    
    # 确保requires_grad为True
    test_input.requires_grad_(True)
    test_phase.requires_grad_(True)
    
    # 前向传播
    output = model(test_input, test_phase)
    print(f"Output requires_grad: {output.requires_grad},output={output}")
    
    # 检查计算图
    if output.grad_fn is None:
        print("ERROR: Output has no grad_fn - computation graph is broken!")
        return False
    
    print(f"Output grad_fn: {output.grad_fn}")
    
    # 手动计算损失并反向传播
    target = torch.randn_like(output)
    loss = nn.MSELoss()(output, target)
    print(f"Test loss: {loss.item():.6f}")
    
    # 反向传播
    model.zero_grad()
    loss.backward()
    
    # 检查梯度
    has_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 1e-10:
                has_grad = True
                print(f"{name}: grad norm = {grad_norm:.6e}")
        else:
            print(f"{name}: grad is None!!!")
    
    print(f'has_grad={has_grad}')
    return has_grad

if __name__ == "__main__":

    agent = FRAPAgent()
    check_computation_graph_connectivity(agent.model)
    agent.train()
    #agent.eval('./checkpoints/frap_checkpoint_176_-38666.250.pth')
