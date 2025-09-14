
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import argparse
import datetime
import os
from TscEnv import TscEnv
from torch.utils.tensorboard import SummaryWriter
from config import Config

'''
特征里，每个方向上的排队长度要安排上
出现了没有车通过，也一直亮着绿灯的情况。所以最大相位时长要做限制。最小的也有必要，虽然学习到了较长的相位，但还是有个别相位时间很短
某个队列长度超过阈值，就终止回合，并做较大的处罚

有的方向上车辆不多，就会一直等一直等。解决方法：各个方向上的排队最大时长要输入，并针对多次达到最大阈值，给与较大的处罚；或者奖励除了是等待车辆数的负数，也是等待时长的负数？两者的数量级要对等才行
'''

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(f'logs/tsc_{datetime.datetime.now().strftime("%y%m%d_%H%M%S")}')
# 超参数
gamma = 0.9  # 折扣因子
epsilon = 1.0  # 初始探索率
epsilon_min = 0.01  # 最低探索率
epsilon_decay = 0.97  # 探索率衰减
learning_rate = 1e-3  # 学习率
batch_size = 128  # 经验回放的批量大小
memory_size = 100000  # 经验池大小
target_update_freq = 3  # 目标网络更新频率

env = TscEnv(writer)
n_state = env.observation_space.shape[0]  # 状态维度
n_action = env.action_space.n  # 动作数量
print(f'action number={n_action}, obs shape={env.observation_space.shape}')


# DQN 网络定义
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# 初始化网络
model = DQN(n_state, n_action).to(device)
target_model = DQN(n_state, n_action).to(device)
target_model.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
memory = deque(maxlen=memory_size)


def select_action(state, epsilon):
    """基于 ε-greedy 选择动作"""
    if random.random() < epsilon:
        return random.randint(0, n_action - 1)  # 随机选择
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # 变换前：[4] -> 变换后：[1, 4]
        return model(state).argmax(1).item()  # 选取 Q 值最大的动作


def train():
    if len(memory) < batch_size:
        return 9999.0  # 经验池数据不足时不训练

    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states).to(device)  # (batch_size, 4)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)  # (batch_size,) -> (batch_size, 1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)  # (batch_size,) -> (batch_size, 1)
    next_states = torch.FloatTensor(next_states).to(device)  # (batch_size, 4)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)  # (batch_size,) -> (batch_size, 1)

    # 计算当前 Q 值
    q_values = model(states).gather(1, actions)  # 从 Q(s, a) 选取执行的动作 Q 值

    # 计算目标 Q 值
    next_q_values = target_model(next_states).max(1, keepdim=True)[0]  # 选取 Q(s', a') 的最大值
    target_q_values = rewards + gamma * next_q_values * (1 - dones)  # TD 目标

    # 计算损失
    loss = F.mse_loss(q_values, target_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def save_checkpoint(id):
    path=f"./checkpoints/dqn_checkpoint_{id}.pth"
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Checkpoint loaded from {path}")
    else:
        print("No checkpoint found, starting from scratch.")


def main(mode):
    global epsilon

    if mode == "train":
        episodes = 1000

        if input("preload model?[Y/n]") == "Y":
            load_checkpoint('./checkpoints/dqn_checkpoint_-20657.283203125.pth')
            epsilon = 0.1
            print(f'preload successfully')

        for episode in range(episodes):
            save_replay = ( episode > 20 and episode % 37 == 7 )
            state = env.reset(save_replay=save_replay)
            state = state[0]  # 适配 Gym v26
            total_reward = 0

            while True:
                action = select_action(state, epsilon)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 经验回放缓存
                memory.append((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward

                # 训练 DQN
                loss = train()

                if done:
                    break

            # 逐步降低 epsilon，减少随机探索，提高利用率
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            # 定期更新目标网络，提高稳定性
            if episode % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())

            # 定期保存模型
            if episode % 50 == 49:
                save_checkpoint(f'{episode}_{total_reward}')

            print(f"{datetime.datetime.now().strftime('%H:%M:%S')} Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.3f}, loss:{loss}")
            writer.add_scalar('train/episode_reward', total_reward, episode)
            writer.add_scalar('train/epsilon', epsilon, episode)
            writer.add_scalar('train/loss', loss, episode)
    else:
        load_checkpoint('./checkpoints/dqn_checkpoint_-16595.48046875.pth')
        model.eval()
        eval_env = TscEnv(writer)

        state = eval_env.reset(save_replay=True)
        state = state[0]  # 适配 Gym v26

        while True:
            action = select_action(state, 0)
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            state = next_state
                
            if done:
                break

        

if __name__ == "__main__":
    main("eval")
    
    

