import datetime
import random
import time

import numpy as np
import torch
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from torch.utils.tensorboard import SummaryWriter
import math
import cityflow
from collections import deque
from config import Config



class TscEnv(gym.Env):
   
    def __init__(self, writer:SummaryWriter):

        super(TscEnv, self).__init__()
        self.writer = writer

        self.eng = cityflow.Engine("/home/bison/tsc_project1/config.json", thread_num=1)
        

       
        # 定义动作空间和状态空间
        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low= np.array([0,0,0,0,  0,  0,0,0,0,  0,0,0,0, 0,0,0,0]),
            high=np.array([1,1,1,1,  1,  1,1,1,1,  1,1,1,1, 1,1,1,1]),
            dtype=np.float32
        )
        

        # 环境参数
        self.max_steps = 10000  # 最大步数
        self.current_step = 0 #每一回合里的步数计数器
        self.total_step = 0  #环境运行过程中一直累加的总的计数器
        self.durations = deque(maxlen=100)

        self.reset()

    

    def reset(self, seed=None, save_replay=False):
        """重置环境到初始状态"""
        #self.eng.set_save_replay(save_replay)
        #self.eng.set_replay_file(f'replay_{datetime.datetime.now().strftime("%y%m%d_%H%M%S")}.txt')
        self.eng.reset()

        self.current_step = 0
        self.current_phase = -1
        self.phase_duration = 0 # 一个相位持续多久
        if len(self.durations) > 0:
            self.writer.add_scalar('phase_duration/max', np.max(self.durations), self.total_step)
            self.writer.add_scalar('phase_duration/min', np.min(self.durations), self.total_step)
            self.writer.add_scalar('phase_duration/avg', np.mean(self.durations), self.total_step)
            self.writer.add_scalar('phase_duration/std', np.std(self.durations), self.total_step)
        self.durations.clear()

        self.long_q_duration = 0 #出现超过30两等待车的累计时间
        
        # 获取初始状态
        state = self._get_state()

        return state,{}

    def step(self, action):

        new_phase = action + 1
        phase_modified = False

        if self.current_phase != new_phase: #相位有变化
            # 先来3s红灯清场
            self.eng.set_tl_phase('intersection_1_1', 0)
            for _ in range(3):
                self.eng.next_step()
            # 然后轮转到下一个方向放行
            self.eng.set_tl_phase('intersection_1_1',new_phase)
            self.current_phase = new_phase
            phase_modified = True

            self.durations.append(self.phase_duration)
            self.phase_duration = 1
        else:
            self.phase_duration += 1

        
        self.eng.next_step()

        self.current_step += 1
        self.total_step += 1

        

        # 获取新状态
        state = self._get_state()

        # 计算奖励
        reward, terminated = self._compute_reward()
        if self.total_step % 997 == 7:
            self.writer.add_scalar('train/step_reward', reward, self.total_step)

        # 检查是否超过最大步数
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True


        info = {
            "steps": self.current_step,
        }


        return state, reward, terminated, truncated, info
    
    def _route2phaseID(self, route:str):
        route = route.strip()
        if route == "road_2_1_2 road_1_1_2" or route == "road_0_1_0 road_1_1_0": # 东西直行
            return 0
        if route == "road_1_2_3 road_1_1_3" or route == "road_1_0_1 road_1_1_1": #南北直行
            return 1
        if route == "road_2_1_2 road_1_1_3" or route == "road_0_1_0 road_1_1_1": # 东西左转
            return 2
        if route == "road_1_2_3 road_1_1_0" or route == "road_1_0_1 road_1_1_2": # 南北左转
            return 3
        assert False, f"invalid route {route} "

    def calc_vehicle_info(self):
        

        self.vehicle_dense = np.zeros(Config.PHASE_NUM, dtype=np.float32) #停止线前面30m范围内的车辆密度
        self.avg_speed = np.zeros(Config.PHASE_NUM, dtype=np.float32)#停止线前面30m范围内的车辆平均速度
        self.waiting_qlen = np.zeros(Config.PHASE_NUM, dtype=np.float32) # 入口车道上排队的车辆数目，由于每个方向都是一个入口车道，所以这里就是等待队列的长度

        speed_list = [deque(maxlen=100) for _ in range(Config.PHASE_NUM)]
        vehicle_ids = self.eng.get_vehicles(True)
        for vid in vehicle_ids:
            try:
                info = self.eng.get_vehicle_info(vid)
                if info.get('running', "0") != "1":
                    continue

                if info.get('intersection', "")!= 'intersection_1_1':
                    continue
                

                speed = info['speed']       
                route = info['route']
                phase_id = self._route2phaseID(route)

                if float(speed) < 0.1:
                    self.waiting_qlen[phase_id] += 1

                drivable = info['drivable']
                vehicle_distance = info["distance"]  # 已行驶距离
                to_stop_line = 300 - float(vehicle_distance)
                if to_stop_line > (Config.CELL_LEN *Config.OBSERV_CELL_NUM) or to_stop_line < 0:
                    continue

                
                self.vehicle_dense[phase_id] += 1.0 / (Config.OBSERV_CELL_NUM * 2) 
                speed_list[phase_id].append( float(speed) )

            except Exception as e:
                print(f"Error retrieving info for vehicle {vid}: {e}")
        
        for i in range(Config.PHASE_NUM):
            if len(speed_list[i]) > 0:
                self.avg_speed[i] = sum(speed_list[i]) / len(speed_list[i])


    def _get_state(self):
        

        self.calc_vehicle_info()

        # 当前相位的one-hot表示
        phase_onehot = np.zeros(Config.PHASE_NUM, dtype=np.long)
        phase_onehot[self.current_phase-1] = 1
               

        # 合并所有状态信息
        state = np.concatenate(
            [
            phase_onehot,
            [self.phase_duration / 30],
            self.vehicle_dense, 
            self.avg_speed / 22,
            self.waiting_qlen / Config.MAX_QLEN,
     
            ], 
            dtype=np.float32)


        return state

    def _compute_reward(self):
        # 简单实现：路口入口车道上等待的车辆数量和的负值。鼓励不要在路口堆积车辆

        # 某些方向的等待车辆堆积超过阈值,
        if np.any(self.waiting_qlen >= Config.MAX_QLEN):
            self.long_q_duration += 1
        else:
            self.long_q_duration = 0

        if self.long_q_duration > 20:
            #print(f'too long queue')
            return -10, False
        
        # 某个相位持续时间超过 30s
        if self.phase_duration >= Config.MAX_PHASE_DUR:  
            #print(f'too long phase')
            return -10, False
        

        vehicle_cnt = np.sum(self.waiting_qlen)

        terminated = False
        if self.phase_duration == 1: #刚经历了切换，有3s红灯时间成本，所以-1处罚
            reward = -(vehicle_cnt**0.25) - 1
        else:
            reward = -(vehicle_cnt**0.25)

        return reward, terminated

    def render(self, mode='human'):
        """渲染环境"""
        pass  # PyBullet会自动处理渲染

    def close(self):
        """关闭环境"""
        pass

