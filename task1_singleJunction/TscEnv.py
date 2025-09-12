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



class TscEnv(gym.Env):
   
    def __init__(self, writer:SummaryWriter):

        super(TscEnv, self).__init__()
        self.writer = writer

        self.eng = cityflow.Engine("/home/bison/tsc_project1/config.json", thread_num=1)
        

       
        # 定义动作空间和状态空间
        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low= np.array([0,0,0,0,  0,  0,0,0,0,  0,0,0,0]),
            high=np.array([1,1,1,1,  1,  1,1,1,1,  1,1,1,1]),
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
        self.eng.set_save_replay(save_replay)
        self.eng.set_replay_file(f'replay_{datetime.datetime.now().strftime("%y%m%d_%H%M%S")}.txt')
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
        if self.total_step % 9997 == 7:
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


    def _get_state(self):
        

        PHASE_NUM = 4
        OBSERV_CELL_NUM = 6
        CELL_LEN = 5

        vehicle_dense = np.zeros(PHASE_NUM, dtype=np.float32)
        avg_speed = np.zeros(PHASE_NUM, dtype=np.float32)

        speed_list = [deque(maxlen=100) for _ in range(PHASE_NUM)]
        vehicle_ids = self.eng.get_vehicles(True)
        for vid in vehicle_ids:
            try:
                info = self.eng.get_vehicle_info(vid)
                if info.get('running', "0") != "1":
                    continue

                if info.get('intersection', "")!= 'intersection_1_1':
                    continue
                

                speed = info['speed']
                distance = info['distance']             
                route = info['route']
                drivable = info['drivable']
                vehicle_distance = info["distance"]  # 已行驶距离
                to_stop_line = 300 - float(vehicle_distance)
                if to_stop_line > (CELL_LEN *OBSERV_CELL_NUM) or to_stop_line < 0:
                    continue

                phase_id = self._route2phaseID(route)
                vehicle_dense[phase_id] += 1.0 / (OBSERV_CELL_NUM * 2) 
                speed_list[phase_id].append( float(speed) )

            except Exception as e:
                print(f"Error retrieving info for vehicle {vid}: {e}")
        
        for i in range(PHASE_NUM):
            if len(speed_list[i]) > 0:
                avg_speed[i] = sum(speed_list[i]) / len(speed_list[i])

        # 当前相位的one-hot表示
        phase_onehot = np.zeros(PHASE_NUM, dtype=np.long)
        phase_onehot[self.current_phase-1] = 1
               

        # 合并所有状态信息
        state = np.concatenate(
            [
            phase_onehot,
            [self.phase_duration / 30],
            vehicle_dense, 
            avg_speed / 22,
     
            ], 
            dtype=np.float32)


        return state

    def _compute_reward(self):
        # 简单实现：路口入口车道上等待的车辆数量和的负值。鼓励不要在路口堆积车辆
        vehicle_ids = self.eng.get_vehicles(True)
        vehicle_cnt = 0
        for vid in vehicle_ids:
            try:
                info = self.eng.get_vehicle_info(vid)
                if info.get('running', "0") != "1":
                    continue

                if info.get('intersection', "") != 'intersection_1_1':
                    continue
                
                speed = info['speed']
                speed = float(speed)
                if speed > 0.1:
                    continue
                vehicle_cnt += 1

            except Exception as e:
                print(f"Error retrieving info for vehicle {vid}: {e}")

        terminated = False
        reward = -(vehicle_cnt**0.25)


        return reward, terminated

    def render(self, mode='human'):
        """渲染环境"""
        pass  # PyBullet会自动处理渲染

    def close(self):
        """关闭环境"""
        pass

