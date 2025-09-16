**Expression might be enough-representing pressure and demand for reinforcement learning based traffic signal control**

venue: ICML

year: 2022

### 1、Introduction

这篇论文主要设计并验证了以下几项新东西，以提升大规模交通信号控制的效率与灵活性。

1. Advanced-MP 方法
   1. 在传统最大压强（Max Pressure）策略基础上，同时考虑交叉口内正在通行的车辆和等待队列车辆。
   2. 引入“请求”（request）概念，当积累到一定阈值时触发相位切换决策。
2. 高级交通状态（ATS）
   1. 将压强（pressure）与“有效通行车辆数”结合，构成多维度交通状态表示。
   2. 有效地捕捉不同车流速度和队列长度对相位决策的影响。
3. Advanced-XLight 算法框架
   1. 提供一个通用模板，可将 ATS 嵌入任意强化学习方法。
   2. 基于此模板，衍生出 Advanced-MPLight（基于 Max Pressure）和 Advanced-CoLight（基于图注意力）的两种 RL 算法。
4. 大规模仿真实验与 SOTA 结果
   1. 在多种复杂路网场景下对比，Advanced-MP 与两种 RL 算法均刷新了现有最优性能。
   2. 验证了新状态设计与决策机制在拥堵缓解、通行效率提升上的显著优势。

### 2、Related Work

简单介绍了：

1. 传统TSC方法：FixedTime / MaxPressure / SOTL 
2. RL-based TSC方法：FRAP /  CoLight / PressLight / MPLight / Efficient-CoLight 

### 3 、Preliminaries

给出了TSC问题中的几个概念的清晰定义：

交通网络 / 交通运动流 / 信号相位 / 高效压力 / 相位压力 / 路口压力

并整理了符号表。这部分是论文写得比较严谨和友好的地方。

摘录一下几个不常见的概念：

![image-20250916124821784](img/image-20250916124821784.png)

### 4、Method

#### 4.1 Advanced Max-Pressure

还是一个传统的TSC方法：

![image-20250916130417008](img/image-20250916130417008.png)