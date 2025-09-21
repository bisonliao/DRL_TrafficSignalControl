

### 概述

大量的论文是关于**多路口的问题**的：

1. 协同机制：
   1. 有HRL、FuN，利用更上层的策略网络来协同
   2. 有使用Pressure来协同路口的，例如PressLight / MPLight / ExpressMightBeEnough。Pressure考虑了下游的车辆数，是相邻路口之间的重要关系
   3. 注意力机制，例如 AttendLight / UniComm / CoLight，注意力机制天然有路口间相互影响力学习能力。
2. 如何高效训练 / 支持更大规模 / 抽象统一方案
   1. Meta Learning：MetaLight，除了迁移学习，它也使用了FRAP来适配不同拓扑的路口
   2. 更好的网络架构：FRAP，它对翻转和旋转的状态不变，对不同拓扑的路口通用，只需要较少的训练数据；MPLight也使用FRAP，并结合了Pressure机制
   3. 聚类成比较少的场景：GPLight，它使用GCN提取每个路口的嵌入向量并聚类，对这些类别的agent使用QMIX网络优化整体目标



还有一些论文是其他方面的改进：

1. 消除Sim2Real的鸿沟
2. 可解释性：【28】通过训练一个多项式来拟合DQN里的DNN，实现可解释和可调节
3. 提升模型性能： 
   1. ExpressMightBeEnough通过改进状态表示提升性能；
   2. IntelliLight通过引入PhaseGate和Memory Palace提升；
   3. PressLight把Pressure机制引入到RL的state和reward设计中

### 可能的idea

1. 都只考虑了车辆，没有考虑行人/非机动车
2. 如何使用meta learning / offline RL / imitation learning来实现迁移学习，加速多个路口场景下的训练，避免每个路口都要从0开始训练
3. 大模型这么火，有没有可结合的机会
4. GreenWave是什么机制，可否结合到RL中
5. 工程效率的课题：大规模交通网络下，交通灯巨多，怎么更高效的组织数据，避免一个一个交通灯手工写代码做特征工程；怎么大量的收集实际数据？与地图app结合？
6. 与地图app/地图有什么可以结合的？用来采集模拟数据？ 更多的训练用特征（例如碧海片区是睡城/粤海街道是工作区）？



### 具体的论文

2020年到2025年CCF A类和B类会议上发表的关于Traffic Signal Control的论文整理如下：

| ID   | venue  | year | title                                                        | cite | 一句话描述                                                   | 侧重的问题                                 |
| ---- | ------ | ---- | ------------------------------------------------------------ | ---- | ------------------------------------------------------------ | ------------------------------------------ |
| 1    | AAAI   | 2024 | Prompt to Transfer: Sim-to-Real Transfer for Traffic Signal Control with Prompt Learning | 47   | 利用LLM引入知识向量输入到GAT旁路中，补充特征输入，更好的解决Sim2Real的鸿沟问题。我认为GAT作用有限，因为模拟器底层可能不支持。 | Sim2Real                                   |
| 2    | AAAI   | 2024 | π-Light: Programmatic Interpretable Reinforcement Learning for Resource-Limited Traffic Signal Control | 17   |                                                              |                                            |
| 3    | AAAI   | 2023 | SafeLight: A Reinforcement Learning Method toward Collision-free Traffic Signal Control | 54   | 作者希望训练RL不做出导致危险的动作，但实际上只需要不到100行代码加上三条人工规则即可。是一篇回字有几种写法的论文，而且论文里引用的事故数据也归因错误，事故出现在交叉路口不表示是信号灯导致的。所以我不认同这个论文。 | 安全问题                                   |
| 4    | AAAI   | 2021 | Hierarchically and Cooperatively Learning Traffic Signal Control | 89   | 使用HRL解决目标不一致和协同问题。HRL总是很复杂不好复现。     | 多路口协同                                 |
| 5    | AAAI   | 2020 | **Toward A Thousand Lights: Decentralized Deep Reinforcement Learning for Large-Scale Traffic Signal Control** | 489  | 输入车道的压力实现相邻路口协同；奖励也为路口的压力负值；网络使用FRAP使得方案更普适。**这个就是被反复提及的MPLight算法** | 多路口如何协同和如何扩展到过千的路口规模。 |
| 6    | AAAI   | 2020 | **MetaLight: Value-Based Meta-Reinforcement Learning for Traffic Signal Control** | 231  | 创新的修改FRAP和元学习算法，提高了新路口agent的训练效率，让模型可以在不同路口上的迁移 | 大规模交通网络下训练效率问题               |
| 7    | ICAPS  | 2020 | Online Traffic Signal Control through Sample-Based Constrained Optimization | 6    |                                                              |                                            |
| 8    | ICAPS  | 2020 | Guidelines for Action Space Definition in Reinforcement Learning-Based Traffic Signal Control Systems | ?    |                                                              |                                            |
| 9    | NeurPS | 2024 | DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data | 8    |                                                              |                                            |
| 10   | NeurPS | 2020 | **AttendLight: Universal Attention-Based Reinforcement Learning Model for Traffic Signal Control.** | 127  | 使用天生支持变长的加性注意力机制来适配多路口不同拓扑不同相位。先使用注意力机制提取各车道的特征，根据phase2lane相关性得到各相位的特征；然后再次使用注意力机制处理各相位的特征，得到各相位的打分，softmax选择动作 | 多路口场景下解决方案的通用性/泛化能力      |
| 11   | ICML   | 2022 | **Expression might be enough: representing pressure and demand for reinforcement learning based traffic signal control** | 72   | 对既有的三个方法（MPLight/CoLight/MP）的输入状态做增强，即将pressure与“有效通行车辆数”构成状态表示，取得了更好的性能 | 通过改进状态表示提升模型性能               |
| 12   | IJCAI  | 2024 | X-Light: Cross-City Traffic Signal Control Using Transformer on Transformer as Meta Multi-Agent Reinforcement Learner. | 16   |                                                              |                                            |
| 13   | IJCAI  | 2023 | DenseLight: Efficient Control for Large-scale Traffic Signals with Dense Feedback | 17   |                                                              |                                            |
| 14   | IJCAI  | 2023 | GPLight: Grouped Multi-agent Reinforcement Learning for Large-scale Traffic Signal Control | 40   | 大规模交通网络下agent如何训练的问题：GCN对每个路口进行Embedding； Group Cohesion根据路口的embedding向量做聚类 ，同一类的路口只训练一个agent；并使用QMIX网络进行多个Agent间的协同，达到全局最优（QMIX是优化最终的全局损失函数的方案） | 多路口训练效率和协同。                     |
| 15   | IJCAI  | 2023 | Reinforcement Learning Approaches for Traffic Signal Control under Missing Data | 22   |                                                              |                                            |
| 16   | IJCAI  | 2023 | InitLight: Initial Model Generation for Traffic Signal Control Using Adversarial Inverse Reinforcement Learning | 11   |                                                              |                                            |
| 16   | IJCAI  | 2022 | Multi-Agent Reinforcement Learning for Traffic Signal Control through Universal Communication Method | 42   | 抽象出一种通用的相邻路口协同机制UniComm：使用自注意力提取相邻路口间的影响信息，并基于此实现了多路口的UniLight方法。UniComm机制也可以应用到其他TSC方法中 | 多路口协同                                 |
| 18   | IJCAI  | 2022 | TinyLight: Adaptive Traffic Signal Control on Devices with Extremely Limited Resources. | 10   |                                                              |                                            |
| 19   | IJCAI  | 2021 | Dynamic Lane Traffic Signal Control with Group Attention and Multi-Timescale Reinforcement Learning | 19   |                                                              |                                            |
| 20   | AAMAS  | 2025 | MacLight: Multi-scene Aggregation Convolutional Learning for Traffic Signal Control. | ?    |                                                              |                                            |
| 21   | AAMAS  | 2025 | FGLight: Learning Neighbor-level Information for Traffic Signal Control | ?    |                                                              |                                            |
| 22   | AAMAS  | 2024 | DuaLight: Enhancing Traffic Signal Control by Leveraging Scenario-Specific and Scenario-Shared Knowledge | 15   |                                                              |                                            |
| 23   | AAMAS  | 2024 | MATLight: Traffic Signal Coordinated Control Algorithm based on Heterogeneous-Agent Mirror Learning with Transformer | 5    |                                                              |                                            |
| 24   | AAMAS  | 2023 | SocialLight: Distributed Cooperation Learning towards Network-Wide Traffic Signal Control | 19   |                                                              |                                            |
| 25   | AAMAS  | 2022 | Fully-Autonomous, Vision-based Traffic Signal Control: From Simulation to Reality | 16   |                                                              |                                            |
| 26   | AAMAS  | 2022 | Reinforcement Learning for Traffic Signal Control Optimization: A Concept for Real-World Implementation | 3    |                                                              |                                            |
| 27   | AAMAS  | 2022 | Multi-agent Traffic Signal Control via Distributed RL with Spatial and Temporal Feature Extraction | 19   |                                                              |                                            |
| 28   | AAMAS  | 2020 | **Learning an Interpretable Traffic Signal Control Policy**  | 82   | 在DQN训练的同时，训练一个多项式函数来拟合Q网络。因为多项式函数具备可解释可调节的属性 | 可解释性可调节性                           |
| 29   | AAMAS  | 2020 | Feudal Multi-Agent Deep Reinforcement Learning for Traffic Signal Control | 99   | 把类似HRL的FuN和Multi-agent Advantage Actor-Critic）方法结合起来：把整个交通网络分割成多个区域，每个区域一个Manager 和 多个Worker，每个worker管理一个信号灯。 | 多路口协同                                 |
|      |        |      |                                                              |      |                                                              |                                            |
| 30   |        | 2013 | Self-Organizing_Traffic_Lights_A_Realistic_Simulate（SOTL）  | 387  | 简单传统方法：当红灯相位累计的等待车辆达到阈值，就切换。常用作BaseLIne | BaseLine                                   |
| 31   |        | 2013 | Max pressure control of a network of signalized intersections（MP） | 692  | 看似简单，实则有点麻烦，我居然没有看懂细节，需要二刷         | BaseLine                                   |
| 32   | CIKM   | 2019 | **Learning Phase Competition for Traffic Signal Control（FRAP）** | 292  | 针对TSC问题的巨大状态空间且探索不足 以及每个路口都要训练的问题，巧妙的设计了FRAP网络，它对翻转和旋转的状态不变，对不同拓扑的路口通用。 | 解决探索不足性能不好的问题                 |
| 33   | CIKM   | 2019 | **CoLight: Learning Network-level Cooperation for Traffic Signal Control** | 480  | 使用图注意力机制，协同相邻的信号灯。类似NLP中的自注意力，把路口i和它的多个邻居路口j的嵌入向量输入到transformer解码器中，再输入DQN来预测动作。一个交通网络里面数百个路口一起训练一个agent，但分开部署为一个agent负责一个路口。 | 多路口的协同和批量训练                     |
| 34   | KDD    | 2018 | IntelliLight：a Reinforcement Learning Approach for Intelligent Traffic Signal Control | 764  | 引入了 Phase Gate 和 Memory Palace 两个机制，分别解决了（相对动作：切/不切）“相位敏感性不足”和“样本不均衡”问题。 | 通过改进RL方法提升性能                     |
| 35   | KDD    | 2019 | Presslight: Learning max pressure control to coordinate traffic signals in arterial network | 475  | 把MaxPressure的理念用于RL控制方法的state和reward的设计，简单有效。 | 多交叉口自动形成绿波，无需复杂的协同       |

来源：“https://dblp.uni-trier.de/search/publ/api?q=traffic signal control&h=1000&format=json”

另外，这个[网站](https://traffic-signal-control.github.io/)有一些经典但偏老（2020年以前）的论文和数据集



