

2020年到2025年CCF A类和B类会议上发表的关于Traffic Signal Control的论文整理如下：

| ID   | venue  | year | title                                                        | cite | proposal                                                     | 优势                                       |
| ---- | ------ | ---- | ------------------------------------------------------------ | ---- | ------------------------------------------------------------ | ------------------------------------------ |
| 1    | AAAI   | 2024 | Prompt to Transfer: Sim-to-Real Transfer for Traffic Signal Control with Prompt Learning | 47   | 用GAT解决Sim2Real的鸿沟，利用LLM引入知识向量到GAT，补充特征输入不足的问题。我认为GAT作用有限，因为模拟器底层可能不支持。 | 这个领域比较偏僻一点                       |
| 2    | AAAI   | 2024 | π-Light: Programmatic Interpretable Reinforcement Learning for Resource-Limited Traffic Signal Control | 17   |                                                              |                                            |
| 3    | AAAI   | 2023 | SafeLight: A Reinforcement Learning Method toward Collision-free Traffic Signal Control | 54   | 作者希望训练RL不做出导致危险的动作，但实际上只需要不到100行代码加上三条人工规则即可。是一篇回字有几种写法的论文，而且论文里引用的事故数据也归因错误，事故出现在交叉路口不表示是信号灯导致的。 | 没啥优势                                   |
| 4    | AAAI   | 2021 | Hierarchically and Cooperatively Learning Traffic Signal Control | 89   | 使用HRL解决目标不一致和协同问题                              |                                            |
| 5    | AAAI   | 2020 | **Toward A Thousand Lights: Decentralized Deep Reinforcement Learning for Large-Scale Traffic Signal Control** | 489  | 输入车道的压力；奖励为路口的压力负值；网络使用FRAP。**这个就是被反复提及的MPLight算法** | 很好的泛化能力，可以扩展到数千个路口       |
| 6    | AAAI   | 2020 | **MetaLight: Value-Based Meta-Reinforcement Learning for Traffic Signal Control** | 231  | 创新的修改FRAP和元学习算法，提高了新路口agent的训练效率      | 训练效率高/最终模型性能高                  |
| 7    | ICAPS  | 2020 | Online Traffic Signal Control through Sample-Based Constrained Optimization | 6    |                                                              |                                            |
| 8    | ICAPS  | 2020 | Guidelines for Action Space Definition in Reinforcement Learning-Based Traffic Signal Control Systems | ?    |                                                              |                                            |
| 9    | NeurPS | 2024 | DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data | 8    |                                                              |                                            |
| 10   | NeurPS | 2020 | **AttendLight: Universal Attention-Based Reinforcement Learning Model for Traffic Signal Control.** | 127  | 使用加性注意力机制处理路口的各车道的特征，根据phase2lane相关性得到各相位的特征；然后再次使用注意力机制处理各相位的特征，得到各相位的打分，softmax选择动作 | 泛化能力和通用性好，单个路口性能也表现突出 |
| 11   | ICML   | 2022 | **Expression might be enough: representing pressure and demand for reinforcement learning based traffic signal control** | 72   | 将pressure与“有效通行车辆数”构成状态表示，并应用到已有的MPLight 和CoLight方法上，对MaxPressure方法修改得到三个方法，都取得了更好的性能 | 修改简单，性能收益不错                     |
| 12   | IJCAI  | 2024 | X-Light: Cross-City Traffic Signal Control Using Transformer on Transformer as Meta Multi-Agent Reinforcement Learner. | 16   |                                                              |                                            |
| 13   | IJCAI  | 2023 | DenseLight: Efficient Control for Large-scale Traffic Signals with Dense Feedback | 17   |                                                              |                                            |
| 14   | IJCAI  | 2023 | GPLight: Grouped Multi-agent Reinforcement Learning for Large-scale Traffic Signal Control | 40   |                                                              |                                            |
| 15   | IJCAI  | 2023 | Reinforcement Learning Approaches for Traffic Signal Control under Missing Data | 22   |                                                              |                                            |
| 16   | IJCAI  | 2023 | InitLight: Initial Model Generation for Traffic Signal Control Using Adversarial Inverse Reinforcement Learning | 11   |                                                              |                                            |
| 16   | IJCAI  | 2022 | Multi-Agent Reinforcement Learning for Traffic Signal Control through Universal Communication Method | 42   |                                                              |                                            |
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
| 28   | AAMAS  | 2020 | **Learning an Interpretable Traffic Signal Control Policy**  | 82   | 在DQN训练的同时，训练一个多项式函数来拟合Q网络。因为多项式函数具备可解释可调节的属性 | 让agent的行为具备可解释可手动调节性        |
| 29   | AAMAS  | 2020 | Feudal Multi-Agent Deep Reinforcement Learning for Traffic Signal Control | 99   | FMA2C把整个交通网络分割成多个区域，每个区域一个Manager 和 多个Worker，每个worker管理一个信号灯。 | 更好的全局协调能力，各项指标超过MA2C       |
|      |        |      |                                                              |      |                                                              |                                            |
|      |        | 2013 | Self-Organizing_Traffic_Lights_A_Realistic_Simulate（SOTL）  | 387  | 简单传统方法：当红灯相位累计的等待车辆达到阈值，就切换。     | 简单有效的传统方法                         |
|      |        | 2013 | Max pressure control of a network of signalized intersections（MP） | 692  | 看似简单，实则有点麻烦                                       |                                            |
|      | CIKM   | 2019 | **Learning Phase Competition for Traffic Signal Control（FRAP）** | 292  | 针对TSC问题的巨大状态空间且探索不足 以及每个路口都要训练的问题，巧妙的设计了FRAP网络，对翻转和旋转的状态不变，对不同路口通用。 | 更快收敛，更充分的探索，对不同路口通用     |
|      | CIKM   | 2019 | **CoLight: Learning Network-level Cooperation for Traffic Signal Control** | 480  | 使用图注意力机制，协同相邻的信号灯                           | 更好的协同效果                             |

来源：“https://dblp.uni-trier.de/search/publ/api?q=traffic signal control&h=1000&format=json”

另外，这个[网站](https://traffic-signal-control.github.io/)有一些经典但偏老（2020年以前）的论文和数据集