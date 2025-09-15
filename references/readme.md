

2020年到2025年CCF A类和B类会议上发表的关于Traffic Signal Control的论文整理如下：

| ID   | venue  | year | title                                                        | cite | proposal                                                    | 优势                                       |
| ---- | ------ | ---- | ------------------------------------------------------------ | ---- | ----------------------------------------------------------- | ------------------------------------------ |
| 1    | AAAI   | 2024 | Prompt to Transfer: Sim-to-Real Transfer for Traffic Signal Control with Prompt Learning | 47   |                                                             |                                            |
| 2    | AAAI   | 2024 | π-Light: Programmatic Interpretable Reinforcement Learning for Resource-Limited Traffic Signal Control | 17   |                                                             |                                            |
| 3    | AAAI   | 2023 | SafeLight: A Reinforcement Learning Method toward Collision-free Traffic Signal Control | 54   |                                                             |                                            |
| 4    | AAAI   | 2021 | Hierarchically and Cooperatively Learning Traffic Signal Control | 89   | 使用HRL解决目标不一致和协同问题                             |                                            |
| 5    | AAAI   | 2020 | **Toward A Thousand Lights: Decentralized Deep Reinforcement Learning for Large-Scale Traffic Signal Control** | 489  | 输入车道的压力；奖励为路口的压力负值；网络使用FRAP          | 很好的泛化能力，可以扩展到数千个路口       |
| 6    | AAAI   | 2020 | **MetaLight: Value-Based Meta-Reinforcement Learning for Traffic Signal Control** | 231  | 创新的修改FRAP和元学习算法，提高了新路口agent的训练效率     | 训练效率高/最终模型性能高                  |
| 7    | ICAPS  | 2020 | Online Traffic Signal Control through Sample-Based Constrained Optimization | 6    |                                                             |                                            |
| 8    | ICAPS  | 2020 | Guidelines for Action Space Definition in Reinforcement Learning-Based Traffic Signal Control Systems | ?    |                                                             |                                            |
| 9    | NeurPS | 2024 | DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data | 8    |                                                             |                                            |
| 10   | NeurPS | 2020 | **AttendLight: Universal Attention-Based Reinforcement Learning Model for Traffic Signal Control.** | 127  | 使用加性注意力机制和LSTM实现统一的agent来应对不同的多个路口 | 泛化能力和通用性好，单个路口性能也表现突出 |
| 11   | ICML   | 2022 | Expression might be enough: representing pressure and demand for reinforcement learning based traffic signal control | 72   |                                                             |                                            |
| 12   | IJCAI  | 2024 | X-Light: Cross-City Traffic Signal Control Using Transformer on Transformer as Meta Multi-Agent Reinforcement Learner. | 16   |                                                             |                                            |
| 13   | IJCAI  | 2023 | DenseLight: Efficient Control for Large-scale Traffic Signals with Dense Feedback | 17   |                                                             |                                            |
| 14   | IJCAI  | 2023 | GPLight: Grouped Multi-agent Reinforcement Learning for Large-scale Traffic Signal Control | 40   |                                                             |                                            |
| 15   | IJCAI  | 2023 | Reinforcement Learning Approaches for Traffic Signal Control under Missing Data | 22   |                                                             |                                            |
| 16   | IJCAI  | 2023 | InitLight: Initial Model Generation for Traffic Signal Control Using Adversarial Inverse Reinforcement Learning | 11   |                                                             |                                            |
| 16   | IJCAI  | 2022 | Multi-Agent Reinforcement Learning for Traffic Signal Control through Universal Communication Method | 42   |                                                             |                                            |
| 18   | IJCAI  | 2022 | TinyLight: Adaptive Traffic Signal Control on Devices with Extremely Limited Resources. | 10   |                                                             |                                            |
| 19   | IJCAI  | 2021 | Dynamic Lane Traffic Signal Control with Group Attention and Multi-Timescale Reinforcement Learning | 19   |                                                             |                                            |
| 20   | AAMAS  | 2025 | MacLight: Multi-scene Aggregation Convolutional Learning for Traffic Signal Control. | ?    |                                                             |                                            |
| 21   | AAMAS  | 2025 | FGLight: Learning Neighbor-level Information for Traffic Signal Control | ?    |                                                             |                                            |
| 22   | AAMAS  | 2024 | DuaLight: Enhancing Traffic Signal Control by Leveraging Scenario-Specific and Scenario-Shared Knowledge | 15   |                                                             |                                            |
| 23   | AAMAS  | 2024 | MATLight: Traffic Signal Coordinated Control Algorithm based on Heterogeneous-Agent Mirror Learning with Transformer | 5    |                                                             |                                            |
| 24   | AAMAS  | 2023 | SocialLight: Distributed Cooperation Learning towards Network-Wide Traffic Signal Control | 19   |                                                             |                                            |
| 25   | AAMAS  | 2022 | Fully-Autonomous, Vision-based Traffic Signal Control: From Simulation to Reality | 16   |                                                             |                                            |
| 26   | AAMAS  | 2022 | Reinforcement Learning for Traffic Signal Control Optimization: A Concept for Real-World Implementation | 3    |                                                             |                                            |
| 27   | AAMAS  | 2022 | Multi-agent Traffic Signal Control via Distributed RL with Spatial and Temporal Feature Extraction | 19   |                                                             |                                            |
| 28   | AAMAS  | 2020 | Learning an Interpretable Traffic Signal Control Policy      | 82   |                                                             |                                            |
| 29   | AAMAS  | 2020 | Feudal Multi-Agent Deep Reinforcement Learning for Traffic Signal Control | 99   |                                                             |                                            |
|      |        |      |                                                              |      |                                                             |                                            |
|      |        |      | SOTL                                                         |      |                                                             |                                            |
|      |        |      | Max-pressure                                                 |      |                                                             |                                            |
|      |        |      | FRAP                                                         |      |                                                             |                                            |

来源：“https://dblp.uni-trier.de/search/publ/api?q=traffic signal control&h=1000&format=json”

另外，这个[网站](https://traffic-signal-control.github.io/)有一些经典但偏老（2020年以前）的论文和数据集