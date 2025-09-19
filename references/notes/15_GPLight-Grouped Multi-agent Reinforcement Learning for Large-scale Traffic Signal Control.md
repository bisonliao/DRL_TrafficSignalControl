**GPLight-Grouped Multi-agent Reinforcement Learning for Large-scale Traffic Signal Control**

venue：IJCAI

year：2023

### 1、Introduction

多智能体控制大规模信号灯问题目前没有得到特别好的解决。主要挑战有：

1. 信号灯之间如何协同达到全局最优
2. 信号灯很多，相互之间存在差异，每个都单独训练太复杂，训练一个agent来处理所有信号灯性能又不够好，怎么取得平衡？业界96%的相关论文都集中在 <100个信号灯的规模，缺乏大规模的解决方案

我们的方法通过GCN网络提取出每一个信号灯的特征向量，并对他们进行聚类，相似的信号灯训练一个agent，这样在复杂度和精确度之间取得平衡。我们的方法可以用于多达1089个信号灯的交通场景。

### 2、Related Work

介绍了DRL进行TSC的相关工作

### 3、Preliminaries

介绍了POMDP的相关概念

### 4、The Proposed Model: GPLight

#### 4.1 Feature Extraction

![image-20250919105617402](img/image-20250919105617402.png)

#### 4.2 Group Cohesion

![image-20250919111925251](img/image-20250919111925251.png)

##### 关于MI Loss和Gather Loss的理解

![image-20250919112209508](img/image-20250919112209508.png)

### 4.3 Q-Learning

![image-20250919115904628](img/image-20250919115904628.png)

### 5、Experiments

#### 5.1 实验设置

1. 模拟器：CityFlow。每次切换，会插入3s黄灯和2s全方向的红灯
2. 数据集：人造数据集（10x10， 33x33）和真实的杭州、济南、纽约红绿灯数据
3. 比较方法：
   1. FixedTime
   2. MaxPressure
   3. IntelliLight
   4. CoLight
   5. MPLight
4. 评估指标：平均通行时间， the average time taken by all vehicles during their journey

#### 5.2 实验结果

验证聚类的效果：

![image-20250919132439928](img/image-20250919132439928.png)

整体性能和消融实验

![image-20250919132702358](img/image-20250919132702358.png)

### 6、Conclusion

本论文把相似拓扑和相似动态流量的路口进行聚类，对同一类路口训练一个agent，这样平衡了精确性和复杂度。

未来的研究方向：

1. 处理异构路口
2. 为分组智能体设计专门的 MARL 算法

