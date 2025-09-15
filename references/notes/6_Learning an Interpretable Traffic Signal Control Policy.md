**Learning an Interpretable Traffic Signal Control Policy**

venue: AAMAS

year: 2020

### 1、Introduction

基于DRL的TSC解决方案，缺乏可解释性，这对于涉及到交通安全的信号灯控制是一个挑战。

本论文提出了三个DQN变体，他们有更强的可解释性和可调节性，并测试证明他们有不错的性能。

这三个方法是：

1. Deep Regulatable Q–learning (DRQ), 
2. Deep regulatable softmax Q-learning (DRSQ), 
3. Deep regulatable hardmax Q–learning (DRHQ)

### 2、BACKGROUND AND RELATED WORK

介绍了TSC问题、RL、当前RL在解决TSC问题的时候面临的挑战：不可解释、不好直观调整、需要线上训练导致不安全

### 3、PROBLEM DEFINITION

![image-20250915213112093](img/image-20250915213112093.png)

![image-20250915214553934](img/image-20250915214553934.png)