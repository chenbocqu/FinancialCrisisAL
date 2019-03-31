# 对抗学习在金融风控中的研究

### 问题描述

在金融风控场景中存在许多用户攻击，欺诈行为。今年来兴起的“薅羊毛”的用户也给公司带来了巨大的经济损失。从**海量数据**中寻找出这些欺诈用户是个急需解决的问题。与此同时，数据具有数量巨大，数据维度高，特征损失严重，人力标签成本大等等问题。本研究旨在结合对抗学习解决金融风控中的上述问题。可以结合所学知识，阅读相关论文，在大规模数据处理，无监督或半监督模型，模型可解释性，模型增量挖掘等等方面提出自己的方法。

### 可研究方向

1. 从海量数据中找出异常数据
2. 对数据进行处理，解决其维度高，特征损失严重的问题
3. 用训练好的模型对数据进行标记，省去人力

### 相关研究工作

以下是可供研究的大方向及其中部分有关工作

#### Anomaly Detection:

1. Efficient GAN-Based Anomaly Detection: This is a CS222 project, which detects abnormal data from a large amount of sequential data. All is available, but not recommended due to its poor dataset and our own model's bad performance.


2. Unsupervised Anomaly Detection via Variational Auto-Encoderfor Seasonal KPIs in Web Applications(<https://arxiv.org/pdf/1802.03903v1.pdf>): 

   a. Previous problems: Existing anomaly detection algorithms suffer from the hassle of algorithm picking/parameter tuning, heavy reliance on labels, unsatisfying performance, and/or lack of theoretical foundations.

   b. Proposed method: Donut, an unsupervised anomaly detection algorithm based on Variational Auto-Encoder (a representativedeep generative model) with solid theoretical explanation.