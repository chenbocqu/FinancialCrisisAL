# 对抗学习在金融风控中的研究

### 问题描述

在金融风控场景中存在许多用户攻击，欺诈行为。今年来兴起的“薅羊毛”的用户也给公司带来了巨大的经济损失。从**海量数据**中寻找出这些欺诈用户是个急需解决的问题。与此同时，数据具有数量巨大，数据维度高，特征损失严重，人力标签成本大等等问题。本研究旨在结合对抗学习解决金融风控中的上述问题。可以结合所学知识，阅读相关论文，在大规模数据处理，无监督或半监督模型，模型可解释性，模型增量挖掘等等方面提出自己的方法。

### 可研究方向

1. 从海量数据中找出异常数据
2. 对数据进行处理，解决其维度高，特征损失严重的问题
3. 用训练好的模型对数据进行标记，省去人力

### 相关研究工作

以下是可供研究的大方向及其中部分有关工作

#### Unsupevised Anomaly Detection:

1. Unsupervised Anomaly Detection via Variational Auto-Encoderfor Seasonal KPIs in Web Applications (<https://paperswithcode.com/paper/unsupervised-anomaly-detection-via#code>): 

   a. Previous problems: Existing anomaly detection algorithms suffer from the hassle of algorithm picking/parameter tuning, heavy reliance on labels, unsatisfying performance, and/or lack of theoretical foundations.

   b. Proposed method: Donut, an unsupervised anomaly detection algorithm based on Variational Auto-Encoder (a representativedeep generative model) with solid theoretical explanation.

2. DOPING: Generative Data Augmentation for Unsupervised Anomaly Detection with GAN (<https://paperswithcode.com/paper/doping-generative-data-augmentation-for#code>) 

   a. The first data augmentation technique focused on improving performance in unsupervised anomaly detection

#### Group Anomaly Detection

1. <https://paperswithcode.com/paper/group-anomaly-detection-using-deep-generative>. Its goal is to detect anomalous collections of individual data points.
2. 如果论文中没有太多相关的，简单看下即可，与主题不是很相符。

#### Anomaly Detection With Large Dataset (大规模数据处理)

1. Expected Similarity Estimation for Large-Scale Batch and Streaming Anomaly Detection (<https://paperswithcode.com/paper/expected-similarity-estimation-for-large>). 

#### Semi-Supervised Anomaly Detection

1. GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training (<https://paperswithcode.com/paper/ganomaly-semi-supervised-anomaly-detection>): 非常契合咱们的主题，建议细看。

#### 中期前任务：

1. 把这5篇都看了，总结一下作为survey，即可撰写报告。最后应该是4个md（4个类别），记录重要内容即可。
2. 截止日期：**4.28**