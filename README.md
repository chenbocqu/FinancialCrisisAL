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

## 实验过程及相关分析

1. 我们利用传统的方法（KNN）与最新的方法（SO_GAAL，MO_GAAL）作为baseline对Yahoo-S5数据集进行异常检测。Donut+DOPING作为我们的state-of-art实验。

2. 我们进一步讨论在异常检测中，数据量，监督/无监督，数据特性等因子对实验结果的影响，并且用实验加以验证。

3. 详细实验结过程在PPT/报告中陈述。我们在下面列出我们的实验结果。虽然我们的实验结果没有获取特别高的数值，但是对于数据与各种方法的深入探究使得我们受益匪浅。我们也是为数不多（第二个）在这个数据集上进行实验的group。

   ![results](/home/jerrry/Docments/ComputerNetwork/FinancialCrisisAL/presentation/results.png)

## 代码使用

1. 可以通过调用我们的代码对我们的实验结果进行验证。或修改代码进行你自己的实验

2. 使用前提：请先安装下列工具：

   1. PyOD：https://pyod.readthedocs.io/en/latest/install.html
   2. Donut：https://github.com/haowen-xu/donut
   3. matplotlib，sklearn等基础工具

3. 使用代码，可以直接

   ```python
   python main.py
   python ourTest.py
   ```

4. 为了更好使用代码，请仔细阅读代码中的注释。由于较短时间内需要进行较多的实验，我们没有将代码分文件处理（各个实验有太多重复部分）。故而，合理注释或去除注释符号即可复现我们的结果。