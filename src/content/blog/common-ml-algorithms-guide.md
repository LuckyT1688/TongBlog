---
title: 常用机器学习算法大盘点：从原理到代码（去公式版）
description: 别被那些数学公式吓跑了。这篇文章用大白话把最常用的机器学习算法讲清楚，包含有监督、无监督和集成学习
publishedAt: 2025-11-22T10:00:00
updatedAt: 2025-11-22T10:00:00
status: published
featured: true
categories:
  - 机器学习
tags:
  - 算法
  - Python
  - Scikit-learn
  - 面试
draft: false
---

做机器学习久了你会发现，虽然学术界天天发新论文，但在工业界干活，翻来覆去用的其实就是那几个经典算法。

很多初学者容易被数学公式劝退，其实如果不搞理论研究，我们更需要理解的是算法的**直觉（Intuition）**：它是什么逻辑？适合什么场景？代码怎么写？

今天就把我常用的算法箱底翻出来，按**有监督**、**无监督**和**集成学习**分个类，咱们不整虚的，直接上干货。

## 一、有监督学习 (Supervised Learning)

这就像是**老师教学生**。你给模型一堆题（特征）和标准答案（标签），让它学会怎么做题。等它学会了，再给它新题，看它能不能做对。

根据"答案"是连续的数字还是具体的类别，又分为**回归**和**分类**。

### 1. 回归算法 (Regression)

预测一个具体的数值，比如房价、气温、股票收益率。

#### 线性回归 (Linear Regression)

**算法思路：**
想象你要预测房价。你收集了一堆数据：面积、楼层、学区等等。线性回归就是想找一条直线（或者多维空间里的超平面），让这条线尽可能接近所有的数据点。

具体怎么做呢？
1. 先随便画一条线（随机初始化权重）
2. 看看这条线和真实数据点的距离有多远（计算误差）
3. 调整线的斜率和位置，让距离变小（梯度下降）
4. 重复调整，直到误差不再明显下降

最后你得到一个公式：`房价 = 面积×5万 + 楼层×2万 + 学区×10万 + 基础价`

**特点：**
- 简单，可解释性强（权重就是特征重要性）
- 容易欠拟合，对异常值敏感
- 适合线性关系明显的场景

```python
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)

print(model.predict([[6]]))
print(model.coef_, model.intercept_)
```

### 2. 分类算法 (Classification)

预测一个类别，比如邮件是"垃圾"还是"正常"，用户是"流失"还是"留存"。

#### 逻辑回归 (Logistic Regression)

**算法思路：**
名字叫回归，其实是分类算法。比如判断一封邮件是不是垃圾邮件。

怎么做呢？
1. 先用线性回归的方法算出一个分数（比如 3.2）
2. 这个分数可能是任意值，我们需要把它压缩到 0-1 之间表示概率
3. 用 Sigmoid 函数做转换：分数越大，概率越接近 1；分数越小，概率越接近 0
4. 设个阈值（通常是 0.5），大于 0.5 就是"垃圾邮件"，小于 0.5 就是"正常邮件"

训练过程和线性回归类似，也是不断调整权重，让预测的概率和真实标签尽可能接近。

**特点：**
- 工业界最爱，速度快，适合大规模稀疏特征（比如广告点击率预估）
- 输出结果是概率，很方便做阈值调整
- 只能处理线性可分的问题

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

clf = LogisticRegression(random_state=0, max_iter=200)
clf.fit(X, y)

print(clf.predict(X[:2, :]))
print(clf.predict_proba(X[:2, :]))
```

#### 决策树 (Decision Tree)

**算法思路：**
这玩意儿最符合人类思维。就像你相亲时的心理活动：

```
有房吗？
├─ 有 → 帅吗？
│       ├─ 帅 → 见面 ✓
│       └─ 不帅 → 有车吗？
│               ├─ 有 → 见面 ✓
│               └─ 没 → 不见 ✗
└─ 没 → 不见 ✗
```

训练过程：
1. 从所有特征里挑一个"最能分类"的特征（比如有房吗）
2. 按这个特征把数据分成两堆
3. 对每一堆数据，再找下一个最能分类的特征
4. 重复这个过程，树越长越大
5. 到一定程度就停下来（剪枝），不然容易过拟合

怎么判断"最能分类"？用信息增益或基尼系数，简单说就是分完之后每一堆数据越"纯"越好（都是同一类）。

**特点：**
- 可解释性无敌，画个图就能给老板讲清楚
- 不需要对特征做归一化
- 容易过拟合（树长得太茂盛了），需要剪枝

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)

print(clf.predict(X[:2]))
```

#### 支持向量机 (SVM)

**算法思路：**
想象桌子上混杂着红球和蓝球，SVM 要做的事情是：

1. 找一根棍子（超平面），把红球和蓝球分开
2. 不是随便找一根，要找那根让两边的球离棍子最远的（最大间隔）
3. 离棍子最近的那几个球叫"支持向量"，它们决定了棍子的位置

如果桌子上的球混在一起分不开怎么办？
- 用核函数：相当于猛拍桌子，把球震到空中（升维）
- 在三维空间里用一张纸把它们分开
- 常用的核函数：RBF（高斯核）、多项式核

**特点：**
- 小样本下效果很好
- 对核函数的选择敏感，调参麻烦
- 计算慢，不适合大数据集（百万级以上就别想了）

```python
from sklearn.svm import SVC
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

clf = SVC(kernel='rbf', probability=True)
clf.fit(X, y)

print(clf.predict(X[:2]))
```

#### 朴素贝叶斯 (Naive Bayes)

**算法思路：**
判断一封邮件是不是垃圾邮件。贝叶斯的思路是用概率：

1. 先统计历史数据：垃圾邮件有 60%，正常邮件有 40%（先验概率）
2. 看这封邮件里有没有"中奖"这个词：
   - 垃圾邮件里出现"中奖"的概率：80%
   - 正常邮件里出现"中奖"的概率：5%
3. 再看有没有"免费"这个词，算概率
4. 把所有特征（词）的概率乘起来
5. 比较是垃圾邮件的概率 vs 正常邮件的概率，哪个大就判断为哪个

为什么叫"朴素"？因为它假设所有特征（词）之间是独立的。实际上"免费"和"中奖"经常一起出现，但朴素贝叶斯装作它们没关系。

**特点：**
- 速度极快，对缺失数据不敏感
- 文本分类效果很好（垃圾邮件、情感分析）
- 假设太强，特征相关性大的场景效果差

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

clf = GaussianNB()
clf.fit(X, y)

print(clf.predict(X[:2]))
```

#### K-近邻 (KNN)

**算法思路：**
俗称"近朱者赤，近墨者黑"。

假设你搬到一个新小区，想知道这个小区治安好不好：

1. 找离你最近的 K 个邻居（比如 K=5）
2. 看这 5 个邻居里，有几个说"治安好"，有几个说"治安差"
3. 少数服从多数：如果 4 个说好，1 个说差，那就判断"治安好"

具体怎么找"最近"的邻居？
- 计算新样本和所有训练样本的距离（欧氏距离、曼哈顿距离等）
- 找出距离最小的 K 个
- 投票决定类别（分类）或取平均值（回归）

**特点：**
- 简单粗暴，没有训练过程（Lazy Learning）
- 预测时要算和所有样本的距离，数据量大时慢得要死
- K 值的选择很关键：太小容易过拟合，太大容易欠拟合

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

print(neigh.predict(X[:2]))
```

## 二、无监督学习 (Unsupervised Learning)

这就像是**自学**。没有老师，没有标准答案。给模型一堆数据，让它自己找规律，看能不能把相似的数据聚在一起。

### 聚类算法 (Clustering)

#### K-Means

**算法思路：**
假设你要把一群人分成 3 个小组（K=3），让同组的人尽可能相似：

1. **随机选 3 个人当组长**（初始化 K 个聚类中心）
   - 这一步很关键，选得不好结果会很差
   
2. **其他人找最近的组长报道**
   - 计算每个人和 3 个组长的距离
   - 距离哪个组长近就加入哪个组
   
3. **重新选组长**
   - 每个组里的人算个平均位置
   - 这个平均位置就是新的组长（聚类中心）
   
4. **重复步骤 2-3**
   - 有人可能换组（离新组长更近了）
   - 再算新组长
   - 直到没人换组了，或者达到最大迭代次数

5. **结束**
   - 得到 K 个组，每个组有一个中心点

**特点：**
- 简单高效，适合大数据集
- K 值需要自己定（用肘部法则或轮廓系数选）
- 对离群点和初始中心点很敏感
- 只能发现球形的簇，不适合复杂形状

```python
from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])

kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
kmeans.fit(X)

print(kmeans.labels_)
print(kmeans.predict([[0, 0], [12, 3]]))
print(kmeans.cluster_centers_)
```

## 三、集成学习 (Ensemble Learning)

这才是现在比赛和工业界的大杀器。
核心思想是**"三个臭皮匠，顶个诸葛亮"**。把多个弱模型（通常是决策树）组合起来，变成一个强模型。

### 1. Bagging (装袋法)

大家并行干活，最后投票。

#### 随机森林 (Random Forest)

**算法思路：**
一棵决策树容易过拟合，那就种一片森林（几百上千棵树），让大家一起投票。

怎么种这片森林？

1. **随机采样训练数据**（Bootstrap）
   - 有 1000 条数据，每棵树随机抽 1000 条（有放回抽样）
   - 有的数据可能被抽中多次，有的可能一次都没抽中
   
2. **随机选择特征**
   - 有 20 个特征，每次分裂节点时随机选 4-5 个特征
   - 不让所有树都用同样的特征，增加多样性
   
3. **种树**
   - 每棵树用自己的数据和特征长成一棵决策树
   - 每棵树都长得不太一样（因为数据和特征都不同）
   
4. **预测时投票**
   - 分类：100 棵树，60 棵说 A，40 棵说 B → 结果是 A
   - 回归：100 棵树预测的平均值

**特点：**
- 鲁棒性极强，几乎不需要调参就能跑出不错的结果
- 不容易过拟合（多样性降低方差）
- 并行处理，速度快
- 可以算特征重要性

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(X, y)

print(clf.predict(X[:2]))
print(clf.feature_importances_)
```

### 2. Boosting (提升法)

大家串行干活，后一个人专门弥补前一个人的错误。

#### XGBoost / LightGBM

**算法思路：**
和随机森林不同，Boosting 是串行的，一个接一个地种树，后面的树专门弥补前面树的错误。

假设预测房价：

1. **种第一棵树**
   - 预测结果：房价 100 万
   - 真实房价：120 万
   - 误差（残差）：+20 万
   
2. **种第二棵树**
   - 不是预测房价，而是预测第一棵树的误差
   - 第二棵树预测：+15 万
   - 现在总预测：100 + 15 = 115 万
   - 新的误差：+5 万
   
3. **种第三棵树**
   - 预测第二轮的误差：+4 万
   - 总预测：100 + 15 + 4 = 119 万
   - 误差越来越小
   
4. **继续种树**
   - 每棵树都在纠正之前所有树的错误
   - 种几百上千棵树，误差降到很小
   
5. **最终预测**
   - 把所有树的预测加起来（加权求和）

**XGBoost vs LightGBM 的区别：**
- XGBoost：level-wise 生长（一层层长）
- LightGBM：leaf-wise 生长（哪个叶子节点增益大就分裂哪个），速度更快

**特点：**
- 精度极高，Kaggle 比赛刷榜神器
- 速度快（特别是 LightGBM）
- 参数较多，调参需要点经验
- 内置正则化，不容易过拟合

```python
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

print(model.predict(X_test[:2]))
```

## 总结一下

面对一个新任务，我的**起手式**通常是这样的：

1.  **数据量小、追求解释性**：逻辑回归（分类）或 线性回归（回归）。
2.  **数据量适中、追求效果、不想调参**：随机森林。
3.  **比赛、追求极致精度**：XGBoost 或 LightGBM。
4.  **看数据分布**：K-Means 聚类一下看看热闹。

算法没有绝对的好坏，只有适不适合。先把简单的跑通，建立了 Baseline，再上复杂的模型去优化，这才是正路子。
