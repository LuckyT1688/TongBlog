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
最基础的算法，假设自变量和因变量之间是线性的。就像你在纸上画散点图，试图用一把直尺画一条直线穿过它们。

**特点：**
- 简单，可解释性强（权重就是特征重要性）。
- 容易欠拟合，对异常值敏感。

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
名字叫回归，其实是分类算法。它在线性回归的基础上套了一个 Sigmoid 函数，把输出压缩到 0 和 1 之间，代表概率。

**特点：**
- 工业界最爱，速度快，适合大规模稀疏特征（比如广告点击率预估）。
- 输出结果是概率，很方便做阈值调整。

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
这玩意儿最符合人类思维。就像你相亲时的心理活动：
"有房吗？" -> 有 -> "帅吗？" -> 帅 -> **见面**
"有房吗？" -> 没 -> **不见**

**特点：**
- 可解释性无敌，画个图就能给老板讲清楚。
- 不需要对特征做归一化。
- 容易过拟合（树长得太茂盛了），需要剪枝。

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)

print(clf.predict(X[:2]))
```

#### 支持向量机 (SVM)
想象桌子上混杂着红球和蓝球，SVM 就是要找一根棍子（超平面），把它们分得越开越好。如果桌子上的球混在一起分不开？那就猛拍桌子，把球震到空中（升维），在三维空间里用一张纸把它们分开。

**特点：**
- 小样本下效果很好。
- 对核函数的选择敏感，计算慢，不适合大数据集。

```python
from sklearn.svm import SVC
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

clf = SVC(kernel='rbf', probability=True)
clf.fit(X, y)

print(clf.predict(X[:2]))
```

#### 朴素贝叶斯 (Naive Bayes)
基于概率论的算法。"朴素"是因为它假设所有特征之间是独立的（虽然现实中很少见）。常用于文本分类，比如垃圾邮件识别。

**特点：**
- 速度极快，对缺失数据不敏感。
- 假设太强，有时候效果不如其他算法。

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

clf = GaussianNB()
clf.fit(X, y)

print(clf.predict(X[:2]))
```

#### K-近邻 (KNN)
俗称"近朱者赤"。要判断一个新样本是什么类别，就看它周围最近的 K 个邻居大部分是什么类别。

**特点：**
- 简单粗暴，没有训练过程（Lazy Learning）。
- 预测时要算和所有样本的距离，数据量大时慢得要死。

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
最经典的聚类算法。
1. 随机选 K 个中心点。
2. 把每个点归到最近的中心点。
3. 重新计算中心点的位置。
4. 重复 2-3，直到中心点不动了。

**特点：**
- 简单高效。
- K 值需要自己定（这是个玄学）。
- 对离群点很敏感。

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
种很多棵决策树。每棵树用的训练数据是随机采样的，用的特征也是随机选的。最后大家投票决定结果。

**特点：**
- 鲁棒性极强，几乎不需要调参就能跑出不错的结果。
- 不容易过拟合。
- 并行处理，速度快。

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
这俩是 Boosting 的巅峰之作。它们不断地生成新的树，每棵新树都是为了拟合之前所有树预测结果的**残差**（误差）。

**特点：**
- 精度极高，Kaggle 比赛刷榜神器。
- 速度快（特别是 LightGBM）。
- 参数较多，调参需要点经验。

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
