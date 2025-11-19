---
title: "机器学习模型评估指标全解析：从混淆矩阵到 ROC 曲线"
description: "深入浅出地讲解分类模型和回归模型的各类评估指标，帮助你在实际项目中选择合适的评估方法"
publishedAt: 2025-11-17T11:30:27
updatedAt: 2025-11-17T11:30:27
author: "Tong"
tags: ["机器学习", "模型评估", "数据分析", "Python"]
categories: ["机器学习"]
status: published
draft: false
complexity: 2
---

## 写在前面

做过几个项目后发现，很多时候模型训练完了，面对一堆指标却不知道该看哪个。准确率 95%，看起来很不错？但实际业务中可能完全不够用。这篇文章就来聊聊这些评估指标到底在说什么，什么时候该用哪个。

## 一、分类模型评估指标

### 1.1 混淆矩阵：一切的基础

先从最基础的混淆矩阵说起。假设你在做一个垃圾邮件检测器：

|          | 预测：正常邮件 | 预测：垃圾邮件 |
|----------|--------------|--------------|
| **实际：正常邮件** | TN (真负例) | FP (假正例) |
| **实际：垃圾邮件** | FN (假负例) | TP (真正例) |

- **TP (True Positive)**：模型说是垃圾邮件，实际也是 ✅
- **TN (True Negative)**：模型说不是垃圾邮件，实际也不是 ✅
- **FP (False Positive)**：模型说是垃圾邮件，但其实不是 ❌ (误报)
- **FN (False Negative)**：模型说不是垃圾邮件，但其实是 ❌ (漏报)

记住这四个，后面所有指标都是从这里派生出来的。

### 1.2 准确率 (Accuracy)：最直观但有时会骗人

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

就是"预测对的占总数的比例"。看起来很直观，但有个致命问题：**样本不平衡时会失效**。

举个例子，信用卡欺诈检测中，欺诈交易可能只占 0.1%。你训练个模型，什么都不做，把所有交易都标记为"正常"，准确率立马 99.9%！但这模型有啥用？

```python
from sklearn.metrics import accuracy_score

y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

accuracy = accuracy_score(y_true, y_pred)
print(f"准确率: {accuracy:.2f}")  # 0.80
```

**什么时候用？** 样本比较均衡，且各类错误的代价差不多时。

### 1.3 精确率 (Precision)：我说的有多靠谱

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

"在我预测为正例的样本中，有多少是真的正例？" 

回到垃圾邮件的例子，高精确率意味着：模型标记为垃圾邮件的，基本都是真垃圾邮件，不会误伤正常邮件。

**什么时候用？** 当误报代价很高时。比如：
- 医疗诊断：误诊健康人为患病（FP）会造成恐慌和不必要的治疗
- 推荐系统：推荐垃圾内容给用户会影响体验

```python
from sklearn.metrics import precision_score

precision = precision_score(y_true, y_pred)
print(f"精确率: {precision:.2f}")
```

### 1.4 召回率 (Recall)：我能找到多少

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

"在所有真正的正例中，我找到了多少？"

高召回率意味着：真正的垃圾邮件，模型基本都能抓出来，不会漏掉太多。

**什么时候用？** 当漏报代价很高时。比如：
- 癌症筛查：漏掉一个真实病例（FN）可能致命
- 欺诈检测：漏掉一笔欺诈交易损失巨大

```python
from sklearn.metrics import recall_score

recall = recall_score(y_true, y_pred)
print(f"召回率: {recall:.2f}")
```

### 1.5 F1-Score：精确率和召回率的平衡

$$
F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

精确率和召回率往往是矛盾的。你想提高召回率，就得把判断标准放宽，但这样精确率就下降了；反之亦然。

F1-Score 就是这两者的调和平均数，适合同时关注精确率和召回率的场景。

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred)
print(f"F1分数: {f1:.2f}")
```

> **小技巧**：如果你更看重召回率，可以用 $F_2$；更看重精确率，用 $F_{0.5}$。

### 1.6 ROC 曲线和 AUC：看模型的全貌

前面的指标都基于一个固定的分类阈值（通常是 0.5）。但阈值不同，结果就不同。ROC 曲线就是把所有可能的阈值都试一遍。

- **横轴 (FPR)**：假正例率 = $\frac{FP}{FP + TN}$ (误报了多少正常样本)
- **纵轴 (TPR)**：真正例率 = $\frac{TP}{TP + FN}$ (就是召回率)

**AUC (Area Under Curve)** 就是 ROC 曲线下的面积：
- AUC = 1：完美分类器
- AUC = 0.5：随机猜测（等于抛硬币）
- AUC < 0.5：比随机还差，赶紧检查代码

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 假设 y_score 是模型输出的概率
y_score = [0.1, 0.9, 0.8, 0.2, 0.7]

# 计算 AUC
auc = roc_auc_score(y_true, y_score)
print(f"AUC: {auc:.3f}")

# 绘制 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_true, y_score)
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

**什么时候用？** 样本不平衡、需要调整分类阈值、或者比较不同模型的整体性能时。

### 1.7 PR 曲线：样本不平衡时的好朋友

当正负样本严重不平衡时，ROC 曲线可能过于"乐观"（因为负样本太多，TN 很容易高，FPR 就低了）。这时候看 PR 曲线更靠谱。

- **横轴**：召回率 (Recall)
- **纵轴**：精确率 (Precision)

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_true, y_score)
ap = average_precision_score(y_true, y_score)

plt.plot(recall, precision, label=f'PR curve (AP = {ap:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
```

### 1.8 Cohen's Kappa：考虑随机因素

有时候即使瞎猜，准确率也能很高（比如样本极度不平衡时）。Kappa 系数通过扣除"随机一致性"来评估模型的真实表现。

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

- $p_o$：实际一致率（准确率）
- $p_e$：期望一致率（随机情况下的一致率）

```python
from sklearn.metrics import cohen_kappa_score

kappa = cohen_kappa_score(y_true, y_pred)
print(f"Kappa: {kappa:.3f}")
```

- κ > 0.8：几乎完全一致
- 0.6 < κ < 0.8：高度一致
- 0.4 < κ < 0.6：中度一致
- κ < 0.4：一致性较差

---

## 二、回归模型评估指标

回归问题预测的是连续值，评估方式和分类完全不同。

### 2.1 MAE (Mean Absolute Error)：平均绝对误差

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
$$

就是"预测值和真实值差了多少，取绝对值，再求平均"。

**优点**：
- 直观，单位和原始数据一致（比如预测房价，MAE = 50000，就是平均差5万元）
- 对异常值不敏感

**缺点**：
- 不够"惩罚"大误差

```python
from sklearn.metrics import mean_absolute_error

y_true = [3.0, 2.5, 4.0, 7.0]
y_pred = [2.8, 2.4, 4.2, 6.5]

mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.3f}")
```

### 2.2 MSE 和 RMSE：均方误差和均方根误差

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

$$
\text{RMSE} = \sqrt{\text{MSE}}
$$

MSE 对误差进行平方，意味着：
- **大误差会被严厉惩罚**（误差10的平方是100，误差1的平方才1）
- 适合你不能容忍大偏差的场景

RMSE 是 MSE 开根号，好处是单位和原始数据一致，更容易解释。

```python
from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
```

**什么时候用？**
- 需要重点关注大误差时用 MSE/RMSE
- 希望指标和原始数据单位一致时用 RMSE

### 2.3 R² (决定系数)：模型解释了多少变化

$$
R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}
$$

- $\bar{y}$ 是真实值的平均数

R² 表示"模型预测解释了目标变量多少比例的方差"：
- R² = 1：完美预测
- R² = 0：模型和直接用平均值一样差
- R² < 0：模型比平均值还差（这就该检查代码了）

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.3f}")
```

**注意**：R² 在多元回归中有个问题——加入更多特征，即使是垃圾特征，R² 也会增加。这时候要看调整后的 R²（Adjusted R²）。

### 2.4 MAPE (Mean Absolute Percentage Error)：相对误差

$$
\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|
$$

用百分比表示误差，适合比较不同量级的数据。

**优点**：
- 易于理解（比如 MAPE = 5%，就是平均偏差 5%）
- 量纲无关，可以跨数据集比较

**缺点**：
- 当真实值接近 0 时会爆炸
- 对低估的惩罚比高估重

```python
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"MAPE: {mape:.2f}%")
```

### 2.5 Huber Loss：MAE 和 MSE 的折中

Huber Loss 在误差小时用 MSE（平滑可导），误差大时用 MAE（抗异常值）：

$$
L_\delta(y, \hat{y}) = 
\begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta \cdot (|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}
$$

适合数据中有异常值，但又不想完全忽略它们的场景。

---

## 三、实际项目中怎么选？

### 分类任务选择指南

| 场景 | 推荐指标 | 原因 |
|------|---------|------|
| 样本均衡，各类错误代价相同 | Accuracy | 简单直观 |
| 样本不平衡 | F1-Score, AUC | 综合考虑精确率和召回率 |
| 误报代价高（不能乱报警） | Precision | 提高查准率 |
| 漏报代价高（不能漏掉真实案例） | Recall | 提高查全率 |
| 需要调整阈值 | ROC-AUC | 看模型整体性能 |
| 极度不平衡（如欺诈检测） | PR-AUC | ROC 可能过于乐观 |

### 回归任务选择指南

| 场景 | 推荐指标 | 原因 |
|------|---------|------|
| 需要直观的误差值 | MAE 或 RMSE | 单位和原始数据一致 |
| 不能容忍大误差 | RMSE 或 MSE | 平方项惩罚大误差 |
| 比较模型解释能力 | R² | 看模型解释了多少方差 |
| 跨数据集比较 | MAPE | 量纲无关 |
| 数据有异常值 | MAE 或 Huber | 更稳健 |

---

## 四、常见误区

### 误区 1：只看一个指标

很多新手训练完模型，就盯着准确率或者 R² 看。实际上，**没有一个指标能完整描述模型表现**。

比如你做信用评分模型，准确率 98%，看起来很棒。但仔细一看，召回率只有 30%，意味着 70% 的违约客户你都没抓到——这模型基本没用。

**建议**：至少看 3 个指标，比如分类任务看 Precision、Recall、F1；回归任务看 MAE、RMSE、R²。

### 误区 2：忽略业务场景

技术指标再漂亮，不符合业务需求也是白搭。

假设你做推荐系统，召回率 99%，但精确率只有 1%。用户点开 100 个推荐，只有 1 个是感兴趣的——这体验能好吗？反过来，精确率 99%，召回率 1%，推荐又太保守，用户永远看不到新内容。

**建议**：和业务方确认清楚，误报和漏报哪个代价更高，再决定优化方向。

### 误区 3：过度拟合评估集

有些人为了让指标好看，反复调参，直到测试集上的指标"完美"。结果模型上线后一塌糊涂——过拟合了。

**建议**：
1. 训练集、验证集、测试集严格分开
2. 只用训练集训练，验证集调参，测试集最后评估一次
3. 有条件的话，用交叉验证

---

## 五、代码实战：完整评估流程

```python
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import numpy as np

# ========== 分类任务 ==========
print("=" * 50)
print("分类模型评估")
print("=" * 50)

# 生成数据
X, y = make_classification(n_samples=1000, n_features=20, 
                          n_informative=15, n_redundant=5, 
                          random_state=42, weights=[0.9, 0.1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 训练模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)
y_score = clf.predict_proba(X_test)[:, 1]

# 评估
print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

print("\n分类报告:")
print(classification_report(y_test, y_pred, 
                          target_names=['负例', '正例']))

print(f"\nROC-AUC: {roc_auc_score(y_test, y_score):.3f}")
print(f"PR-AUC: {average_precision_score(y_test, y_score):.3f}")

# ========== 回归任务 ==========
print("\n" + "=" * 50)
print("回归模型评估")
print("=" * 50)

# 生成数据
X, y = make_regression(n_samples=1000, n_features=20, 
                      noise=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 训练模型
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)

# 预测
y_pred = reg.predict(X_test)

# 评估
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nMAE: {mae:.3f}")
print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")

# 可视化预测结果
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', lw=2)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title(f'回归预测结果 (R² = {r2:.3f})')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('regression_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n预测结果图已保存为 regression_results.png")
```

---

## 总结

评估指标没有"最好"的，只有"最合适"的。记住几个要点：

1. **分类任务**：先看混淆矩阵，再根据业务需求选 Precision、Recall 还是 F1
2. **回归任务**：MAE 直观，RMSE 惩罚大误差，R² 看解释能力
3. **样本不平衡**：别只看 Accuracy，多看 F1、AUC、PR 曲线
4. **业务优先**：技术指标服务业务目标，不是为了好看而好看

希望这篇文章能帮你在实际项目中少走弯路。下次训练模型时，记得多看几个指标，结合业务场景综合判断。

有问题欢迎留言讨论！

---

**参考资料**：
- [Scikit-learn 官方文档 - Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Understanding ROC Curves](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
- [The Relationship Between Precision-Recall and ROC Curves](https://www.biostat.wisc.edu/~page/rocpr.pdf)
