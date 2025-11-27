---
title: 信贷风控模型评估：从KS到PSI，指标背后的业务逻辑
description: 模型跑完了，KS=0.35，领导问这个数好不好？AUC=0.72呢？模型上线半年后，PSI飙到0.3，是出什么问题了？
publishedAt: 2025-11-27T10:00:00
updatedAt: 2025-11-27T10:00:00
status: published
featured: true
categories:
  - 风控
tags:
  - 机器学习
  - 风控模型
  - 模型评估
  - Python
draft: false
---

去年刚进风控部门时，我训练了个逻辑回归模型，跑完测试集一看：准确率85%。正准备庆祝，老大过来瞄了一眼："准确率没用，KS多少？"

我一脸懵："KS是啥？"

老大叹了口气："你这是分类模型的思路，风控模型不能这么看。"

那次之后我才明白，**风控模型的评估体系和普通分类模型完全不是一回事**。今天就把这套体系掰开了讲清楚。

## 为什么不能用准确率？

先说个真实场景。

假设你有1万个用户申请贷款，其中：
- 9500人会按时还款（好客户）
- 500人会逾期（坏客户）

坏客户占比只有5%（这在信贷场景里很常见）。

现在你训练了个"智能"模型，它的策略很简单：**全部预测为好客户**。

算一下准确率：9500 / 10000 = **95%**

看起来很高对吧？但这个模型毫无价值，因为它根本没有识别出任何一个坏客户。

**这就是风控模型的核心矛盾：**
- 样本极度不均衡（好客户远多于坏客户）
- 关注点不是"预测对了多少"，而是"抓住了多少坏客户，同时不要误伤太多好客户"

所以风控模型有自己的一套评估体系。

## 核心概念：混淆矩阵

先把基础打牢。所有风控指标都基于这个表：

```
                预测
              好客户  坏客户
实际 好客户    TN      FP
    坏客户    FN      TP
```

- **TP (True Positive)**：真阳性，模型预测为坏客户，实际也是坏客户（抓对了）
- **FP (False Positive)**：假阳性，模型预测为坏客户，实际是好客户（误伤）
- **TN (True Negative)**：真阴性，模型预测为好客户，实际也是好客户（放对了）
- **FN (False Negative)**：假阴性，模型预测为好客户，实际是坏客户（漏掉了）

**风控最怕什么？**
- 不怕误伤（FP）：大不了少赚点钱
- **最怕漏掉（FN）**：坏客户被放贷，直接亏钱

## 评估指标体系

### 1. KS值（Kolmogorov-Smirnov）

**KS是风控模型最核心的指标，没有之一。**

#### 它在衡量什么？

KS值衡量的是：**模型区分好坏客户的能力**。

具体来说：
- 把所有客户按模型预测的"坏客户概率"从高到低排序
- 分成10组（或20组），每组10%的人
- 计算每组的"坏客户累计占比"和"好客户累计占比"
- 两条曲线之间的最大距离，就是KS值

#### 直观理解

想象你是贷款审批员，模型给你一份名单，按"可疑程度"从高到低排列。

- **KS值高**：前面全是坏客户，后面全是好客户，一眼就能分开
- **KS值低**：好客户坏客户混在一起，分不清

#### 计算方法

```python
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_ks(y_true, y_pred_proba, n_bins=10):
    df = pd.DataFrame({
        'true': y_true,
        'prob': y_pred_proba
    })
    
    df = df.sort_values('prob', ascending=False).reset_index(drop=True)
    df['bin'] = pd.qcut(df.index, n_bins, labels=False, duplicates='drop')
    
    grouped = df.groupby('bin').agg({
        'true': ['sum', 'count']
    }).reset_index()
    
    grouped.columns = ['bin', 'bad', 'total']
    grouped['good'] = grouped['total'] - grouped['bad']
    
    total_bad = grouped['bad'].sum()
    total_good = grouped['good'].sum()
    
    grouped['bad_rate'] = grouped['bad'] / grouped['total']
    grouped['cum_bad_pct'] = grouped['bad'].cumsum() / total_bad
    grouped['cum_good_pct'] = grouped['good'].cumsum() / total_good
    grouped['ks'] = abs(grouped['cum_bad_pct'] - grouped['cum_good_pct'])
    
    max_ks = grouped['ks'].max()
    max_ks_bin = grouped.loc[grouped['ks'].idxmax(), 'bin']
    
    print(f"最大KS值: {max_ks:.4f}")
    print(f"出现在第 {max_ks_bin + 1} 组")
    print("\n各组详情:")
    print(grouped[['bin', 'bad_rate', 'cum_bad_pct', 'cum_good_pct', 'ks']])
    
    return max_ks, grouped

y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                   1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
y_pred = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
                   0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95, 0.98])

ks, ks_table = calculate_ks(y_true, y_pred)
```

#### KS值怎么判断好坏？

**经验标准：**
- KS < 0.2：模型基本没用
- 0.2 ≤ KS < 0.3：模型一般，能用但效果有限
- 0.3 ≤ KS < 0.4：模型不错，达到可用水平
- 0.4 ≤ KS < 0.5：模型很好
- KS ≥ 0.5：模型非常好（或者数据有问题，比如过拟合）

**我的真实经验：**
- 新客户模型：KS一般在0.25-0.35之间
- 老客户模型：KS可以到0.4-0.5（有历史还款数据）
- KS超过0.6：99%是过拟合了，赶紧检查

### 2. AUC（Area Under Curve）

#### 它在衡量什么？

AUC是ROC曲线下的面积，衡量的是：**随便抽一个坏客户和一个好客户，模型把坏客户评分评得更高的概率**。

- AUC = 0.5：模型等于瞎猜
- AUC = 1.0：模型完美区分

#### 和KS的区别

- **KS**：看的是某个点上的区分能力（实用）
- **AUC**：看的是整体区分能力（学术）

风控实战中，**KS比AUC更重要**，因为：
1. KS能直接对应到业务决策点（批准率多少，坏账率多少）
2. AUC只是一个整体指标，不能指导具体策略

#### 代码实现

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc_and_ks(y_true, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    ks = max(tpr - fpr)
    ks_threshold_idx = np.argmax(tpr - fpr)
    ks_threshold = thresholds[ks_threshold_idx]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
    ax1.set_xlabel('假阳性率 (FPR)')
    ax1.set_ylabel('真阳性率 (TPR)')
    ax1.set_title('ROC曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(thresholds, tpr, label='TPR (坏客户召回率)', lw=2)
    ax2.plot(thresholds, fpr, label='FPR (好客户误伤率)', lw=2)
    ax2.plot(thresholds, tpr - fpr, label='KS = TPR - FPR', lw=2)
    ax2.axvline(ks_threshold, color='red', linestyle='--', alpha=0.5, label=f'最大KS阈值 = {ks_threshold:.3f}')
    ax2.set_xlabel('阈值')
    ax2.set_ylabel('比率')
    ax2.set_title(f'KS曲线 (最大KS = {ks:.4f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"AUC: {roc_auc:.4f}")
    print(f"最大KS: {ks:.4f}")
    print(f"最优阈值: {ks_threshold:.4f}")
    
    return roc_auc, ks, ks_threshold

roc_auc, ks, threshold = plot_roc_and_ks(y_true, y_pred)
```

### 3. PSI（Population Stability Index）

**这是风控最容易被忽视，但最重要的指标。**

#### 它在衡量什么？

PSI衡量的是：**模型上线后，客户群体的分布有没有发生变化**。

想象这个场景：
- 你用2023年的数据训练了个模型
- 2024年拿去线上用
- 结果2024年来申请贷款的人，和2023年完全不一样了

比如：
- 2023年主要是上班族
- 2024年大量个体户涌入

这时候模型就不准了，因为**数据分布变了**。

#### 直观理解

PSI就像体检报告里的"对比上次检查"：
- PSI很小：客户群体没变，模型还能用
- PSI很大：客户群体变了，模型可能失效了

#### 计算方法

```python
def calculate_psi(expected, actual, bins=10):
    def scale_range(data, bins):
        if isinstance(bins, int):
            bins = np.linspace(0, 1, bins + 1)
        
        expected_binned = pd.cut(expected, bins=bins, labels=False, duplicates='drop')
        actual_binned = pd.cut(actual, bins=bins, labels=False, duplicates='drop')
        
        expected_pct = pd.Series(expected_binned).value_counts(normalize=True).sort_index()
        actual_pct = pd.Series(actual_binned).value_counts(normalize=True).sort_index()
        
        psi_df = pd.DataFrame({
            'expected_pct': expected_pct,
            'actual_pct': actual_pct
        }).fillna(0.0001)
        
        psi_df['psi'] = (psi_df['actual_pct'] - psi_df['expected_pct']) * \
                        np.log(psi_df['actual_pct'] / psi_df['expected_pct'])
        
        psi_value = psi_df['psi'].sum()
        
        return psi_value, psi_df
    
    psi_value, psi_detail = scale_range(expected, actual, bins)
    
    print(f"PSI值: {psi_value:.4f}")
    print("\n各分箱详情:")
    print(psi_detail)
    
    if psi_value < 0.1:
        print("\n✓ PSI < 0.1: 分布稳定，模型可继续使用")
    elif psi_value < 0.25:
        print("\n⚠ 0.1 ≤ PSI < 0.25: 分布有轻微变化，需关注")
    else:
        print("\n✗ PSI ≥ 0.25: 分布显著变化，模型需重新训练")
    
    return psi_value

train_scores = np.random.beta(2, 5, 1000)
test_scores_stable = np.random.beta(2, 5, 1000)
test_scores_unstable = np.random.beta(5, 2, 1000)

print("场景1: 分布稳定")
psi_stable = calculate_psi(train_scores, test_scores_stable)

print("\n" + "="*50)
print("\n场景2: 分布变化")
psi_unstable = calculate_psi(train_scores, test_scores_unstable)
```

#### PSI的判断标准

- **PSI < 0.1**：分布非常稳定，模型可以继续用
- **0.1 ≤ PSI < 0.25**：分布有轻微变化，需要密切关注
- **PSI ≥ 0.25**：分布显著变化，模型必须重新训练

#### 实战中的应用

```python
def monitor_psi_monthly(model, train_data, months_data):
    train_scores = model.predict_proba(train_data)[:, 1]
    
    psi_results = []
    
    for month, month_data in months_data.items():
        month_scores = model.predict_proba(month_data)[:, 1]
        psi = calculate_psi(train_scores, month_scores)
        
        psi_results.append({
            'month': month,
            'psi': psi,
            'status': 'stable' if psi < 0.1 else 'warning' if psi < 0.25 else 'alert'
        })
    
    psi_df = pd.DataFrame(psi_results)
    
    plt.figure(figsize=(12, 5))
    colors = psi_df['status'].map({'stable': 'green', 'warning': 'orange', 'alert': 'red'})
    plt.bar(psi_df['month'], psi_df['psi'], color=colors)
    plt.axhline(y=0.1, color='orange', linestyle='--', label='预警线 (0.1)')
    plt.axhline(y=0.25, color='red', linestyle='--', label='危险线 (0.25)')
    plt.xlabel('月份')
    plt.ylabel('PSI')
    plt.title('模型PSI月度监控')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return psi_df
```

### 4. Lift（提升度）

#### 它在衡量什么？

Lift衡量的是：**用模型比不用模型，能提升多少效率**。

举个例子：
- 不用模型，随机抽100个人，能抓到5个坏客户（5%命中率）
- 用模型，按分数从高到低抽100个人，能抓到15个坏客户（15%命中率）
- Lift = 15% / 5% = 3

**Lift = 3 的意思是：用模型能提升3倍效率。**

#### 代码实现

```python
def calculate_lift(y_true, y_pred_proba, n_bins=10):
    df = pd.DataFrame({
        'true': y_true,
        'prob': y_pred_proba
    })
    
    df = df.sort_values('prob', ascending=False).reset_index(drop=True)
    df['bin'] = pd.qcut(df.index, n_bins, labels=False, duplicates='drop')
    
    grouped = df.groupby('bin').agg({
        'true': ['sum', 'count']
    }).reset_index()
    
    grouped.columns = ['bin', 'bad', 'total']
    
    overall_bad_rate = y_true.sum() / len(y_true)
    
    grouped['bad_rate'] = grouped['bad'] / grouped['total']
    grouped['lift'] = grouped['bad_rate'] / overall_bad_rate
    
    print("各组Lift值:")
    print(grouped[['bin', 'bad_rate', 'lift']])
    
    plt.figure(figsize=(10, 5))
    plt.bar(grouped['bin'], grouped['lift'])
    plt.axhline(y=1, color='red', linestyle='--', label='基线 (随机)')
    plt.xlabel('分组 (按预测概率从高到低)')
    plt.ylabel('Lift')
    plt.title('Lift曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return grouped

lift_result = calculate_lift(y_true, y_pred)
```

### 5. 业务指标

这些才是老板真正关心的：

#### Bad Rate（坏账率）

```python
def calculate_bad_rate_by_score(y_true, y_pred_proba, threshold):
    approved = y_pred_proba <= threshold
    
    total_approved = approved.sum()
    bad_approved = (y_true[approved] == 1).sum()
    
    bad_rate = bad_approved / total_approved if total_approved > 0 else 0
    approval_rate = total_approved / len(y_true)
    
    print(f"阈值: {threshold:.3f}")
    print(f"批准率: {approval_rate:.2%}")
    print(f"批准客户中的坏账率: {bad_rate:.2%}")
    
    return bad_rate, approval_rate

bad_rate, approval_rate = calculate_bad_rate_by_score(y_true, y_pred, threshold=0.5)
```

#### 批准率 vs 坏账率曲线

```python
def plot_approval_vs_bad_rate(y_true, y_pred_proba):
    thresholds = np.linspace(0, 1, 100)
    approval_rates = []
    bad_rates = []
    
    for threshold in thresholds:
        approved = y_pred_proba <= threshold
        
        total_approved = approved.sum()
        if total_approved == 0:
            approval_rates.append(0)
            bad_rates.append(0)
            continue
        
        bad_approved = (y_true[approved] == 1).sum()
        
        approval_rate = total_approved / len(y_true)
        bad_rate = bad_approved / total_approved
        
        approval_rates.append(approval_rate)
        bad_rates.append(bad_rate)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(thresholds, approval_rates, 'b-', label='批准率', linewidth=2)
    ax1.set_xlabel('分数阈值')
    ax1.set_ylabel('批准率', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(thresholds, bad_rates, 'r-', label='坏账率', linewidth=2)
    ax2.set_ylabel('坏账率', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title('批准率 vs 坏账率权衡曲线')
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2)
    plt.tight_layout()
    plt.show()

plot_approval_vs_bad_rate(y_true, y_pred)
```

## 完整的模型评估框架

```python
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt

class CreditScoringEvaluator:
    def __init__(self, y_true, y_pred_proba):
        self.y_true = np.array(y_true)
        self.y_pred_proba = np.array(y_pred_proba)
        
    def calculate_all_metrics(self, threshold=0.5):
        y_pred = (self.y_pred_proba >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()
        
        metrics = {
            'AUC': roc_auc_score(self.y_true, self.y_pred_proba),
            'KS': self._calculate_ks(),
            'Accuracy': (tp + tn) / (tp + tn + fp + fn),
            'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'F1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'Bad_Rate': (tp + fn) / len(self.y_true),
        }
        
        return metrics
    
    def _calculate_ks(self, n_bins=10):
        df = pd.DataFrame({
            'true': self.y_true,
            'prob': self.y_pred_proba
        })
        
        df = df.sort_values('prob', ascending=False).reset_index(drop=True)
        df['bin'] = pd.qcut(df.index, n_bins, labels=False, duplicates='drop')
        
        grouped = df.groupby('bin')['true'].agg(['sum', 'count']).reset_index()
        grouped.columns = ['bin', 'bad', 'total']
        grouped['good'] = grouped['total'] - grouped['bad']
        
        total_bad = grouped['bad'].sum()
        total_good = grouped['good'].sum()
        
        grouped['cum_bad_pct'] = grouped['bad'].cumsum() / total_bad
        grouped['cum_good_pct'] = grouped['good'].cumsum() / total_good
        grouped['ks'] = abs(grouped['cum_bad_pct'] - grouped['cum_good_pct'])
        
        return grouped['ks'].max()
    
    def calculate_psi(self, train_proba, bins=10):
        expected_binned = pd.cut(train_proba, bins=bins, labels=False, duplicates='drop')
        actual_binned = pd.cut(self.y_pred_proba, bins=bins, labels=False, duplicates='drop')
        
        expected_pct = pd.Series(expected_binned).value_counts(normalize=True)
        actual_pct = pd.Series(actual_binned).value_counts(normalize=True)
        
        psi_df = pd.DataFrame({
            'expected': expected_pct,
            'actual': actual_pct
        }).fillna(0.0001)
        
        psi_df['psi'] = (psi_df['actual'] - psi_df['expected']) * \
                        np.log(psi_df['actual'] / psi_df['expected'])
        
        return psi_df['psi'].sum()
    
    def plot_comprehensive_report(self):
        fig = plt.figure(figsize=(16, 10))
        
        ax1 = plt.subplot(2, 3, 1)
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba)
        auc_score = roc_auc_score(self.y_true, self.y_pred_proba)
        ax1.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_score:.4f}')
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax1.set_xlabel('FPR')
        ax1.set_ylabel('TPR')
        ax1.set_title('ROC曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(2, 3, 2)
        ks = self._calculate_ks()
        ax2.text(0.5, 0.5, f'KS = {ks:.4f}', 
                ha='center', va='center', fontsize=24, weight='bold')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title('KS值')
        
        ax3 = plt.subplot(2, 3, 3)
        bins = np.linspace(0, 1, 11)
        ax3.hist(self.y_pred_proba[self.y_true == 0], bins=bins, alpha=0.5, label='好客户', density=True)
        ax3.hist(self.y_pred_proba[self.y_true == 1], bins=bins, alpha=0.5, label='坏客户', density=True)
        ax3.set_xlabel('预测概率')
        ax3.set_ylabel('密度')
        ax3.set_title('分数分布')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = plt.subplot(2, 3, 4)
        df = pd.DataFrame({'true': self.y_true, 'prob': self.y_pred_proba})
        df = df.sort_values('prob', ascending=False).reset_index(drop=True)
        df['bin'] = pd.qcut(df.index, 10, labels=False, duplicates='drop')
        lift_data = df.groupby('bin')['true'].mean() / self.y_true.mean()
        ax4.bar(range(len(lift_data)), lift_data)
        ax4.axhline(y=1, color='r', linestyle='--')
        ax4.set_xlabel('分组')
        ax4.set_ylabel('Lift')
        ax4.set_title('Lift曲线')
        ax4.grid(True, alpha=0.3)
        
        ax5 = plt.subplot(2, 3, 5)
        thresholds = np.linspace(0, 1, 100)
        approval_rates = []
        bad_rates = []
        for t in thresholds:
            approved = self.y_pred_proba <= t
            if approved.sum() == 0:
                approval_rates.append(0)
                bad_rates.append(0)
            else:
                approval_rates.append(approved.sum() / len(self.y_true))
                bad_rates.append(self.y_true[approved].sum() / approved.sum())
        
        ax5.plot(thresholds, approval_rates, label='批准率')
        ax5_twin = ax5.twinx()
        ax5_twin.plot(thresholds, bad_rates, 'r', label='坏账率')
        ax5.set_xlabel('阈值')
        ax5.set_ylabel('批准率', color='b')
        ax5_twin.set_ylabel('坏账率', color='r')
        ax5.set_title('批准率 vs 坏账率')
        ax5.grid(True, alpha=0.3)
        
        ax6 = plt.subplot(2, 3, 6)
        metrics = self.calculate_all_metrics()
        metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        ax6.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace', va='center')
        ax6.axis('off')
        ax6.set_title('评估指标汇总')
        
        plt.tight_layout()
        plt.show()

np.random.seed(42)
y_true = np.random.choice([0, 1], size=1000, p=[0.95, 0.05])
y_pred_proba = np.where(y_true == 1, 
                        np.random.beta(5, 2, sum(y_true == 1)),
                        np.random.beta(2, 5, sum(y_true == 0)))

evaluator = CreditScoringEvaluator(y_true, y_pred_proba)

print("完整评估报告")
print("=" * 50)
metrics = evaluator.calculate_all_metrics(threshold=0.5)
for metric, value in metrics.items():
    print(f"{metric:15s}: {value:.4f}")

evaluator.plot_comprehensive_report()
```

## 实战案例：模型上线后的监控

```python
class ModelMonitor:
    def __init__(self, model, train_data):
        self.model = model
        self.train_scores = model.predict_proba(train_data)[:, 1]
        self.baseline_metrics = {}
        
    def evaluate_baseline(self, X_test, y_test):
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        evaluator = CreditScoringEvaluator(y_test, y_pred_proba)
        self.baseline_metrics = evaluator.calculate_all_metrics()
        
        print("基线指标（测试集）:")
        for k, v in self.baseline_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        return self.baseline_metrics
    
    def monthly_check(self, X_month, y_month, month_name):
        y_pred_proba = self.model.predict_proba(X_month)[:, 1]
        
        evaluator = CreditScoringEvaluator(y_month, y_pred_proba)
        current_metrics = evaluator.calculate_all_metrics()
        psi = evaluator.calculate_psi(self.train_scores)
        
        print(f"\n{month_name} 监控报告")
        print("=" * 50)
        print(f"PSI: {psi:.4f}")
        
        if psi >= 0.25:
            print("⚠️  警告: PSI过高，模型可能失效！")
        elif psi >= 0.1:
            print("⚠️  注意: PSI有所升高，需关注")
        else:
            print("✓ PSI正常")
        
        print("\n指标对比:")
        print(f"{'指标':15s} {'基线':>10s} {'当前':>10s} {'变化':>10s}")
        print("-" * 50)
        
        for metric in ['AUC', 'KS', 'Recall', 'Precision']:
            baseline = self.baseline_metrics[metric]
            current = current_metrics[metric]
            change = current - baseline
            change_pct = (change / baseline * 100) if baseline != 0 else 0
            
            status = '✓' if abs(change_pct) < 5 else '⚠️'
            print(f"{metric:15s} {baseline:10.4f} {current:10.4f} {change_pct:9.2f}% {status}")
        
        return psi, current_metrics
```

## 总结

风控模型评估和普通分类模型完全不同，核心要记住：

**训练阶段：**
1. **KS值** - 最核心，看区分能力（目标≥0.3）
2. **AUC** - 参考指标，看整体效果
3. **Lift** - 看业务价值，前几组Lift越高越好

**上线阶段：**
1. **PSI** - 最重要，监控数据漂移（警戒线0.1，危险线0.25）
2. **批准率vs坏账率** - 业务KPI，找平衡点
3. **定期重算KS/AUC** - 看模型是否退化

**老板最关心的：**
- 批准率多少？（影响业务量）
- 坏账率多少？（影响利润）
- 模型什么时候需要重训？（看PSI）

最后说一句：**指标是死的，业务是活的**。不要死盯指标，要理解背后的业务逻辑。KS只有0.25但能控制住坏账率，比KS 0.4但批准率太低要强得多。

---

**相关阅读：**
- 逻辑回归在风控中的应用
- XGBoost调参实战
- 特征工程核心技巧
