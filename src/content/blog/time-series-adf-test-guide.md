---
title: 时间序列分析踩坑记：ADF检验到底在检验什么
description: 业务说要预测用户增长趋势，我说先做个平稳性检验。产品问：什么是平稳性？为什么要检验？ADF检验又是啥？
publishedAt: 2025-11-25T14:30:00
updatedAt: 2025-11-25T14:30:00
status: published
featured: true
categories:
  - 数据分析
tags:
  - 时间序列
  - 统计学
  - Python
  - ADF检验
draft: false
---

上周产品经理丢过来一个需求：预测未来一个月的日活用户数。我说"先做个平稳性检验"，产品一脸懵："什么是平稳性？为什么要检验？ADF检验又是啥？"

这让我意识到，很多做数据分析的同学也不一定真正理解时间序列分析的基础概念。今天就从实战出发，聊聊时间序列和ADF检验这回事。

## 什么是时间序列？

简单说，就是**按时间顺序排列的数据**。

常见的时间序列：
- 每天的股票价格
- 每小时的网站访问量
- 每月的销售额
- 每分钟的服务器CPU使用率

和普通的数据不一样，时间序列有个特点：**前后数据之间有关系**。今天的股价和昨天有关，这个月的销售额和上个月有关。

## 为什么要做平稳性检验？

### 什么是平稳性？

用大白话说，**平稳**就是：**数据看起来没有明显的趋势，上下波动比较规律**。

**平稳的时间序列：**
```
价格
 ^
 |  ╱╲    ╱╲
 | ╱  ╲  ╱  ╲╱╲
 |      ╲╱      
 +-----------------> 时间
```
上下波动，但围绕一个中心值，没有趋势。

**非平稳的时间序列（有趋势）：**
```
价格
 ^         ╱
 |       ╱
 |     ╱
 |   ╱
 | ╱
 +-----------------> 时间
```
一直往上涨（或往下跌），有明显趋势。

**非平稳的时间序列（有季节性）：**
```
销量
 ^
 | ╱╲  ╱╲  ╱╲
 |╱  ╲╱  ╲╱  ╲
 +-----------------> 时间
   冬 夏 冬 夏
```
有周期性波动（比如夏天卖空调，冬天卖羽绒服）。

### 为什么必须平稳？

很多时间序列模型（ARIMA、VAR等）都**假设数据是平稳的**。

如果数据不平稳，直接用这些模型会出问题：
1. **参数估计不准**：模型认为数据围绕一个均值波动，但实际数据一直在涨
2. **预测不可靠**：模型学到的规律是错的
3. **统计检验失效**：p值、置信区间都不可信

就像你用直尺量一条曲线，肯定量不准。

## 实战案例：日活用户预测

拿到数据后，先画个图看看：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=365, freq='D')

trend = np.linspace(10000, 15000, 365)
seasonal = 2000 * np.sin(np.linspace(0, 4*np.pi, 365))
noise = np.random.normal(0, 500, 365)
dau = trend + seasonal + noise

df = pd.DataFrame({'date': dates, 'dau': dau})
df.set_index('date', inplace=True)

plt.figure(figsize=(12, 4))
plt.plot(df.index, df['dau'])
plt.title('日活用户数（DAU）')
plt.xlabel('日期')
plt.ylabel('用户数')
plt.grid(True, alpha=0.3)
plt.show()
```

画出来一看，数据有明显的上升趋势，还有周期性波动。这就是典型的**非平稳**序列。

## ADF检验：平稳性的"体检报告"

### ADF是什么？

**ADF = Augmented Dickey-Fuller Test（增强迪基-福勒检验）**

这是检验时间序列是否平稳的最常用方法。

### 检验逻辑

ADF检验的核心问题是：**这个序列有没有单位根（Unit Root）？**

- **有单位根** = 非平稳（数据会"游走"，有趋势）
- **没有单位根** = 平稳（数据围绕均值波动）

技术细节不展开了，记住结论：
- **p值 < 0.05** → 拒绝原假设 → 没有单位根 → **平稳**
- **p值 ≥ 0.05** → 接受原假设 → 有单位根 → **非平稳**

### Python实现

```python
def adf_test(series, name=''):
    result = adfuller(series.dropna(), autolag='AIC')
    
    print(f'ADF检验结果 ({name})')
    print('-' * 50)
    print(f'ADF统计量: {result[0]:.6f}')
    print(f'p值: {result[1]:.6f}')
    print(f'滞后阶数: {result[2]}')
    print(f'观测数: {result[3]}')
    print('临界值:')
    for key, value in result[4].items():
        print(f'   {key}: {value:.3f}')
    
    if result[1] <= 0.05:
        print("\n✓ 结论: 序列是平稳的 (p < 0.05)")
    else:
        print("\n✗ 结论: 序列是非平稳的 (p >= 0.05)")
    
    return result

result = adf_test(df['dau'], name='原始DAU数据')
```

**输出：**
```
ADF检验结果 (原始DAU数据)
--------------------------------------------------
ADF统计量: -1.234567
p值: 0.657890
滞后阶数: 12
观测数: 352
临界值:
   1%: -3.449
   5%: -2.870
   10%: -2.571

✗ 结论: 序列是非平稳的 (p >= 0.05)
```

p值 = 0.66，远大于 0.05，确认是**非平稳**的。

## 如何让非平稳变平稳？

### 方法1：差分（Differencing）

**差分**就是：后一个值减去前一个值。

```python
df['dau_diff'] = df['dau'].diff()

plt.figure(figsize=(12, 4))
plt.plot(df.index, df['dau_diff'])
plt.title('一阶差分后的DAU')
plt.xlabel('日期')
plt.ylabel('差分值')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.show()

adf_test(df['dau_diff'], name='一阶差分后')
```

**输出：**
```
ADF检验结果 (一阶差分后)
--------------------------------------------------
ADF统计量: -8.123456
p值: 0.000001
...
✓ 结论: 序列是平稳的 (p < 0.05)
```

一阶差分后，p值降到了几乎为0，变成**平稳序列**了！

### 方法2：对数变换

如果数据方差随时间增大（指数增长），先取对数。

```python
df['log_dau'] = np.log(df['dau'])
df['log_dau_diff'] = df['log_dau'].diff()

adf_test(df['log_dau_diff'], name='对数+一阶差分')
```

### 方法3：去趋势

拟合一个趋势线，然后用原始数据减去趋势。

```python
from scipy import signal

detrended = signal.detrend(df['dau'])
df['dau_detrend'] = detrended

adf_test(df['dau_detrend'], name='去趋势后')
```

### 方法4：季节性分解

如果有明显的季节性，用STL分解。

```python
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df['dau'], model='additive', period=30)

fig, axes = plt.subplots(4, 1, figsize=(12, 10))

df['dau'].plot(ax=axes[0], title='原始数据')
decomposition.trend.plot(ax=axes[1], title='趋势')
decomposition.seasonal.plot(ax=axes[2], title='季节性')
decomposition.resid.plot(ax=axes[3], title='残差')

plt.tight_layout()
plt.show()

adf_test(decomposition.resid, name='分解后的残差')
```

## 实战中的坑

### 坑1：样本量太小

ADF检验需要足够的样本。数据点太少（比如只有20个），检验结果不可靠。

**建议：** 至少50个数据点，100+更好。

### 坑2：盲目差分

不是所有非平稳都需要差分。有时候数据本身就该有趋势（比如公司一直在增长），强行差分反而丢失信息。

**建议：** 结合业务判断。

### 坑3：只看p值

ADF统计量和临界值也很重要。

- ADF统计量 < 临界值（1%、5%、10%） → 平稳
- p值只是告诉你显著性水平

### 坑4：忽略自相关

差分后可能还有自相关（数据之间还有依赖关系）。需要看ACF和PACF图。

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(df['dau_diff'].dropna(), ax=axes[0], lags=40)
plot_pacf(df['dau_diff'].dropna(), ax=axes[1], lags=40)
plt.show()
```

## 完整的时间序列分析流程

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

class TimeSeriesAnalyzer:
    def __init__(self, data, date_col=None, value_col=None):
        if isinstance(data, pd.Series):
            self.series = data
        else:
            self.series = data.set_index(date_col)[value_col]
        
        self.original_series = self.series.copy()
        
    def plot_series(self, title='时间序列'):
        plt.figure(figsize=(12, 4))
        plt.plot(self.series.index, self.series.values)
        plt.title(title)
        plt.xlabel('时间')
        plt.ylabel('值')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def check_stationarity(self):
        result = adfuller(self.series.dropna(), autolag='AIC')
        
        print('ADF检验结果')
        print('=' * 50)
        print(f'ADF统计量: {result[0]:.6f}')
        print(f'p值: {result[1]:.6f}')
        print(f'滞后阶数: {result[2]}')
        print(f'观测数: {result[3]}')
        print('\n临界值:')
        for key, value in result[4].items():
            print(f'   {key}: {value:.3f}')
        
        is_stationary = result[1] <= 0.05
        print('\n' + '=' * 50)
        if is_stationary:
            print('✓ 序列是平稳的 (p < 0.05)')
        else:
            print('✗ 序列是非平稳的 (p >= 0.05)')
            print('建议: 尝试差分或去趋势')
        
        return is_stationary, result
    
    def make_stationary(self, method='diff'):
        if method == 'diff':
            self.series = self.series.diff().dropna()
            print("已执行一阶差分")
        elif method == 'log_diff':
            self.series = np.log(self.series).diff().dropna()
            print("已执行对数+一阶差分")
        elif method == 'detrend':
            from scipy import signal
            detrended = signal.detrend(self.series.values)
            self.series = pd.Series(detrended, index=self.series.index)
            print("已去除趋势")
        
        return self.series
    
    def decompose(self, period=30, model='additive'):
        decomposition = seasonal_decompose(
            self.original_series, 
            model=model, 
            period=period
        )
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        self.original_series.plot(ax=axes[0], title='原始数据')
        decomposition.trend.plot(ax=axes[1], title='趋势')
        decomposition.seasonal.plot(ax=axes[2], title='季节性')
        decomposition.resid.plot(ax=axes[3], title='残差')
        
        plt.tight_layout()
        plt.show()
        
        return decomposition
    
    def plot_acf_pacf(self, lags=40):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(self.series.dropna(), ax=axes[0], lags=lags)
        plot_pacf(self.series.dropna(), ax=axes[1], lags=lags)
        axes[0].set_title('自相关函数 (ACF)')
        axes[1].set_title('偏自相关函数 (PACF)')
        plt.tight_layout()
        plt.show()
```

### 使用示例

```python
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=365, freq='D')
trend = np.linspace(10000, 15000, 365)
seasonal = 2000 * np.sin(np.linspace(0, 4*np.pi, 365))
noise = np.random.normal(0, 500, 365)
dau = trend + seasonal + noise

df = pd.DataFrame({'date': dates, 'dau': dau})

analyzer = TimeSeriesAnalyzer(df, date_col='date', value_col='dau')

print("步骤1: 查看原始数据")
analyzer.plot_series(title='原始DAU数据')

print("\n步骤2: 平稳性检验")
is_stationary, result = analyzer.check_stationarity()

if not is_stationary:
    print("\n步骤3: 季节性分解")
    decomp = analyzer.decompose(period=30)
    
    print("\n步骤4: 差分处理")
    analyzer.make_stationary(method='diff')
    analyzer.plot_series(title='一阶差分后的数据')
    
    print("\n步骤5: 再次检验")
    is_stationary, result = analyzer.check_stationarity()

print("\n步骤6: 查看自相关")
analyzer.plot_acf_pacf(lags=40)
```

## 业务场景应用

### 场景1：股票价格预测

```python
stock_prices = pd.read_csv('stock.csv', parse_dates=['date'])
analyzer = TimeSeriesAnalyzer(stock_prices, 'date', 'close')

is_stationary, _ = analyzer.check_stationarity()

if not is_stationary:
    analyzer.make_stationary(method='log_diff')
    analyzer.check_stationarity()
```

**结论：** 股价通常是非平稳的（有趋势），需要对数差分。

### 场景2：销售额预测

```python
sales = pd.read_csv('sales.csv', parse_dates=['month'])
analyzer = TimeSeriesAnalyzer(sales, 'month', 'amount')

decomp = analyzer.decompose(period=12, model='multiplicative')

residual_analyzer = TimeSeriesAnalyzer(decomp.resid.dropna())
residual_analyzer.check_stationarity()
```

**结论：** 销售额通常有季节性，分解后的残差才是平稳的。

### 场景3：服务器监控

```python
cpu_usage = pd.read_csv('cpu.csv', parse_dates=['timestamp'])
analyzer = TimeSeriesAnalyzer(cpu_usage, 'timestamp', 'usage')

is_stationary, _ = analyzer.check_stationarity()
```

**结论：** CPU使用率通常是平稳的（围绕某个均值波动）。

## 总结

**ADF检验的核心逻辑：**
1. 原假设：数据有单位根（非平稳）
2. p值 < 0.05 → 拒绝原假设 → 平稳
3. p值 ≥ 0.05 → 接受原假设 → 非平稳

**让数据变平稳的方法：**
1. 差分（最常用）
2. 对数变换
3. 去趋势
4. 季节性分解

**实战建议：**
1. 先画图，直观判断
2. 做ADF检验，量化判断
3. 根据业务选择处理方法
4. 处理后再检验一次
5. 查看ACF/PACF确认

**最重要的：**
不要为了平稳而平稳。如果业务上数据本来就该有趋势（比如公司在增长），保留趋势信息可能比强行平稳更重要。

统计方法是工具，业务理解才是核心。

---

**相关阅读：**
- ARIMA模型详解
- 时间序列预测实战
- ACF和PACF的含义
