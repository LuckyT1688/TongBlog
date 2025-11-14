---
title: "量化投资策略回测框架搭建"
description: "从零开始构建一个简单但完整的量化投资策略回测系统，包括数据准备、策略实现、性能评估等"
publishedAt: 2025-11-13
categories: ["量化投资"]
tags: ["Python", "回测", "量化策略", "投资组合"]
featured: true
status: published
draft: false
---

在量化投资中，回测是验证策略有效性的重要环节。本文将带你从零开始搭建一个简单的回测框架。

## 回测框架的核心组件

一个完整的回测系统通常包含以下模块：

1. **数据层**：获取和管理历史数据
2. **策略层**：定义交易逻辑
3. **执行层**：模拟订单执行
4. **评估层**：计算性能指标

## 框架设计

### 1. 数据管理器

```python
import pandas as pd
import numpy as np
from datetime import datetime

class DataHandler:
    """数据处理类"""
    
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        
    def load_data(self, filepath):
        """加载历史数据"""
        self.data = pd.read_csv(
            filepath,
            parse_dates=['date'],
            index_col='date'
        )
        self.data = self.data.loc[self.start_date:self.end_date]
        return self.data
    
    def get_latest_bar(self, N=1):
        """获取最新N条数据"""
        return self.data.iloc[-N:]
```

### 2. 策略基类

```python
class Strategy:
    """策略基类"""
    
    def __init__(self, data):
        self.data = data
        self.signals = pd.DataFrame(index=data.index)
        self.signals['signal'] = 0
        
    def generate_signals(self):
        """生成交易信号 - 子类需要重写此方法"""
        raise NotImplementedError("需要实现 generate_signals 方法")
```

### 3. 双均线策略示例

```python
class MovingAverageCrossStrategy(Strategy):
    """双均线交叉策略"""
    
    def __init__(self, data, short_window=5, long_window=20):
        super().__init__(data)
        self.short_window = short_window
        self.long_window = long_window
        
    def generate_signals(self):
        """生成交易信号"""
        # 计算短期和长期均线
        self.signals['short_ma'] = self.data['close'].rolling(
            window=self.short_window
        ).mean()
        self.signals['long_ma'] = self.data['close'].rolling(
            window=self.long_window
        ).mean()
        
        # 生成信号：短期均线上穿长期均线时买入(1)，下穿时卖出(-1)
        self.signals['signal'] = 0
        self.signals.loc[
            self.signals['short_ma'] > self.signals['long_ma'], 
            'signal'
        ] = 1
        self.signals.loc[
            self.signals['short_ma'] < self.signals['long_ma'], 
            'signal'
        ] = -1
        
        # 仅在信号变化时记录
        self.signals['positions'] = self.signals['signal'].diff()
        
        return self.signals
```

### 4. 回测引擎

```python
class Backtester:
    """回测引擎"""
    
    def __init__(self, symbol, signals, initial_capital=100000.0):
        self.symbol = symbol
        self.signals = signals
        self.initial_capital = float(initial_capital)
        self.positions = self.generate_positions()
        
    def generate_positions(self):
        """根据信号生成持仓"""
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        positions[self.symbol] = 100 * self.signals['signal']
        return positions
    
    def backtest_portfolio(self, data):
        """计算投资组合价值"""
        portfolio = pd.DataFrame(index=data.index)
        
        # 持仓数量
        portfolio['holdings'] = self.positions[self.symbol] * data['close']
        
        # 现金
        portfolio['cash'] = self.initial_capital - \
            (self.positions[self.symbol].diff() * data['close']).cumsum()
        
        # 总资产
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        
        # 收益
        portfolio['returns'] = portfolio['total'].pct_change()
        
        return portfolio
```

### 5. 性能评估

```python
class PerformanceAnalyzer:
    """性能分析类"""
    
    def __init__(self, portfolio, data):
        self.portfolio = portfolio
        self.data = data
        
    def calculate_metrics(self):
        """计算性能指标"""
        returns = self.portfolio['returns'].dropna()
        
        # 总收益率
        total_return = (
            self.portfolio['total'].iloc[-1] / 
            self.portfolio['total'].iloc[0] - 1
        ) * 100
        
        # 年化收益率
        days = (self.portfolio.index[-1] - self.portfolio.index[0]).days
        annual_return = ((1 + total_return/100) ** (365/days) - 1) * 100
        
        # 夏普比率
        sharpe_ratio = (
            returns.mean() / returns.std() * np.sqrt(252)
        )
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # 胜率
        win_rate = len(returns[returns > 0]) / len(returns[returns != 0]) * 100
        
        metrics = {
            '总收益率 (%)': round(total_return, 2),
            '年化收益率 (%)': round(annual_return, 2),
            '夏普比率': round(sharpe_ratio, 2),
            '最大回撤 (%)': round(max_drawdown, 2),
            '胜率 (%)': round(win_rate, 2),
            '交易次数': len(returns[returns != 0])
        }
        
        return metrics
    
    def print_metrics(self):
        """打印性能指标"""
        metrics = self.calculate_metrics()
        print("=" * 50)
        print("策略性能报告")
        print("=" * 50)
        for key, value in metrics.items():
            print(f"{key}: {value}")
        print("=" * 50)
```

## 完整示例

```python
# 运行回测
if __name__ == "__main__":
    # 1. 加载数据
    data_handler = DataHandler(
        symbol='AAPL',
        start_date='2023-01-01',
        end_date='2024-12-31'
    )
    data = data_handler.load_data('aapl_data.csv')
    
    # 2. 生成交易信号
    strategy = MovingAverageCrossStrategy(
        data=data,
        short_window=5,
        long_window=20
    )
    signals = strategy.generate_signals()
    
    # 3. 执行回测
    backtester = Backtester(
        symbol='AAPL',
        signals=signals,
        initial_capital=100000
    )
    portfolio = backtester.backtest_portfolio(data)
    
    # 4. 性能分析
    analyzer = PerformanceAnalyzer(portfolio, data)
    analyzer.print_metrics()
    
    # 5. 可视化（可选）
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 价格和均线
    ax1.plot(data.index, data['close'], label='价格')
    ax1.plot(signals.index, signals['short_ma'], label='短期均线')
    ax1.plot(signals.index, signals['long_ma'], label='长期均线')
    ax1.set_ylabel('价格')
    ax1.legend()
    
    # 组合价值
    ax2.plot(portfolio.index, portfolio['total'])
    ax2.set_ylabel('组合价值')
    ax2.set_xlabel('日期')
    
    plt.tight_layout()
    plt.show()
```

## 框架优化方向

### 1. 交易成本

```python
def apply_commission(self, trades, commission_rate=0.001):
    """应用交易佣金"""
    return trades * (1 - commission_rate)
```

### 2. 滑点模拟

```python
def apply_slippage(self, price, slippage_pct=0.001):
    """应用滑点"""
    return price * (1 + slippage_pct)
```

### 3. 仓位管理

```python
def kelly_criterion(self, win_rate, win_loss_ratio):
    """凯利公式计算最优仓位"""
    return (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
```

## 注意事项

1. **避免未来信息泄露**：确保策略只使用历史数据
2. **数据质量**：处理停牌、分红、拆股等公司行为
3. **过度拟合**：避免针对历史数据过度优化参数
4. **现实约束**：考虑流动性、交易限制等实际因素

## 总结

本文介绍了一个基础的回测框架，包含：

- 模块化设计便于扩展
- 清晰的策略接口
- 完整的性能评估

在实际应用中，可以根据需求添加更多功能：

- 多资产组合回测
- 风险管理模块
- 参数优化工具
- 实时交易接口

下一篇文章将介绍如何使用机器学习方法优化策略参数。
