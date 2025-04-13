# ORB Trading Strategy

An implementation of the Opening Range Breakout (ORB) trading strategy in Python. This project provides a complete backtesting framework for trading strategies based on the first 5-minute candle of the trading day.

## Features

- Automated backtesting of ORB strategy across multiple symbols
- Comprehensive risk management with position sizing
- Detailed trade analysis and reporting
- Excel reports with equity curves and performance metrics
- Automatic data fetching from Interactive Brokers

## Requirements

- Python 3.6+
- pandas
- numpy
- matplotlib
- openpyxl
- Interactive Brokers TWS or IB Gateway (for data fetching)

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/crazyfrank2000/orb-trading-strategy.git
cd orb-trading-strategy
pip install -r requirements.txt
```

## Configuration

Edit `config.py` to set up your backtest parameters:

- Trading symbols
- Date range
- Initial capital
- Risk parameters
- Data directories

## Usage

Run the main script to execute the backtest:

```bash
python ORB_Trading_Strategy.py
```

Results will be saved to the output directory specified in the config file.

## Data Preparation

The script can automatically fetch data from Interactive Brokers or use pre-downloaded CSV files. The 5-minute data should be in the format:

- date: Datetime in UTC
- open, high, low, close: OHLC prices
- volume: Trading volume

## License

MIT

## Disclaimer

This software is for educational purposes only. Use it at your own risk. Trading involves substantial risk of loss and is not suitable for all investors.

# ORB Trading Strategy - README

## 概述

ORB Trading Strategy 是一个基于开盘区间突破 (Opening Range Breakout) 策略的自动化交易系统。该系统从 Interactive Brokers (IBKR) 获取 5 分钟 K 线数据，根据每个交易日开盘后的第一个 5 分钟 K 线形成的价格区间进行交易决策，实现了完整的回测功能，包括风险管理、资金管理、交易执行和结果分析。系统包含原始 ORB 策略和使用 ATR 动态止损的改进版本。

## 主要功能

1. **数据处理**
   - 自动从 IBKR 获取或加载 5 分钟 K 线数据
   - 支持多种交易品种的数据处理
   - 根据配置文件灵活设置数据日期范围

2. **交易策略**
   - 基于首个 5 分钟 K 线方向的交易信号生成
   - 精确的入场、止损和止盈逻辑
   - 十字星日自动跳过交易
   - ATR 动态止损的改进版策略

3. **风险管理**
   - 实现 1% 风险管理规则，每笔交易风险控制在账户资金的 1%
   - 最大杠杆控制在 4 倍，符合美国经纪商监管要求
   - 基于风险单位 (R) 的止盈目标设置
   - ATR 动态止损机制

4. **结果分析**
   - 交易统计和绩效衡量
   - 完整的资金曲线生成
   - 策略表现与买入持有策略的对比
   - 原始策略与 ATR 改进版策略的对比

5. **报告生成**
   - 详细的 Excel 交易报告，包含每笔交易记录
   - 摘要统计报告，展示关键绩效指标
   - 可视化图表，展示策略表现

## 项目结构

```
IBKR/
├── config.py                 # 配置文件，包含交易参数和数据路径
├── ibkr_5min_data_fetcher.py # IBKR 5分钟数据获取模块
├── market_calendar.py        # 市场日历模块，处理交易日
├── ORB_Trading_Strategy.py   # 原始 ORB 策略实现
├── ORB_ATR_Strategy.py       # 基于 ATR 的改进版 ORB 策略
├── data/                     # 存储交易数据
│   ├── raw/                  # 原始数据
│   └── processed/            # 处理后的数据
└── output/                   # 存储输出报告和图表
    ├── reports/              # Excel和文本报告
    ├── charts/               # 图表输出
    └── strategy_comparison/  # 策略对比结果
```

## 算法实现

### 1. 交易信号生成

交易信号基于以下逻辑生成：

```
如果第一根 K 线上涨 (收盘价 > 开盘价):
    在第二根 5 分钟 K 线的开盘价做多
如果第一根 K 线下跌 (收盘价 < 开盘价):
    在第二根 5 分钟 K 线的开盘价做空
如果第一根 K 线为十字星 (开盘价 ≈ 收盘价):
    当天不交易
```

### 2. 止损设置

#### 原始 ORB 策略止损:

止损点位设置在第一根 K 线的极值点：

```
做多: 止损设置在第一根 5 分钟 K 线的最低点
做空: 止损设置在第一根 5 分钟 K 线的最高点
```

#### ATR 改进版止损:

使用 ATR (Average True Range) 指标计算动态止损距离:

```
ATR = N日平均真实波幅
止损距离 = ATR * 乘数(默认为0.05)
做多: 止损设置在入场价格 - 止损距离
做空: 止损设置在入场价格 + 止损距离
```

ATR 计算方法:
1. 计算真实波幅 (True Range):
   - TR1 = 当日最高价 - 当日最低价
   - TR2 = |当日最高价 - 前一日收盘价|
   - TR3 = |当日最低价 - 前一日收盘价|
   - TR = max(TR1, TR2, TR3)

2. 计算平均真实波幅:
   - ATR = N天TR的简单移动平均 (默认N=14)

3. 设置止损距离:
   - 止损距离 = ATR * 乘数 (默认乘数=0.05)
   - 此方法使止损距离根据市场波动性动态调整

### 3. 止盈目标

#### 原始 ORB 策略止盈:

止盈目标设置为止损距离的 10 倍（即 10R）：

```
止盈价格 = 入场价格 + 10 * 风险距离 * 方向
```

其中，方向为 1（做多）或 -1（做空）。

#### ATR 改进版:

取消固定止盈目标，改为收盘平仓（End of Day）：

```
无论价格如何变动，当日收盘时平仓退出
```

这一改进基于研究发现，在日内交易策略中，收盘平仓通常比设置固定止盈目标能够获得更好的收益率，因为它允许获利头寸尽可能运行，充分捕捉当天的趋势。

### 4. 仓位计算

两个策略均采用相同的仓位计算公式：

```
Shares = int[min((A * 0.01 / R), (4 * A / P))]

A: 账户资金
R: 风险（入场价与止损价的差距）
P: 入场价格（第二根 5 分钟 K 线开盘价）
```

这确保了每笔交易的风险控制在账户资金的 1%，且最大杠杆不超过 4 倍。

### 5. 出场逻辑

#### 原始 ORB 策略出场:

交易出场基于三种情况：

```
1. 价格触及止损点 → 止损出场（Stop Loss）
2. 价格触及止盈点 → 止盈出场（Take Profit）
3. 当天收盘前未达到上述条件 → 收盘平仓（End of Day）
```

#### ATR 改进版出场:

交易出场基于两种情况：

```
1. 价格触及ATR动态止损点 → 止损出场（Stop Loss）
2. 当天收盘时 → 收盘平仓（End of Day）
```

系统会遍历当天所有剩余的 5 分钟 K 线，检查是否触及止损价格，如果全天都未触及，则以收盘价平仓。

## 策略对比

系统支持原始 ORB 策略与 ATR 改进版策略的对比分析，提供以下指标的比较：

- 总交易次数与胜率
- 总盈亏和总收益率
- 平均 R 倍数
- 出场原因分析
- 权益曲线对比

研究表明，ATR 动态止损 + EoD 平仓的改进策略在多数情况下能显著提高原始 ORB 策略的表现，主要优势包括：

1. **动态止损调整**：ATR 策略根据市场波动性自动调整止损距离，避免止损过紧或过松
2. **更高的盈亏比**：取消固定止盈，允许盈利头寸充分运行
3. **更好的资金曲线**：通常具有更低的回撤和更高的夏普比率
4. **更好的适应性**：可以适应不同波动性的市场环境

## ATR 策略实现细节

ATR策略在原始ORB策略基础上做了两个关键改进：

1. **止损计算**：
   ```python
   # 获取当天的ATR值
   atr = self._get_atr_for_date(date)
   
   # 计算止损距离
   atr_stop_distance = self.atr_multiplier * atr
   
   # 设置止损价格
   stop_loss_price = entry_price - (atr_stop_distance * direction)
   ```

2. **出场逻辑**：
   ```python
   # 遍历当天剩余K线
   for i in range(entry_index + 1, len(day_data)):
       current_candle = day_data.iloc[i]
       
       # 检查是否触及止损
       if direction == 1 and current_candle['low'] <= stop_loss_price:
           exit_price = stop_loss_price
           exit_reason = 'Stop Loss'
           exit_time = current_candle['date']
           break
       elif direction == -1 and current_candle['high'] >= stop_loss_price:
           exit_price = stop_loss_price
           exit_reason = 'Stop Loss'
           exit_time = current_candle['date']
           break
   
   # 如果未触及止损，收盘平仓
   if exit_price is None:
       exit_price = day_data.iloc[-1]['close']
       exit_reason = 'End of Day'
       exit_time = day_data.iloc[-1]['date']
   ```

## 使用说明

### 基本使用

```bash
# 运行原始 ORB 策略
python ORB_Trading_Strategy.py --symbol QQQ

# 运行 ATR 改进版策略
python ORB_ATR_Strategy.py --symbol QQQ --atr_period 14 --atr_multiplier 0.05

# 比较两种策略
python ORB_ATR_Strategy.py --symbol QQQ --compare
```

### 参数说明

- `--symbol`: 交易品种代码
- `--start_date`: 回测开始日期 (YYYY-MM-DD)
- `--end_date`: 回测结束日期 (YYYY-MM-DD)
- `--capital`: 初始资金
- `--atr_period`: ATR计算周期 (默认14)
- `--atr_multiplier`: ATR乘数 (默认0.05)
- `--compare`: 比较原始策略和ATR改进版
- `--all`: 运行所有配置的交易品种

### 对比分析示例

```bash
# 比较不同ATR乘数的策略效果
python ORB_ATR_Strategy.py --symbol SPY --atr_multiplier 0.03 --compare
python ORB_ATR_Strategy.py --symbol SPY --atr_multiplier 0.05 --compare
python ORB_ATR_Strategy.py --symbol SPY --atr_multiplier 0.10 --compare

# 批量运行多个交易品种并生成对比报告
python batch_run.py --compare --symbols SPY,QQQ,IWM,AAPL,MSFT
```

