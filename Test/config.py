"""
ORB策略回测配置文件
"""

# 回测标的和时间配置
SYMBOLS = ['QQQ']  # 回测的标的列表
START_DATE = '2022-01-01'  # 回测开始日期
END_DATE = '2022-12-31'    # 回测结束日期
INITIAL_CAPITAL = 25000    # 初始资金

# 数据源配置
DATA_SOURCE = 'csv'  # 'ibkr' 或 'csv'，如果IBKR连接失败就使用csv

# 策略参数
COMMISSION_PER_SHARE = 0.0005  # 每股佣金
MAX_LEVERAGE = 4           # 最大杠杆率
RISK_PER_TRADE = 0.01      # 每笔交易风险占比
PROFIT_TARGET_MULTIPLIER = 10  # 止盈目标是止损的倍数

# IBKR连接设置
IBKR_HOST = '127.0.0.1'    # TWS/IB Gateway主机地址
IBKR_PORT = 7497           # TWS/IB Gateway端口号(7497为模拟账户,7496为实盘)
CLIENT_ID = 1              # 客户端ID 