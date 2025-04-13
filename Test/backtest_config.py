"""
ORB策略回测配置文件
可以在这里修改回测参数
"""

# 回测参数
SYMBOLS = ['QQQ']  # 要回测的标的
START_DATE = '2016-01-01'  # 回测开始日期
END_DATE = '2023-12-31'    # 回测结束日期
INITIAL_CAPITAL = 25000    # 初始资金

# 策略参数
MAX_LEVERAGE = 4           # 最大杠杆率
RISK_PER_TRADE = 0.01      # 每笔交易风险占比（1%）
COMMISSION_PER_SHARE = 0.0005  # 每股佣金 