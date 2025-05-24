"""
ORB策略回测配置文件
"""

# 基本回测参数
SYMBOLS = ['TQQQ']  # 交易品种列表，可设置多个股票进行回测
START_DATE = '2024-01-01'  # 回测开始日期
END_DATE = '2025-05-23'    # 回测结束日期
INITIAL_CAPITAL = 25000    # 初始资金

# 策略参数
COMMISSION_PER_SHARE = 0.0005  # 每股佣金
MAX_LEVERAGE = 4               # 最大杠杆率
RISK_PER_TRADE = 0.01          # 每笔交易风险占比(1%)
TAKE_PROFIT_R = 10             # 止盈目标 (风险的倍数)

# IBKR连接参数
IBKR_HOST = '127.0.0.1'        # IBKR服务器地址
IBKR_PORT = 7497               # IBKR端口 (7497为模拟账户, 7496为实盘账户)
IBKR_CLIENT_ID = 1            # 客户端ID

# 数据和输出设置
DATA_DIR = 'data'              # 数据存储目录
OUTPUT_DIR = 'orb_backtest_results'  # 回测结果保存目录
USE_RTH = True                 # 是否仅使用常规交易时段数据

# 时区和市场时间设置
TIMEZONE = 'America/New_York'  # 使用纽约时区
MARKET_OPEN_TIME = '09:30'     # 市场开盘时间
MARKET_CLOSE_TIME = '16:00'    # 市场收盘时间

# 数据获取设置
MAX_RETRIES = 3               # 最大重试次数
RETRY_DELAY = 2              # 重试延迟（秒）
REQUEST_DELAY = 1            # 请求间隔（秒）

# 日志设置
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s' 