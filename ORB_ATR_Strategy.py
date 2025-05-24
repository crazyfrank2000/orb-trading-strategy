"""
ORB_ATR_Strategy.py - 带有风险仓和杠杆仓详细信息的版本

基于 ORB 策略的改进版，使用 ATR 动态止损 + EoD 止盈
自动将每日交易结果输出到CSV文件（包含风险仓和杠杆仓信息）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from datetime import timedelta, date
import config
import pytz
from market_calendar import MarketCalendar


class ORB_ATR_Strategy:
    def __init__(self, symbol, start_date=None, end_date=None, initial_capital=None, atr_period=14, atr_multiplier=0.05):
        """
        初始化ORB ATR策略
        """
        self.symbol = symbol
        self.start_date = start_date if start_date else config.START_DATE
        self.end_date = end_date if end_date else config.END_DATE
        self.initial_capital = initial_capital if initial_capital else config.INITIAL_CAPITAL
        self.current_capital = self.initial_capital
        
        # ATR 参数
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        
        # 从配置文件获取策略参数
        self.commission_per_share = config.COMMISSION_PER_SHARE
        self.max_leverage = config.MAX_LEVERAGE
        self.risk_per_trade = config.RISK_PER_TRADE
        
        # 交易记录
        self.trades = []
        
        # 初始化市场日历
        self.market_calendar = MarketCalendar(config.TIMEZONE)
        
        # 设置时区
        self.et_tz = pytz.timezone('America/New_York')
        
        # 设置日志目录和CSV文件路径
        self.log_dir = 'log'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # 创建交易日志CSV文件路径
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.trade_log_file = os.path.join(self.log_dir, f"{self.symbol}_trades_{timestamp}.csv")
        
        # 初始化CSV文件头部
        self._initialize_trade_log_csv()
        
        # 寻找可用的数据文件
        data_path = self._find_data_file()
        print(f"使用数据文件: {data_path}")
        
        # 读取5分钟数据
        self.data = self._load_market_data(data_path)
        
        # 计算日线数据和ATR
        self.daily_data = self._calculate_daily_data()
        
        # 处理结果
        self.trading_results = None
    
    def _initialize_trade_log_csv(self):
        """初始化交易日志CSV文件"""
        headers = [
            '交易日期', '品种', '方向', '入场时间(ET)', '出场时间(ET)', 
            '入场价格', '出场价格', '止损价格', '股数', '风险仓', '杠杆仓', '仓位价值',
            'ATR', 'ATR止损距离', '风险点数', '风险金额', '杠杆倍数',
            '佣金', '盈亏', '盈亏比例', 'R倍数', '出场原因',
            '交易前资金', '交易后资金', '是否交易日'
        ]
        
        # 创建CSV文件并写入头部（使用UTF-8 BOM确保Excel正确显示中文）
        with open(self.trade_log_file, 'w', newline='', encoding='utf-8-sig') as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(headers)
        
        print(f"交易日志文件已创建: {self.trade_log_file}")
    
    def _convert_to_et_time(self, timestamp):
        """将时间戳转换为美东时间"""
        if pd.isna(timestamp):
            return None
            
        # 如果时间戳已经有时区信息，直接转换
        if hasattr(timestamp, 'tz') and timestamp.tz is not None:
            return timestamp.tz_convert(self.et_tz)
        else:
            # 如果没有时区信息，假设是UTC时间
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            # 设置为UTC时区然后转换为美东时间
            timestamp_utc = timestamp.tz_localize('UTC')
            return timestamp_utc.tz_convert(self.et_tz)
    
    def _log_trade_to_csv(self, trade, capital_before):
        """将交易记录写入CSV文件"""
        # 格式化交易方向
        direction_str = "做多" if trade['direction'] == 1 else "做空"
        
        # 计算仓位价值
        position_value = trade['shares'] * trade['entry_price']
        
        # 计算杠杆倍数
        leverage_ratio = position_value / capital_before if capital_before > 0 else 0
        
        # 检查是否为交易日
        is_trading_day = self.market_calendar.is_trading_day(trade['date'])
        
        # 转换时间为美东时间
        entry_time_et = self._convert_to_et_time(trade['entry_time'])
        exit_time_et = self._convert_to_et_time(trade['exit_time'])
        
        # 准备CSV行数据（保留2位小数）
        row_data = [
            trade['date'].strftime('%Y-%m-%d'),  # 交易日期
            self.symbol,  # 品种
            direction_str,  # 方向
            entry_time_et.strftime('%H:%M:%S') if entry_time_et else 'N/A',  # 入场时间（美东时间）
            exit_time_et.strftime('%H:%M:%S') if exit_time_et else 'N/A',  # 出场时间（美东时间）
            f"{trade['entry_price']:.2f}",  # 入场价格
            f"{trade['exit_price']:.2f}",  # 出场价格
            f"{trade['stop_loss']:.2f}",  # 止损价格
            trade['shares'],  # 股数
            trade['risk_based_shares'],  # 风险仓
            trade['leverage_based_shares'],  # 杠杆仓
            f"{position_value:.2f}",  # 仓位价值
            f"{trade['atr']:.2f}",  # ATR
            f"{trade['atr_stop_distance']:.2f}",  # ATR止损距离
            f"{trade['risk_in_points']:.2f}",  # 风险点数
            f"{trade['risk_amount']:.2f}",  # 风险金额
            f"{leverage_ratio:.2f}",  # 杠杆倍数
            f"{trade['commission']:.2f}",  # 佣金
            f"{trade['pnl']:.2f}",  # 盈亏
            f"{trade['pnl_pct']:.4f}",  # 盈亏比例
            f"{trade['pnl_in_r']:.2f}",  # R倍数
            trade['exit_reason'],  # 出场原因
            f"{capital_before:.2f}",  # 交易前资金
            f"{self.current_capital:.2f}",  # 交易后资金
            "是" if is_trading_day else "否"  # 是否交易日
        ]
        
        # 写入CSV文件（使用UTF-8 BOM确保Excel正确显示中文）
        try:
            with open(self.trade_log_file, 'a', newline='', encoding='utf-8-sig') as f:
                import csv
                writer = csv.writer(f)
                writer.writerow(row_data)
        except Exception as e:
            print(f"写入交易日志失败: {e}")
    
    def _find_data_file(self):
        """寻找可用的数据文件"""
        import glob
        
        # 首先尝试精确匹配
        exact_path = f"{config.DATA_DIR}/{self.symbol}_5min_full_{self.start_date}_to_{self.end_date}.csv"
        if os.path.exists(exact_path):
            return exact_path
        
        # 查找当前symbol的所有5分钟数据文件
        pattern = f"{config.DATA_DIR}/{self.symbol}_5min_full_*.csv"
        files = glob.glob(pattern)
        
        if files:
            print(f"找到 {len(files)} 个可用的数据文件:")
            for file in files:
                print(f"  - {file}")
            # 返回第一个找到的文件
            return files[0]
        else:
            raise FileNotFoundError(f"未找到 {self.symbol} 的5分钟数据文件，请先运行数据获取程序")
    
    def _load_market_data(self, data_path):
        """加载5分钟市场数据"""
        print(f"加载 {self.symbol} 5分钟市场数据: {data_path}")
        
        data = pd.read_csv(data_path)
        
        # 处理日期列
        try:
            print(f"Date column format sample: {data['date'].iloc[0]}")
            
            # 转换为datetime，设置utc=True避免混合时区警告
            data['date'] = pd.to_datetime(data['date'], utc=True)
            
            # 转换为美东时间
            data['date_et'] = data['date'].dt.tz_convert(self.et_tz)
            
            # 提取交易日期
            data['trade_date'] = data['date_et'].dt.date
        
        except Exception as e:
            print(f"处理日期时出错: {str(e)}")
            raise
        
        # 注意：保留所有数据用于ATR计算
        print(f"保留所有数据用于ATR计算，包含 {len(data['trade_date'].unique())} 个交易日")
        
        if data.empty:
            raise ValueError(f"数据为空")
        
        return data
    
    def _calculate_daily_data(self):
        """计算日线数据和ATR指标"""
        print(f"计算 {self.symbol} 日线数据和ATR指标...")
        
        # 创建日线数据
        daily_data = self.data.groupby('trade_date').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).reset_index()
        
        # 确保日期类型正确
        daily_data['trade_date'] = pd.to_datetime(daily_data['trade_date'])
        daily_data = daily_data.sort_values('trade_date')
        
        # 计算真实波幅 (True Range)
        daily_data['prev_close'] = daily_data['close'].shift(1)
        daily_data['tr1'] = daily_data['high'] - daily_data['low']
        daily_data['tr2'] = abs(daily_data['high'] - daily_data['prev_close'])
        daily_data['tr3'] = abs(daily_data['low'] - daily_data['prev_close'])
        daily_data['tr'] = daily_data[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # 计算ATR (N日平均真实波幅)
        daily_data['atr'] = daily_data['tr'].rolling(window=self.atr_period).mean()
        
        # 处理NaN值
        daily_data = daily_data.dropna(subset=['atr']).copy()
        
        print(f"成功计算ATR指标，剩余 {len(daily_data)} 个交易日")
        return daily_data
    
    def _get_atr_for_date(self, date):
        """获取指定日期的ATR值"""
        date_dt = pd.to_datetime(date).date()
        atr_row = self.daily_data[self.daily_data['trade_date'].dt.date == date_dt]
        if not atr_row.empty:
            return atr_row.iloc[0]['atr']
        return None
    
    def run_strategy(self):
        """运行ORB ATR策略"""
        print(f"开始运行 {self.symbol} ORB ATR策略 ({self.start_date} 至 {self.end_date})...")
        
        # 筛选交易日期范围
        start_date = pd.to_datetime(self.start_date).date()
        end_date = pd.to_datetime(self.end_date).date()
        
        # 按交易日分组处理数据，只处理指定日期范围内的数据
        all_trade_dates = self.data['trade_date'].unique()
        trade_dates = [date for date in all_trade_dates 
                      if isinstance(date, (pd.Timestamp, datetime.date, datetime.datetime)) and 
                      (date.date() if hasattr(date, 'date') else date) >= start_date and 
                      (date.date() if hasattr(date, 'date') else date) <= end_date]
        
        # 过滤交易日
        trade_dates = [date for date in trade_dates if self.market_calendar.is_trading_day(date)]
        
        print(f"在指定日期范围内找到 {len(trade_dates)} 个交易日")
        
        for date in trade_dates:
            # 获取当天的ATR值
            atr = self._get_atr_for_date(date)
            if atr is None:
                print(f"警告: {date} ATR数据不可用，跳过")
                continue
                
            # 获取当天的数据
            day_data = self.data[self.data['trade_date'] == date].sort_values('date')
            
            # 确保当天至少有两根K线
            if len(day_data) < 2:
                print(f"警告: {date} 数据不足，跳过")
                continue
            
            # 提取前两根5分钟K线
            first_candle = day_data.iloc[0]
            second_candle = day_data.iloc[1]
            
            # 跳过十字星日
            if abs(first_candle['close'] - first_candle['open']) < 0.0001:
                print(f"{date}: 第一根K线为十字星，跳过交易")
                continue
            
            # 确定交易方向
            direction = 1 if first_candle['close'] > first_candle['open'] else -1
            
            # 入场价格 (第二根5分钟K线开盘价)
            entry_price = second_candle['open']
            
            # 使用ATR计算止损距离
            atr_stop_distance = self.atr_multiplier * atr
            
            # 设置止损价格 (ATR动态止损)
            stop_loss_price = entry_price - (atr_stop_distance * direction)
            
            # 计算风险
            risk = abs(entry_price - stop_loss_price)
            if risk < 0.0001:  # 防止除以零
                print(f"{date}: 风险太小，跳过交易")
                continue
            
            # 计算持仓规模（返回详细信息）
            position_info = self._calculate_position_size(entry_price, stop_loss_price)
            shares = position_info['shares']
            
            # 如果持仓规模为0，跳过此次交易
            if shares == 0:
                print(f"{date}: 计算持仓为0，跳过交易")
                continue
            
            # 计算佣金
            commission = shares * self.commission_per_share
            
            # 提取当天剩余K线数据，用于模拟交易过程
            remaining_candles = day_data.iloc[2:]
            
            # 初始化出场变量
            exit_price = None
            exit_reason = None
            exit_time = None
            
            # 遍历剩余K线，检查是否触及止损
            for _, candle in remaining_candles.iterrows():
                # 做多情况
                if direction == 1:
                    # 检查是否触及止损
                    if candle['low'] <= stop_loss_price:
                        exit_price = stop_loss_price
                        exit_reason = 'Stop Loss'
                        exit_time = candle['date']
                        break
                # 做空情况
                else:
                    # 检查是否触及止损
                    if candle['high'] >= stop_loss_price:
                        exit_price = stop_loss_price
                        exit_reason = 'Stop Loss'
                        exit_time = candle['date']
                        break
            
            # 如果没有触及止损，则以收盘价平仓
            if exit_price is None:
                last_candle = day_data.iloc[-1]
                exit_price = last_candle['close']
                exit_reason = 'End of Day'
                exit_time = last_candle['date']
            
            # 计算交易盈亏
            trade_pnl = (exit_price - entry_price) * direction * shares - commission
            pnl_pct = trade_pnl / self.current_capital
            pnl_in_r = ((exit_price - entry_price) * direction) / risk
            
            # 记录交易（包含仓位详细信息）
            trade = {
                'date': day_data.iloc[0]['date_et'].date() if 'date_et' in day_data.columns else day_data.iloc[0]['date'].date(),
                'entry_time': second_candle['date'],
                'exit_time': exit_time,
                'direction': direction,
                'entry_price': entry_price,
                'stop_loss': stop_loss_price,
                'atr': atr,
                'atr_stop_distance': atr_stop_distance,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'shares': shares,
                'risk_based_shares': position_info['risk_based_shares'],    # 风险仓
                'leverage_based_shares': position_info['leverage_based_shares'],  # 杠杆仓
                'risk_amount': position_info['risk_amount'],
                'risk_in_points': risk,
                'commission': commission,
                'pnl': trade_pnl,
                'pnl_pct': pnl_pct,
                'pnl_in_r': pnl_in_r
            }
            
            # 记录交易前的资金
            capital_before_trade = self.current_capital
            
            # 更新账户资金
            self.current_capital += trade_pnl
            
            # 添加交易记录
            self.trades.append(trade)
            
            # 写入CSV日志
            self._log_trade_to_csv(trade, capital_before_trade)
            
            # 打印交易摘要 (使用美东时间)
            trade_direction = "多" if direction == 1 else "空"
            position_value = shares * entry_price
            leverage_ratio = position_value / capital_before_trade
            entry_time_et = self._convert_to_et_time(second_candle['date'])
            entry_time_str = entry_time_et.strftime('%H:%M:%S') if entry_time_et else 'N/A'
            print(f"{date}: {trade_direction}头交易, 入场时间: {entry_time_str}(ET), 风险仓: {position_info['risk_based_shares']}, 杠杆仓: {position_info['leverage_based_shares']}, 实际股数: {shares}, 出场原因: {exit_reason}, 盈亏: {pnl_in_r:.2f}R (${trade_pnl:.2f})")
        
        # 计算交易结果
        self._calculate_trading_results()
        
        return self.trading_results
    
    def _calculate_position_size(self, entry_price, stop_loss_price):
        """计算持仓规模，返回详细信息"""
        risk = abs(entry_price - stop_loss_price)
        risk_amount = self.current_capital * self.risk_per_trade
        
        # 基于风险的头寸大小
        risk_based_shares = int(risk_amount / risk)
        
        # 基于杠杆的头寸大小
        leverage_based_shares = int((self.max_leverage * self.current_capital) / entry_price)
        
        # 取两者的较小值
        shares = min(risk_based_shares, leverage_based_shares)
        
        # 返回详细信息
        return {
            'shares': shares,
            'risk_based_shares': risk_based_shares,
            'leverage_based_shares': leverage_based_shares,
            'risk_amount': risk_amount,
            'risk_per_share': risk
        }
    
    def _calculate_trading_results(self):
        """计算交易结果统计数据"""
        if not self.trades:
            self.trading_results = {
                'symbol': self.symbol,
                'total_trades': 0,
                'profitable_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_pnl_pct': 0,
                'avg_pnl_in_r': 0,
                'initial_capital': self.initial_capital,
                'final_capital': self.current_capital,
                'trades': []
            }
            return
        
        # 转换交易记录为DataFrame以便分析
        trades_df = pd.DataFrame(self.trades)
        
        # 计算基本统计数据
        total_trades = len(trades_df)
        profitable_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = profitable_trades / total_trades
        
        total_pnl = sum(trade['pnl'] for trade in self.trades)
        total_pnl_pct = (self.current_capital - self.initial_capital) / self.initial_capital
        
        # 存储交易结果
        self.trading_results = {
            'symbol': self.symbol,
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'avg_pnl_in_r': trades_df['pnl_in_r'].mean(),
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'trades': trades_df
        }
        
        return self.trading_results


if __name__ == "__main__":
    # 测试代码
    strategy = ORB_ATR_Strategy('TQQQ')
    results = strategy.run_strategy()
    
    print("\n===== 策略运行结果 =====")
    print(f"总交易次数: {results['total_trades']}")
    print(f"盈利次数: {results['profitable_trades']}")
    print(f"亏损次数: {results['total_trades'] - results['profitable_trades']}")
    print(f"胜率: {results['win_rate']:.2%}")
    print(f"总盈亏: ${results['total_pnl']:.2f}")
    print(f"收益率: {results['total_pnl_pct']:.2%}")
    print(f"平均R倍数: {results['avg_pnl_in_r']:.2f}R")
    print("策略运行完成") 