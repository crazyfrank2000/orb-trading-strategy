import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from ibkr_data_fetcher import IBKRDataFetcher, fetch_and_save_data

class ORBBacktest:
    def __init__(self, symbol, start_date, end_date, initial_capital=25000, data_source='ibkr'):
        """
        初始化ORB策略回测
        
        参数:
        symbol (str): 交易的股票代码
        start_date (str): 回测开始日期，格式为'YYYY-MM-DD'
        end_date (str): 回测结束日期，格式为'YYYY-MM-DD'
        initial_capital (float): 初始资金，默认25,000美元
        data_source (str): 数据源，'ibkr'或'csv'
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_per_share = 0.0005  # 每股佣金
        self.max_leverage = 4  # 最大杠杆率
        self.risk_per_trade = 0.01  # 每笔交易风险占比（1%）
        self.data_source = data_source
        
        # 交易记录
        self.trades = []
        
        # 获取历史数据
        self.data = self._get_data()
        
        # 回测结果
        self.backtest_results = None
    
    def _get_data(self):
        """获取历史5分钟K线数据，支持多种数据源"""
        data_dir = 'data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        csv_file = f"{data_dir}/{self.symbol}_5min_data.csv"
        
        # 如果使用IBKR数据源或CSV文件不存在，则获取数据
        if self.data_source == 'ibkr' or not os.path.exists(csv_file):
            print(f"从IBKR获取 {self.symbol} 的5分钟历史数据...")
            
            try:
                # 获取数据并保存
                result = fetch_and_save_data(
                    [self.symbol], 
                    self.start_date, 
                    self.end_date,
                    data_dir
                )
                
                if not result[self.symbol]['success']:
                    raise ValueError(f"获取数据失败: {result[self.symbol]['error']}")
                
            except Exception as e:
                print(f"无法从IBKR获取数据: {str(e)}")
                
                # 如果CSV文件存在，尝试使用已有数据
                if os.path.exists(csv_file):
                    print(f"尝试使用现有CSV文件: {csv_file}")
                else:
                    print("尝试从Yahoo Finance获取替代数据...")
                    try:
                        # 使用yfinance作为备用数据源
                        import yfinance as yf
                        from datetime import datetime, timedelta
                        
                        print(f"从Yahoo Finance获取 {self.symbol} 的数据...")
                        
                        # 获取1天的数据，作为检测该标的是否可用的测试
                        test_data = yf.download(self.symbol, period="1d")
                        if test_data.empty:
                            raise ValueError(f"Yahoo Finance中未找到 {self.symbol} 的数据")
                        
                        # 由于yfinance的5分钟数据限制，我们需要分批获取
                        start_date = datetime.strptime(self.start_date, '%Y-%m-%d')
                        end_date = datetime.strptime(self.end_date, '%Y-%m-%d')
                        
                        all_data = []
                        current_start = start_date
                        
                        while current_start < end_date:
                            current_end = min(current_start + timedelta(days=60), end_date)
                            
                            # 格式化日期
                            cs_str = current_start.strftime('%Y-%m-%d')
                            ce_str = current_end.strftime('%Y-%m-%d')
                            
                            print(f"获取 {cs_str} 到 {ce_str} 的数据...")
                            
                            # 获取5分钟K线数据
                            df = yf.download(self.symbol, start=cs_str, end=ce_str, interval="5m")
                            
                            if not df.empty:
                                all_data.append(df)
                            
                            current_start = current_end + timedelta(days=1)
                        
                        if not all_data:
                            raise ValueError(f"未能获取 {self.symbol} 的历史数据")
                        
                        # 合并所有数据
                        yf_data = pd.concat(all_data)
                        
                        # 重命名列以匹配IBKR格式
                        yf_data = yf_data.rename(columns={
                            'Open': 'open',
                            'High': 'high',
                            'Low': 'low',
                            'Close': 'close',
                            'Volume': 'volume'
                        })
                        
                        # 添加日期列
                        yf_data['date'] = yf_data.index
                        yf_data['Date'] = yf_data.index.date
                        
                        # 保存到CSV
                        yf_data.to_csv(csv_file)
                        print(f"Yahoo Finance数据已保存至: {csv_file}")
                        
                    except Exception as yf_error:
                        print(f"从Yahoo Finance获取数据也失败: {str(yf_error)}")
                        raise ValueError(f"无法获取 {self.symbol} 的历史数据") from e
        
        # 从CSV文件加载数据
        print(f"从CSV文件加载 {self.symbol} 数据...")
        try:
            data = pd.read_csv(csv_file, parse_dates=['date'])
            
            # 尝试解析Date列
            if 'Date' in data.columns:
                try:
                    data['Date'] = pd.to_datetime(data['Date']).dt.date
                except:
                    # 如果Date列解析失败，从date列生成
                    data['Date'] = pd.to_datetime(data['date']).dt.date
            else:
                # 如果没有Date列，从date列生成
                data['Date'] = pd.to_datetime(data['date']).dt.date
            
            # 筛选日期范围
            start = datetime.strptime(self.start_date, '%Y-%m-%d').date()
            end = datetime.strptime(self.end_date, '%Y-%m-%d').date()
            
            data = data[(data['Date'] >= start) & (data['Date'] <= end)]
            
            # 确保列名一致性
            column_mapping = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            
            # 检查列是否存在，然后重命名
            for old_col, new_col in column_mapping.items():
                if old_col in data.columns and new_col not in data.columns:
                    data = data.rename(columns={old_col: new_col})
            
            print(f"成功加载 {len(data)} 条数据")
            return data
        
        except Exception as e:
            print(f"加载CSV数据失败: {str(e)}")
            raise
    
    # 以下代码与原来的ORBBacktest相同
    def _calculate_position_size(self, entry_price, stop_loss_price):
        """
        计算持仓规模
        
        公式: Shares = int[min((A * 0.01 / R), (4 * A / P))]
        A: 账户资金
        R: 入场价与止损价的差距
        P: 入场价格
        """
        risk = abs(entry_price - stop_loss_price)
        risk_amount = self.current_capital * self.risk_per_trade
        
        # 基于风险的头寸大小
        risk_based_shares = int(risk_amount / risk)
        
        # 基于杠杆的头寸大小
        leverage_based_shares = int((self.max_leverage * self.current_capital) / entry_price)
        
        # 取两者的较小值
        shares = min(risk_based_shares, leverage_based_shares)
        
        return shares
    
    def run_backtest(self):
        """运行ORB策略回测"""
        print("开始回测ORB策略...")
        
        # 按日期分组
        grouped_data = self.data.groupby('Date')
        
        for date, day_data in grouped_data:
            if len(day_data) < 2:  # 确保有足够的数据
                continue
            
            # 提取开盘5分钟的数据
            first_5min = day_data.iloc[0]
            
            # 检查是否为十字星（开盘价等于收盘价）
            if abs(first_5min['Open'] - first_5min['Close']) < 0.0001:
                continue  # 如果是十字星，当天不交易
            
            # 确定方向
            direction = 1 if first_5min['Close'] > first_5min['Open'] else -1
            
            # 确定进场价格（第二根5分钟蜡烛的开盘价）
            if len(day_data) < 2:
                continue
            
            entry_price = day_data.iloc[1]['Open']
            
            # 确定止损价格
            stop_loss_price = first_5min['Low'] if direction == 1 else first_5min['High']
            
            # 计算风险（R）
            risk = abs(entry_price - stop_loss_price)
            
            # 设置止盈目标（10R）
            take_profit_price = entry_price + (10 * risk * direction)
            
            # 计算持仓规模
            shares = self._calculate_position_size(entry_price, stop_loss_price)
            
            # 如果持仓规模为0，跳过此次交易
            if shares == 0:
                continue
            
            # 计算佣金
            commission = shares * self.commission_per_share
            
            # 记录交易
            trade = {
                'date': date,
                'entry_time': day_data.iloc[1].name,
                'direction': direction,
                'entry_price': entry_price,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'shares': shares,
                'risk_amount': self.current_capital * self.risk_per_trade,
                'commission': commission,
                'exit_price': None,
                'exit_time': None,
                'pnl': None,
                'pnl_pct': None,
                'exit_reason': None
            }
            
            # 模拟交易过程
            day_data_after_entry = day_data.iloc[2:]  # 从第三根K线开始
            
            # 初始化退出标志
            exit_executed = False
            
            for idx, bar in day_data_after_entry.iterrows():
                # 检查是否触及止损
                if (direction == 1 and bar['Low'] <= stop_loss_price) or \
                   (direction == -1 and bar['High'] >= stop_loss_price):
                    trade['exit_price'] = stop_loss_price
                    trade['exit_time'] = idx
                    trade['exit_reason'] = 'Stop Loss'
                    exit_executed = True
                    break
                
                # 检查是否触及止盈
                if (direction == 1 and bar['High'] >= take_profit_price) or \
                   (direction == -1 and bar['Low'] <= take_profit_price):
                    trade['exit_price'] = take_profit_price
                    trade['exit_time'] = idx
                    trade['exit_reason'] = 'Take Profit'
                    exit_executed = True
                    break
            
            # 如果没有触及止损或止盈，在当天收盘时以收盘价平仓
            if not exit_executed:
                trade['exit_price'] = day_data.iloc[-1]['Close']
                trade['exit_time'] = day_data.iloc[-1].name
                trade['exit_reason'] = 'End of Day'
            
            # 计算交易盈亏
            trade_pnl = (trade['exit_price'] - trade['entry_price']) * direction * shares - commission
            trade['pnl'] = trade_pnl
            trade['pnl_pct'] = trade_pnl / self.current_capital
            
            # 更新账户资金
            self.current_capital += trade_pnl
            
            # 添加交易记录
            self.trades.append(trade)
        
        # 计算回测结果
        self._calculate_backtest_results()
        
        return self.backtest_results
    
    def _calculate_backtest_results(self):
        """计算回测结果统计数据"""
        if not self.trades:
            self.backtest_results = {
                'total_trades': 0,
                'profitable_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_pnl_pct': 0,
                'avg_pnl_pct': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
            return
        
        # 转换交易记录为DataFrame以便分析
        trades_df = pd.DataFrame(self.trades)
        
        # 计算基本统计数据
        total_trades = len(trades_df)
        profitable_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(trade['pnl'] for trade in self.trades)
        total_pnl_pct = (self.current_capital - self.initial_capital) / self.initial_capital
        
        avg_pnl_pct = trades_df['pnl_pct'].mean() if total_trades > 0 else 0
        
        # 计算最大回撤
        capital_history = [self.initial_capital]
        for trade in self.trades:
            capital_history.append(capital_history[-1] + trade['pnl'])
        
        max_drawdown = 0
        peak = capital_history[0]
        
        for value in capital_history:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # 简化的夏普比率计算（假设无风险利率为0）
        if len(trades_df) > 1:
            returns = trades_df['pnl_pct'].values
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 存储回测结果
        self.backtest_results = {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'avg_pnl_pct': avg_pnl_pct,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'daily_returns': self._calculate_daily_returns()
        }
    
    def _calculate_daily_returns(self):
        """计算每日收益率"""
        if not self.trades:
            return pd.Series()
        
        # 创建日期和每日盈亏的字典
        daily_pnl = {}
        
        for trade in self.trades:
            date = trade['date']
            if date in daily_pnl:
                daily_pnl[date] += trade['pnl']
            else:
                daily_pnl[date] = trade['pnl']
        
        # 转换为Series
        daily_pnl_series = pd.Series(daily_pnl)
        
        # 计算每日收益率
        initial_capital = self.initial_capital
        daily_returns = {}
        
        for date, pnl in daily_pnl.items():
            daily_returns[date] = pnl / initial_capital
            initial_capital += pnl
        
        return pd.Series(daily_returns)
    
    def plot_results(self, save_path=None):
        """绘制回测结果图表"""
        if not self.trades:
            print("没有交易记录，无法绘制图表")
            return
        
        plt.figure(figsize=(15, 12))
        
        # 绘制账户价值曲线
        plt.subplot(3, 1, 1)
        capital_history = [self.initial_capital]
        dates = [self.trades[0]['date']]
        
        for trade in self.trades:
            capital_history.append(capital_history[-1] + trade['pnl'])
            dates.append(trade['date'])
        
        plt.plot(dates, capital_history)
        plt.title('账户价值变化')
        plt.xlabel('日期')
        plt.ylabel('账户价值 ($)')
        plt.grid(True)
        
        # 绘制每日收益率
        plt.subplot(3, 1, 2)
        daily_returns = self.backtest_results['daily_returns']
        daily_returns.plot(kind='bar')
        plt.title('每日收益率')
        plt.xlabel('日期')
        plt.ylabel('收益率 (%)')
        plt.grid(True)
        
        # 绘制交易分布
        plt.subplot(3, 1, 3)
        trades_pnl = [trade['pnl'] for trade in self.trades]
        plt.hist(trades_pnl, bins=20)
        plt.title('交易盈亏分布')
        plt.xlabel('盈亏 ($)')
        plt.ylabel('交易次数')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"图表已保存至 {save_path}")
        
        plt.show()
    
    def generate_report(self, save_path=None):
        """生成回测报告"""
        if not self.backtest_results:
            print("没有回测结果，请先运行回测")
            return
        
        report = f"""
        =============================================
                    ORB策略回测报告
        =============================================
        
        交易标的: {self.symbol}
        回测期间: {self.start_date} 至 {self.end_date}
        初始资金: ${self.initial_capital:.2f}
        数据源: IBKR
        
        ---------------------------------------------
                    回测结果统计
        ---------------------------------------------
        总交易次数: {self.backtest_results['total_trades']}
        盈利交易次数: {self.backtest_results['profitable_trades']}
        胜率: {self.backtest_results['win_rate']:.2%}
        
        总盈亏: ${self.backtest_results['total_pnl']:.2f}
        总收益率: {self.backtest_results['total_pnl_pct']:.2%}
        平均每笔收益率: {self.backtest_results['avg_pnl_pct']:.2%}
        
        最大回撤: {self.backtest_results['max_drawdown']:.2%}
        夏普比率: {self.backtest_results['sharpe_ratio']:.2f}
        
        期末资金: ${self.backtest_results['final_capital']:.2f}
        
        ---------------------------------------------
                    交易明细
        ---------------------------------------------
        """
        
        # 添加最近10笔交易记录
        recent_trades = self.trades[-10:] if len(self.trades) > 10 else self.trades
        
        for i, trade in enumerate(recent_trades):
            report += f"""
        交易 {i+1}:
            日期: {trade['date']}
            方向: {'多' if trade['direction'] == 1 else '空'}
            入场时间: {trade['entry_time']}
            入场价格: ${trade['entry_price']:.2f}
            止损价格: ${trade['stop_loss']:.2f}
            止盈价格: ${trade['take_profit']:.2f}
            股数: {trade['shares']}
            退出时间: {trade['exit_time']}
            退出价格: ${trade['exit_price']:.2f}
            退出原因: {trade['exit_reason']}
            盈亏: ${trade['pnl']:.2f} ({trade['pnl_pct']:.2%})
            """
        
        report += "\n        =============================================\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"报告已保存至 {save_path}")
        
        return report 