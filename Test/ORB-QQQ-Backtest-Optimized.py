"""
ORB (Opening Range Breakout) Strategy Backtest - QQQ Version
Based on pre-processed ORB data from IBKR to improve backtest accuracy
Implementation of strategy from paper "Can Day Trading Really Be Profitable?"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import matplotlib.dates as mdates
import matplotlib.ticker as mtick

# 设置matplotlib样式
plt.rcParams['figure.dpi'] = 120
try:
    plt.style.use('seaborn-v0_8-whitegrid')  # 新版matplotlib
except:
    try:
        plt.style.use('seaborn-whitegrid')  # 旧版matplotlib
    except:
        plt.style.use('default')
        plt.rcParams['axes.grid'] = True

class ORB_Strategy:
    def __init__(self, orb_data_path, start_date, end_date, initial_capital=25000):
        """
        初始化ORB策略回测
        
        参数:
        orb_data_path (str): ORB专用数据文件路径
        start_date (str): 回测开始日期，格式'YYYY-MM-DD'
        end_date (str): 回测结束日期，格式'YYYY-MM-DD'
        initial_capital (float): 初始资金，默认$25,000
        """
        self.symbol = 'QQQ'
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_per_share = 0.0005  # 每股佣金
        self.max_leverage = 4  # 最大杠杆率
        self.risk_per_trade = 0.01  # 每笔交易风险占比（1%）
        
        # 交易记录
        self.trades = []
        
        # 读取ORB专用数据
        self.data = self._load_orb_data(orb_data_path)
        
        # 回测结果
        self.backtest_results = None
    
    def _load_orb_data(self, data_path):
        """加载ORB专用数据"""
        print(f"加载ORB策略数据: {data_path}")
        
        data = pd.read_csv(data_path)
        
        # 确保Date列为日期类型
        data['Date'] = pd.to_datetime(data['Date'])
        
        # 筛选日期范围
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        data = data[(data['Date'] >= start) & (data['Date'] <= end)]
        
        if data.empty:
            raise ValueError(f"数据筛选后为空，请检查日期范围是否有效")
        
        print(f"成功加载{len(data)}天的ORB策略数据")
        return data
    
    def run_backtest(self):
        """运行ORB策略回测"""
        print(f"开始回测ORB策略 ({self.start_date} 至 {self.end_date})...")
        
        # 交易统计
        total_days = len(self.data)
        days_with_trades = 0
        days_skipped_doji = 0
        
        # 遍历每一天
        for _, day in self.data.iterrows():
            # 跳过十字星日
            if day['is_doji']:
                days_skipped_doji += 1
                continue
            
            # 获取交易方向（首根5分钟K线方向）
            direction = day['direction']
            
            # 入场价格（第二根5分钟K线开盘价）
            entry_price = day['second_5min_open']
            
            # 止损价格（首根5分钟K线的高/低点）
            stop_loss_price = day['first_5min_low'] if direction == 1 else day['first_5min_high']
            
            # 计算风险
            risk = abs(entry_price - stop_loss_price)
            if risk < 0.0001:  # 防止除以零
                continue
            
            # 设置止盈目标（10R）
            take_profit_price = entry_price + (10 * risk * direction)
            
            # 计算持仓规模
            shares = self._calculate_position_size(entry_price, stop_loss_price)
            
            # 如果持仓规模为0，跳过此次交易
            if shares == 0:
                continue
            
            # 计算佣金
            commission = shares * self.commission_per_share
            
            # 确定出场价格
            # 在实际情况中，我们需要模拟日内价格走势来确定是否触及止损或止盈
            # 由于我们只有日OHLC数据，这里采用简化方法估算出场价格
            
            # 多头情况
            if direction == 1:
                # 检查是否触及止损
                if day['day_low'] <= stop_loss_price:
                    exit_price = stop_loss_price
                    exit_reason = 'Stop Loss'
                # 检查是否触及止盈
                elif day['day_high'] >= take_profit_price:
                    exit_price = take_profit_price
                    exit_reason = 'Take Profit'
                # 当日收盘
                else:
                    exit_price = day['day_close']
                    exit_reason = 'End of Day'
            # 空头情况
            else:
                # 检查是否触及止损
                if day['day_high'] >= stop_loss_price:
                    exit_price = stop_loss_price
                    exit_reason = 'Stop Loss'
                # 检查是否触及止盈
                elif day['day_low'] <= take_profit_price:
                    exit_price = take_profit_price
                    exit_reason = 'Take Profit'
                # 当日收盘
                else:
                    exit_price = day['day_close']
                    exit_reason = 'End of Day'
            
            # 计算交易盈亏
            trade_pnl = (exit_price - entry_price) * direction * shares - commission
            pnl_pct = trade_pnl / self.current_capital
            pnl_in_r = ((exit_price - entry_price) * direction) / risk
            
            # 记录交易
            trade = {
                'date': day['Date'],
                'direction': direction,
                'entry_price': entry_price,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'shares': shares,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'pnl': trade_pnl,
                'pnl_pct': pnl_pct,
                'pnl_in_r': pnl_in_r,
                'risk_amount': self.current_capital * self.risk_per_trade,
                'risk_in_points': risk,
                'commission': commission
            }
            
            # 更新账户资金
            self.current_capital += trade_pnl
            
            # 添加交易记录
            self.trades.append(trade)
            days_with_trades += 1
        
        print(f"回测完成: 总计{total_days}天, {days_with_trades}天产生交易, {days_skipped_doji}天因十字星跳过")
        
        # 计算回测结果
        self._calculate_backtest_results()
        
        return self.backtest_results
    
    def _calculate_position_size(self, entry_price, stop_loss_price):
        """
        计算持仓规模，根据论文中的公式:
        Shares = int[min((A * 0.01 / R), (4 * A / P))]
        
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
                'sharpe_ratio': 0,
                'initial_capital': self.initial_capital,
                'final_capital': self.current_capital
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
        
        # R单位平均盈亏
        avg_pnl_in_r = trades_df['pnl_in_r'].mean() if total_trades > 0 else 0
        
        # 创建完整的日期范围，包括无交易日
        all_dates = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
        equity_curve = pd.DataFrame(index=all_dates)
        equity_curve['capital'] = np.nan
        
        # 设置初始资金
        equity_curve.loc[equity_curve.index[0], 'capital'] = self.initial_capital
        
        # 按日期累积权益
        for trade in sorted(self.trades, key=lambda x: x['date']):
            trade_date = pd.to_datetime(trade['date'])
            # 如果日期存在于索引中
            if trade_date in equity_curve.index:
                previous_capital = equity_curve.loc[:trade_date, 'capital'].dropna().iloc[-1] \
                    if not pd.isna(equity_curve.loc[:trade_date, 'capital']).all() else self.initial_capital
                equity_curve.loc[trade_date, 'capital'] = previous_capital + trade['pnl']
        
        # 前向填充缺失值
        equity_curve['capital'] = equity_curve['capital'].fillna(method='ffill')
        
        # 计算回撤
        equity_curve['peak'] = equity_curve['capital'].cummax()
        equity_curve['drawdown'] = (equity_curve['peak'] - equity_curve['capital']) / equity_curve['peak']
        max_drawdown = equity_curve['drawdown'].max()
        
        # 计算年化收益率
        days_in_market = (pd.to_datetime(self.end_date) - pd.to_datetime(self.start_date)).days
        annual_return = (self.current_capital / self.initial_capital) ** (365 / days_in_market) - 1
        
        # 计算夏普比率
        equity_curve['daily_return'] = equity_curve['capital'].pct_change()
        sharpe_ratio = equity_curve['daily_return'].mean() / equity_curve['daily_return'].std() * np.sqrt(252) if equity_curve['daily_return'].std() > 0 else 0
        
        # 存储回测结果
        self.backtest_results = {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'avg_pnl_pct': avg_pnl_pct,
            'avg_pnl_in_r': avg_pnl_in_r,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'annual_return': annual_return,
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'equity_curve': equity_curve
        }
    
    def plot_equity_curve_comparison(self, save_path=None):
        """绘制与买入持有策略的权益曲线对比图（复现论文图1）"""
        if not self.backtest_results:
            print("没有回测结果，请先运行回测")
            return
        
        # 获取ORB策略权益曲线
        equity_curve = self.backtest_results['equity_curve']
        
        # 生成QQQ的同期买入持有权益曲线
        # 为简化起见，使用首日和末日的收盘价计算买入持有收益
        start_date = equity_curve.index[0]
        end_date = equity_curve.index[-1]
        
        # 找到开始日期最接近的交易记录和结束日期最接近的交易记录
        start_price = self.data[self.data['Date'] >= start_date].iloc[0]['day_close']
        end_price = self.data[self.data['Date'] <= end_date].iloc[-1]['day_close']
        
        # 计算买入持有的收益曲线
        hold_shares = self.initial_capital / start_price
        buy_hold_curve = pd.DataFrame(index=equity_curve.index)
        
        # 简化模拟买入持有策略
        # 实际情况下，应该使用每日收盘价，这里我们做线性插值
        date_range = (end_date - start_date).days
        price_range = end_price - start_price
        
        daily_price_change = price_range / date_range
        
        buy_hold_curve['price'] = [start_price + daily_price_change * i for i in range(len(buy_hold_curve))]
        buy_hold_curve['capital'] = buy_hold_curve['price'] * hold_shares
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制权益曲线
        ax.plot(equity_curve.index, equity_curve['capital'], 'k-', linewidth=1.5, label='ORB QQQ')
        ax.plot(buy_hold_curve.index, buy_hold_curve['capital'], 'r--', linewidth=1.5, label='Buy&Hold QQQ')
        
        # 设置对数y轴
        ax.set_yscale('log')
        
        # 设置y轴为美元格式
        formatter = mtick.StrMethodFormatter('${x:,.0f}')
        ax.yaxis.set_major_formatter(formatter)
        
        # 设置x轴为年月格式
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        
        # 设置标题和标签
        ax.set_title('ORB Strategy vs Buy & Hold QQQ', fontsize=14, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Portfolio Value (Log Scale $)', fontsize=12)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 添加图例
        ax.legend(loc='upper left', frameon=True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"权益曲线对比图已保存至: {save_path}")
        
        plt.show()
    
    def plot_pnl_in_risk_units(self, save_path=None):
        """绘制PnL的风险单位表示（复现论文图2）"""
        if not self.trades:
            print("没有交易记录，无法绘制图表")
            return
        
        # 提取交易PnL（按R单位）
        trades_df = pd.DataFrame(self.trades)
        pnl_in_r = trades_df['pnl_in_r'].values
        
        # 计算统计数据
        avg_pnl_in_r = np.mean(pnl_in_r)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制柱状图
        bars = ax.bar(range(len(pnl_in_r)), pnl_in_r, width=0.8, 
                      color=np.where(pnl_in_r >= 0, 'skyblue', 'lightcoral'), 
                      alpha=0.7, edgecolor='blue')
        
        # 添加平均线
        ax.axhline(y=avg_pnl_in_r, color='red', linestyle='-', alpha=0.7, 
                   label=f'Average PnL = {avg_pnl_in_r:.2f}R')
        
        # 设置标题和标签
        ax.set_title('Daily PnL in Risk Units (R)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Trade #', fontsize=12)
        ax.set_ylabel('PnL (R)', fontsize=12)
        
        # 添加图例
        ax.legend(loc='upper right')
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 设置y轴限制以便更好地展示数据
        max_pnl = max(pnl_in_r)
        min_pnl = min(pnl_in_r)
        y_range = max(abs(max_pnl), abs(min_pnl))
        ax.set_ylim(min(-2, -y_range * 1.1), max(10, y_range * 1.1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"风险单位PnL图已保存至: {save_path}")
        
        plt.show()
    
    def generate_report(self, include_comparison=True, save_path=None):
        """生成回测报告"""
        if not self.backtest_results:
            print("没有回测结果，请先运行回测")
            return
        
        report = f"""
=============================================
          QQQ ORB Strategy Backtest Report
=============================================

Backtest Period: {self.start_date} to {self.end_date}
Initial Capital: ${self.initial_capital:.2f}

---------------------------------------------
                Backtest Results
---------------------------------------------
Total Trades: {self.backtest_results['total_trades']}
Profitable Trades: {self.backtest_results['profitable_trades']}
Win Rate: {self.backtest_results['win_rate']:.2%}

Total P&L: ${self.backtest_results['total_pnl']:.2f}
Total Return: {self.backtest_results['total_pnl_pct']:.2%}
Annualized Return: {self.backtest_results['annual_return']:.2%}
Average P&L in Risk Units (R): {self.backtest_results['avg_pnl_in_r']:.2f}

Maximum Drawdown: {self.backtest_results['max_drawdown']:.2%}
Sharpe Ratio: {self.backtest_results['sharpe_ratio']:.2f}

Final Capital: ${self.backtest_results['final_capital']:.2f}
"""
        
        # 添加交易统计
        if self.trades:
            # 计算R单位的盈亏分布
            trades_df = pd.DataFrame(self.trades)
            pnl_in_r = trades_df['pnl_in_r'].values
            
            profitable_trades_r = pnl_in_r[pnl_in_r > 0]
            losing_trades_r = pnl_in_r[pnl_in_r <= 0]
            
            avg_profit_r = np.mean(profitable_trades_r) if len(profitable_trades_r) > 0 else 0
            avg_loss_r = np.mean(losing_trades_r) if len(losing_trades_r) > 0 else 0
            
            report += f"""
---------------------------------------------
             R-Multiple Statistics
---------------------------------------------
Average Winning Trade: {avg_profit_r:.2f}R
Average Losing Trade: {avg_loss_r:.2f}R
Profit Factor: {abs(np.sum(profitable_trades_r) / np.sum(losing_trades_r)):.2f} (Sum of profits / Sum of losses)

Largest Winning Trade: {np.max(pnl_in_r):.2f}R
Largest Losing Trade: {np.min(pnl_in_r):.2f}R
"""
        
        report += "\n=============================================\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"报告已保存至: {save_path}")
        
        return report


def main():
    """主函数"""
    # 设置回测参数 - 与论文匹配
    start_date = '2016-01-01'
    end_date = '2023-02-17'
    initial_capital = 25000
    
    # ORB策略数据文件 - 使用预处理好的数据
    # 注意：这是假设的文件名，需要根据实际生成的文件名调整
    orb_data_path = 'data/QQQ_ORB_data_2016-01-01_to_2023-02-17.csv'
    
    # 检查数据文件是否存在
    if not os.path.exists(orb_data_path):
        print(f"ORB策略数据文件 {orb_data_path} 不存在！")
        print("请先使用ibkr_5min_data_fetcher.py获取数据")
        return
    
    # 创建输出目录
    output_dir = 'orb_backtest_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建并运行ORB策略回测
    orb_strategy = ORB_Strategy(orb_data_path, start_date, end_date, initial_capital)
    results = orb_strategy.run_backtest()
    
    # 生成报告
    report = orb_strategy.generate_report(
        include_comparison=True, 
        save_path=f"{output_dir}/QQQ_ORB_report.txt"
    )
    print(report)
    
    # 绘制权益曲线对比图（复现论文图1）
    orb_strategy.plot_equity_curve_comparison(
        save_path=f"{output_dir}/Figure1_QQQ_ORB_vs_BuyHold.png"
    )
    
    # 绘制PnL的风险单位表示（复现论文图2）
    orb_strategy.plot_pnl_in_risk_units(
        save_path=f"{output_dir}/Figure2_QQQ_ORB_PnL_in_R.png"
    )
    
    print(f"\n回测完成! 所有结果已保存到 {output_dir} 目录")


if __name__ == "__main__":
    main() 