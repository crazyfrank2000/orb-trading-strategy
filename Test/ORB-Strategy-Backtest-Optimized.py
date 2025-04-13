"""
ORB (Opening Range Breakout) Strategy Backtest - Optimized Version
Based on pre-processed ORB data from IBKR to improve backtest accuracy
Implementation of strategy from paper "Can Day Trading Really Be Profitable?"
Uses config file for easier parameter management and supports multiple symbols
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import config  # 导入配置文件

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
    def __init__(self, symbol, start_date=None, end_date=None, initial_capital=None):
        """
        初始化ORB策略回测
        
        参数:
        symbol (str): 交易品种代码
        start_date (str): 回测开始日期，格式'YYYY-MM-DD'，若为None则使用配置
        end_date (str): 回测结束日期，格式'YYYY-MM-DD'，若为None则使用配置
        initial_capital (float): 初始资金，若为None则使用配置
        """
        self.symbol = symbol
        self.start_date = start_date if start_date else config.START_DATE
        self.end_date = end_date if end_date else config.END_DATE
        self.initial_capital = initial_capital if initial_capital else config.INITIAL_CAPITAL
        self.current_capital = self.initial_capital
        
        # 从配置文件获取策略参数
        self.commission_per_share = config.COMMISSION_PER_SHARE
        self.max_leverage = config.MAX_LEVERAGE
        self.risk_per_trade = config.RISK_PER_TRADE
        self.take_profit_r = config.TAKE_PROFIT_R
        
        # 交易记录
        self.trades = []
        
        # 构建ORB数据文件路径
        orb_data_path = f"{config.DATA_DIR}/{self.symbol}_ORB_data_{self.start_date}_to_{self.end_date}.csv"
        
        # 检查数据文件是否存在
        if not os.path.exists(orb_data_path):
            raise FileNotFoundError(f"ORB数据文件不存在: {orb_data_path}，请先使用fetcher获取数据")
        
        # 读取ORB专用数据
        self.data = self._load_orb_data(orb_data_path)
        
        # 回测结果
        self.backtest_results = None
    
    def _load_orb_data(self, data_path):
        """加载ORB专用数据"""
        print(f"加载 {self.symbol} ORB策略数据: {data_path}")
        
        data = pd.read_csv(data_path)
        
        # 确保Date列为日期类型
        data['Date'] = pd.to_datetime(data['Date'])
        
        # 筛选日期范围
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        data = data[(data['Date'] >= start) & (data['Date'] <= end)]
        
        if data.empty:
            raise ValueError(f"数据筛选后为空，请检查日期范围是否有效")
        
        print(f"成功加载{len(data)}天的 {self.symbol} ORB策略数据")
        return data
    
    def run_backtest(self):
        """运行ORB策略回测"""
        print(f"开始回测 {self.symbol} ORB策略 ({self.start_date} 至 {self.end_date})...")
        
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
            
            # 设置止盈目标（配置的R倍数）
            take_profit_price = entry_price + (self.take_profit_r * risk * direction)
            
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
                'symbol': self.symbol,
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
            'symbol': self.symbol,
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
        
        # 生成交易品种的同期买入持有权益曲线
        start_date = equity_curve.index[0]
        end_date = equity_curve.index[-1]
        
        # 找到开始日期最接近的交易记录和结束日期最接近的交易记录
        start_price = self.data[self.data['Date'] >= start_date].iloc[0]['day_close']
        end_price = self.data[self.data['Date'] <= end_date].iloc[-1]['day_close']
        
        # 计算买入持有的收益曲线
        hold_shares = self.initial_capital / start_price
        buy_hold_curve = pd.DataFrame(index=equity_curve.index)
        
        # 简化模拟买入持有策略
        date_range = (end_date - start_date).days
        price_range = end_price - start_price
        
        daily_price_change = price_range / date_range if date_range > 0 else 0
        
        buy_hold_curve['price'] = [start_price + daily_price_change * i for i in range(len(buy_hold_curve))]
        buy_hold_curve['capital'] = buy_hold_curve['price'] * hold_shares
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制权益曲线
        ax.plot(equity_curve.index, equity_curve['capital'], 'k-', linewidth=1.5, label=f'ORB {self.symbol}')
        ax.plot(buy_hold_curve.index, buy_hold_curve['capital'], 'r--', linewidth=1.5, label=f'Buy&Hold {self.symbol}')
        
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
        ax.set_title(f'ORB Strategy vs Buy & Hold {self.symbol}', fontsize=14, fontweight='bold')
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
        ax.set_title(f'{self.symbol} Daily PnL in Risk Units (R)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Trade #', fontsize=12)
        ax.set_ylabel('PnL (R)', fontsize=12)
        
        # 添加图例
        ax.legend(loc='upper right')
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 设置y轴限制以便更好地展示数据
        max_pnl = max(pnl_in_r) if len(pnl_in_r) > 0 else 0 
        min_pnl = min(pnl_in_r) if len(pnl_in_r) > 0 else 0
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
         {self.symbol} ORB Strategy Backtest Report
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
            
            profit_factor = abs(np.sum(profitable_trades_r) / np.sum(losing_trades_r)) if np.sum(losing_trades_r) != 0 else float('inf')
            
            report += f"""
---------------------------------------------
             R-Multiple Statistics
---------------------------------------------
Average Winning Trade: {avg_profit_r:.2f}R
Average Losing Trade: {avg_loss_r:.2f}R
Profit Factor: {profit_factor:.2f} (Sum of profits / Sum of losses)

Largest Winning Trade: {np.max(pnl_in_r) if len(pnl_in_r) > 0 else 0:.2f}R
Largest Losing Trade: {np.min(pnl_in_r) if len(pnl_in_r) > 0 else 0:.2f}R
"""
        
        report += "\n=============================================\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"报告已保存至: {save_path}")
        
        return report


def run_symbol_backtest(symbol, start_date=None, end_date=None, initial_capital=None):
    """运行单个交易品种的回测"""
    # 使用配置文件中的日期和资金，除非明确指定
    start_date = start_date or config.START_DATE
    end_date = end_date or config.END_DATE
    initial_capital = initial_capital or config.INITIAL_CAPITAL
    
    # 创建输出目录
    output_dir = f"{config.OUTPUT_DIR}/{symbol}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # 创建并运行ORB策略回测
        orb_strategy = ORB_Strategy(symbol, start_date, end_date, initial_capital)
        results = orb_strategy.run_backtest()
        
        # 生成报告
        report_path = f"{output_dir}/{symbol}_ORB_report.txt"
        report = orb_strategy.generate_report(save_path=report_path)
        print(report)
        
        # 绘制权益曲线对比图（复现论文图1）
        equity_curve_path = f"{output_dir}/{symbol}_ORB_vs_BuyHold.png"
        orb_strategy.plot_equity_curve_comparison(save_path=equity_curve_path)
        
        # 绘制PnL的风险单位表示（复现论文图2）
        pnl_r_path = f"{output_dir}/{symbol}_ORB_PnL_in_R.png"
        orb_strategy.plot_pnl_in_risk_units(save_path=pnl_r_path)
        
        print(f"{symbol} ORB策略回测完成! 结果已保存到 {output_dir}")
        
        return {
            'strategy': orb_strategy,
            'results': results,
            'report_path': report_path,
            'equity_curve_path': equity_curve_path,
            'pnl_r_path': pnl_r_path
        }
    
    except Exception as e:
        print(f"运行 {symbol} 回测时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_all_symbols_backtest():
    """运行所有配置的交易品种回测"""
    results = {}
    
    for symbol in config.SYMBOLS:
        print(f"\n{'='*50}")
        print(f"开始 {symbol} 回测")
        print(f"{'='*50}")
        
        result = run_symbol_backtest(symbol)
        results[symbol] = result
    
    return results


def generate_summary_report(results, save_path=None):
    """生成所有品种的汇总报告"""
    if not results:
        print("没有回测结果可供汇总")
        return
    
    summary = f"""
=============================================
      ORB策略回测汇总报告
=============================================

回测期间: {config.START_DATE} 至 {config.END_DATE}
初始资金: ${config.INITIAL_CAPITAL:.2f}

---------------------------------------------
              各品种回测结果
---------------------------------------------
"""
    
    # 表格标题
    summary += f"{'品种':<10}{'总交易次数':<12}{'胜率':<10}{'年化收益':<12}{'最大回撤':<12}{'夏普比率':<12}{'最终资金':<15}\n"
    summary += "-" * 80 + "\n"
    
    # 按品种添加结果
    for symbol, result in results.items():
        if result and result['results']:
            r = result['results']
            summary += f"{symbol:<10}{r['total_trades']:<12}{r['win_rate']:.2%:<10}{r['annual_return']:.2%:<12}{r['max_drawdown']:.2%:<12}{r['sharpe_ratio']:.2f:<12}${r['final_capital']:.2f:<15}\n"
        else:
            summary += f"{symbol:<10}回测失败或无结果\n"
    
    summary += "\n=============================================\n"
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(summary)
        print(f"汇总报告已保存至: {save_path}")
    
    return summary


def main():
    """主函数"""
    print(f"ORB策略回测系统 - 优化版")
    print(f"回测期间: {config.START_DATE} 至 {config.END_DATE}")
    print(f"交易品种: {', '.join(config.SYMBOLS)}")
    print(f"初始资金: ${config.INITIAL_CAPITAL:.2f}")
    
    # 创建输出主目录
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
    
    # 运行所有品种回测
    results = run_all_symbols_backtest()
    
    # 生成汇总报告
    summary_path = f"{config.OUTPUT_DIR}/ORB_Summary_Report.txt"
    summary = generate_summary_report(results, save_path=summary_path)
    print("\n" + summary)
    
    print(f"\n所有回测完成！详细结果已保存到 {config.OUTPUT_DIR} 目录")


if __name__ == "__main__":
    main() 