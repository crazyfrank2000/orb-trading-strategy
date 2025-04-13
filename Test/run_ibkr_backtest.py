import os
from orb_backtest_ibkr import ORBBacktest
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compare_with_buy_and_hold(symbol, start_date, end_date, initial_capital=25000):
    """比较ORB策略与买入持有策略的表现"""
    
    # 创建输出目录
    output_dir = 'backtest_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 运行ORB策略回测
    orb = ORBBacktest(symbol, start_date, end_date, initial_capital, data_source='ibkr')
    orb_results = orb.run_backtest()
    
    # 如果没有交易记录，直接返回
    if orb_results['total_trades'] == 0:
        print(f"警告: {symbol} 没有产生任何交易")
        return None
    
    # 获取同期的每日收盘价数据 (从我们的IBKR数据中提取)
    daily_data = orb.data.groupby('Date')['Close'].last().to_frame()
    daily_data.index = pd.to_datetime(daily_data.index)
    
    if daily_data.empty:
        print(f"无法获取 {symbol} 的历史数据")
        return None
    
    # 计算买入持有策略的收益率
    start_price = daily_data['Close'].iloc[0]
    end_price = daily_data['Close'].iloc[-1]
    
    buy_hold_shares = initial_capital / start_price
    buy_hold_final = buy_hold_shares * end_price
    buy_hold_return = (buy_hold_final - initial_capital) / initial_capital
    
    # 计算买入持有策略的最大回撤
    daily_data['Portfolio'] = daily_data['Close'] * buy_hold_shares
    daily_data['Cummax'] = daily_data['Portfolio'].cummax()
    daily_data['Drawdown'] = (daily_data['Cummax'] - daily_data['Portfolio']) / daily_data['Cummax']
    buy_hold_max_drawdown = daily_data['Drawdown'].max()
    
    # 计算买入持有策略的夏普比率
    daily_data['Daily_Return'] = daily_data['Close'].pct_change()
    buy_hold_sharpe = daily_data['Daily_Return'].mean() / daily_data['Daily_Return'].std() * np.sqrt(252) if daily_data['Daily_Return'].std() > 0 else 0
    
    # 打印比较结果
    print("\n========== 策略比较 ==========")
    print(f"交易标的: {symbol}")
    print(f"回测期间: {start_date} 至 {end_date}")
    print(f"初始资金: ${initial_capital:.2f}")
    
    print("\n----- ORB策略 -----")
    print(f"期末资金: ${orb_results['final_capital']:.2f}")
    print(f"总收益率: {orb_results['total_pnl_pct']:.2%}")
    print(f"最大回撤: {orb_results['max_drawdown']:.2%}")
    print(f"夏普比率: {orb_results['sharpe_ratio']:.2f}")
    
    print("\n----- 买入持有策略 -----")
    print(f"期末资金: ${buy_hold_final:.2f}")
    print(f"总收益率: {buy_hold_return:.2%}")
    print(f"最大回撤: {buy_hold_max_drawdown:.2%}")
    print(f"夏普比率: {buy_hold_sharpe:.2f}")
    
    # 绘制策略对比图
    plt.figure(figsize=(15, 10))
    
    # 计算ORB策略的每日净值
    orb_daily_returns = orb_results['daily_returns']
    orb_equity_curve = [initial_capital]
    
    # 转换索引确保可以比较
    orb_daily_returns.index = pd.to_datetime(orb_daily_returns.index)
    
    # 创建完整的日期范围
    all_dates = pd.date_range(start=daily_data.index.min(), end=daily_data.index.max(), freq='B')
    
    # 初始化ORB策略的权益曲线
    orb_equity = pd.Series(initial_capital, index=[all_dates[0] - pd.Timedelta(days=1)])
    
    # 计算每日权益
    for date in all_dates:
        if date in orb_daily_returns.index:
            orb_equity[date] = orb_equity.iloc[-1] * (1 + orb_daily_returns[date])
        else:
            orb_equity[date] = orb_equity.iloc[-1]
    
    # 绘制净值曲线对比
    plt.subplot(2, 1, 1)
    plt.plot(daily_data.index, daily_data['Portfolio'], label='买入持有策略')
    plt.plot(orb_equity.index[1:], orb_equity.values[1:], label='ORB策略')
    plt.title('策略净值对比')
    plt.xlabel('日期')
    plt.ylabel('净值 ($)')
    plt.legend()
    plt.grid(True)
    
    # 绘制回撤对比
    plt.subplot(2, 1, 2)
    
    # 计算ORB策略的回撤
    orb_equity_series = orb_equity[1:]  # 移除初始值
    orb_cummax = orb_equity_series.cummax()
    orb_drawdown = (orb_cummax - orb_equity_series) / orb_cummax
    
    plt.plot(daily_data.index, daily_data['Drawdown'], label='买入持有策略回撤')
    plt.plot(orb_drawdown.index, orb_drawdown.values, label='ORB策略回撤')
    plt.title('策略回撤对比')
    plt.xlabel('日期')
    plt.ylabel('回撤')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(f"{output_dir}/{symbol}_strategy_comparison.png")
    plt.show()
    
    return {
        'orb': {
            'final_capital': orb_results['final_capital'],
            'total_return': orb_results['total_pnl_pct'],
            'max_drawdown': orb_results['max_drawdown'],
            'sharpe_ratio': orb_results['sharpe_ratio']
        },
        'buy_hold': {
            'final_capital': buy_hold_final,
            'total_return': buy_hold_return,
            'max_drawdown': buy_hold_max_drawdown,
            'sharpe_ratio': buy_hold_sharpe
        }
    }

def run_multiple_symbols(symbols, start_date, end_date, initial_capital=25000):
    """运行多个交易标的的回测"""
    results = {}
    
    for symbol in symbols:
        print(f"\n\n====== 开始对 {symbol} 进行回测 ======\n")
        
        # 创建回测对象
        orb = ORBBacktest(symbol, start_date, end_date, initial_capital, data_source='ibkr')
        
        try:
            # 运行回测
            backtest_results = orb.run_backtest()
            
            # 保存结果
            output_dir = 'backtest_results'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 生成报告
            report = orb.generate_report(save_path=f"{output_dir}/{symbol}_orb_report.txt")
            
            # 绘制图表
            orb.plot_results(save_path=f"{output_dir}/{symbol}_orb_results.png")
            
            # 与买入持有策略比较
            compare_results = compare_with_buy_and_hold(symbol, start_date, end_date, initial_capital)
            
            # 保存结果
            results[symbol] = {
                'backtest_results': backtest_results,
                'comparison': compare_results
            }
            
        except Exception as e:
            print(f"{symbol} 回测失败: {str(e)}")
            results[symbol] = {'error': str(e)}
    
    # 生成综合报告
    generate_summary_report(results, start_date, end_date, initial_capital)
    
    return results

def generate_summary_report(results, start_date, end_date, initial_capital):
    """生成所有标的的综合回测报告"""
    output_dir = 'backtest_results'
    
    with open(f"{output_dir}/summary_report.txt", 'w') as f:
        f.write("=============================================\n")
        f.write("            ORB策略综合回测报告              \n")
        f.write("=============================================\n\n")
        
        f.write(f"回测期间: {start_date} 至 {end_date}\n")
        f.write(f"初始资金: ${initial_capital:.2f}\n\n")
        
        f.write("----------------- 回测结果 -----------------\n\n")
        
        # 创建表格格式
        f.write(f"{'标的':^10}{'总交易':^10}{'胜率':^10}{'总收益率':^12}{'最大回撤':^12}{'夏普比率':^10}{'vs买入持有':^15}\n")
        f.write("-" * 80 + "\n")
        
        for symbol, result in results.items():
            if 'error' in result:
                f.write(f"{symbol:^10}回测失败: {result['error']}\n")
                continue
            
            backtest = result['backtest_results']
            comparison = result['comparison']
            
            if not comparison:
                comparison_str = "无法比较"
            else:
                orb_return = comparison['orb']['total_return']
                bh_return = comparison['buy_hold']['total_return']
                diff = orb_return - bh_return
                comparison_str = f"{diff:+.2%}"
            
            f.write(f"{symbol:^10}{backtest['total_trades']:^10}{backtest['win_rate']:.2%}^10{backtest['total_pnl_pct']:.2%}^12{backtest['max_drawdown']:.2%}^12{backtest['sharpe_ratio']:.2f}^10{comparison_str:^15}\n")
        
        f.write("\n=============================================\n")
    
    print(f"综合报告已保存至 {output_dir}/summary_report.txt")

def main():
    """主函数"""
    # 设置回测参数
    symbols = ['QQQ', 'SPY']  # 可以添加更多标的
    start_date = '2022-01-01'
    end_date = '2022-12-31'
    initial_capital = 25000
    
    # 运行多个标的回测
    results = run_multiple_symbols(symbols, start_date, end_date, initial_capital)
    
    print("\n所有回测完成！")

if __name__ == "__main__":
    main() 