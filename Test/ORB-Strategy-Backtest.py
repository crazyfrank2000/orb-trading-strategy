import os
import importlib.util
import config  # 添加config模块导入

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


def main():
    """主函数"""
    print(f"ORB策略回测系统")
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