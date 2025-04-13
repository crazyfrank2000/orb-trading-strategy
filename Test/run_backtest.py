"""
ORB策略回测主程序
"""

from run_ibkr_backtest import run_multiple_symbols
from config import SYMBOLS, START_DATE, END_DATE, INITIAL_CAPITAL, DATA_SOURCE

def main():
    """
    运行ORB策略回测
    """
    print(f"开始对以下标的进行ORB策略回测: {', '.join(SYMBOLS)}")
    print(f"回测期间: {START_DATE} 至 {END_DATE}")
    print(f"初始资金: ${INITIAL_CAPITAL}")
    print(f"数据源: {DATA_SOURCE}")
    
    # 运行回测
    run_multiple_symbols(SYMBOLS, START_DATE, END_DATE, INITIAL_CAPITAL, data_source=DATA_SOURCE)
    
    print("\n回测完成! 结果已保存到 backtest_results 目录.")

if __name__ == "__main__":
    main() 