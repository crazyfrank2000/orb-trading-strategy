"""
ORB策略系统运行脚本
集成数据获取和回测功能
"""

import os
import argparse
import config
from ibkr_5min_data_fetcher import fetch_all_symbols
from ORB_Strategy_Backtest import run_all_symbols_backtest, generate_summary_report


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ORB策略系统 - 数据获取和回测')
    
    parser.add_argument('--mode', type=str, choices=['fetch', 'backtest', 'all'], default='all',
                        help='运行模式: fetch=仅获取数据, backtest=仅回测, all=获取数据并回测')
    
    parser.add_argument('--symbols', type=str, nargs='+',
                        help='交易品种代码，多个用空格分隔 (如 --symbols QQQ SPY)')
    
    parser.add_argument('--start', type=str, help='开始日期，格式YYYY-MM-DD')
    parser.add_argument('--end', type=str, help='结束日期，格式YYYY-MM-DD')
    
    parser.add_argument('--capital', type=float, help='初始资金')
    
    return parser.parse_args()


def update_config(args):
    """根据命令行参数更新配置"""
    if args.symbols:
        config.SYMBOLS = args.symbols
        print(f"已设置交易品种: {', '.join(config.SYMBOLS)}")
    
    if args.start:
        config.START_DATE = args.start
        print(f"已设置开始日期: {config.START_DATE}")
    
    if args.end:
        config.END_DATE = args.end
        print(f"已设置结束日期: {config.END_DATE}")
    
    if args.capital:
        config.INITIAL_CAPITAL = args.capital
        print(f"已设置初始资金: ${config.INITIAL_CAPITAL}")


def run_data_fetching():
    """运行数据获取"""
    print("\n" + "="*50)
    print("开始获取 ORB 策略数据")
    print("="*50)
    
    # 确保数据目录存在
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)
    
    # 获取所有品种数据
    fetch_all_symbols()


def run_backtest():
    """运行回测"""
    print("\n" + "="*50)
    print("开始运行 ORB 策略回测")
    print("="*50)
    
    # 确保输出目录存在
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
    
    # 运行所有品种回测
    results = run_all_symbols_backtest()
    
    # 生成汇总报告
    summary_path = f"{config.OUTPUT_DIR}/ORB_Summary_Report.txt"
    summary = generate_summary_report(results, save_path=summary_path)
    print("\n" + summary)
    
    print(f"\n所有回测完成！详细结果已保存到 {config.OUTPUT_DIR} 目录")


def main():
    """主函数"""
    args = parse_args()
    update_config(args)
    
    print("ORB策略系统")
    print(f"交易品种: {', '.join(config.SYMBOLS)}")
    print(f"日期范围: {config.START_DATE} 至 {config.END_DATE}")
    print(f"初始资金: ${config.INITIAL_CAPITAL:.2f}")
    
    # 根据模式运行不同功能
    if args.mode in ['fetch', 'all']:
        run_data_fetching()
    
    if args.mode in ['backtest', 'all']:
        run_backtest()


if __name__ == "__main__":
    main() 