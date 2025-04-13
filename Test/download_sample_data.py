"""
下载样本数据用于测试ORB策略
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def download_sample_data(symbol, start_date, end_date, output_dir='data'):
    """
    使用yfinance下载样本数据
    """
    print(f"下载 {symbol} 从 {start_date} 到 {end_date} 的样本数据...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    csv_file = f"{output_dir}/{symbol}_5min_data.csv"
    
    # 转换日期格式
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    all_data = []
    current_start = start
    
    # yfinance每次只能下载60天的5分钟数据，因此需要分批下载
    while current_start < end:
        current_end = min(current_start + timedelta(days=60), end)
        
        # 格式化日期
        cs_str = current_start.strftime('%Y-%m-%d')
        ce_str = current_end.strftime('%Y-%m-%d')
        
        print(f"下载 {cs_str} 到 {ce_str} 的数据...")
        
        # 下载5分钟K线数据
        df = yf.download(symbol, start=cs_str, end=ce_str, interval="5m")
        
        if not df.empty:
            all_data.append(df)
        
        current_start = current_end + timedelta(days=1)
    
    if not all_data:
        print(f"未能获取 {symbol} 的历史数据")
        return False
    
    # 合并所有数据
    combined_data = pd.concat(all_data)
    
    # 重命名列以匹配IBKR格式
    combined_data = combined_data.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    # 添加日期列
    combined_data['date'] = combined_data.index
    combined_data['Date'] = combined_data.index.date
    
    # 保存到CSV
    combined_data.to_csv(csv_file)
    print(f"数据已保存至: {csv_file}")
    print(f"总共 {len(combined_data)} 条记录")
    
    return True

def main():
    symbols = ['QQQ', 'SPY']
    start_date = '2022-01-01'
    end_date = '2022-12-31'
    
    for symbol in symbols:
        success = download_sample_data(symbol, start_date, end_date)
        if success:
            print(f"{symbol} 数据下载成功")
        else:
            print(f"{symbol} 数据下载失败")

if __name__ == "__main__":
    main() 