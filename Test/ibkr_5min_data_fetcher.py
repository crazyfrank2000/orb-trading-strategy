"""
专门为ORB策略获取IBKR 5分钟数据
使用配置文件中的参数设置
"""

from ib_insync import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
import config  # 导入配置文件

class IBKR5MinDataFetcher:
    def __init__(self, host=config.IBKR_HOST, port=config.IBKR_PORT, client_id=config.IBKR_CLIENT_ID):
        """初始化IBKR连接"""
        self.ib = IB()
        try:
            self.ib.connect(host, port, clientId=client_id)
            print(f"已连接到IBKR (TWS/Gateway), 服务器版本: {self.ib.client.serverVersion()}")
        except Exception as e:
            print(f"连接IBKR失败: {str(e)}")
            print("请确保TWS或IB Gateway已运行并且API连接已启用")
            raise

    def __del__(self):
        """在对象销毁时确保断开连接"""
        if hasattr(self, 'ib') and self.ib.isConnected():
            self.ib.disconnect()
            print("已断开IBKR连接")

    def get_contract(self, symbol, sec_type='STK', exchange='SMART', currency='USD'):
        """获取合约对象"""
        print(f"创建{symbol}合约...")
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency

        # 确保合约有效
        qualified = self.ib.qualifyContracts(contract)
        if not qualified:
            raise ValueError(f"无法获取合格的合约: {symbol}")
        
        print(f"成功创建合约: {qualified[0]}")
        return qualified[0]

    def get_5min_data(self, symbol, start_date, end_date, use_rth=config.USE_RTH):
        """获取指定日期范围内的5分钟数据"""
        print(f"获取{symbol}从{start_date}到{end_date}的5分钟数据...")
        
        contract = self.get_contract(symbol)
        
        # 将字符串日期转换为datetime对象
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # 添加一天以包含结束日期
        end = end + timedelta(days=1)
        
        all_data = []
        current_date = start
        
        # 每次获取一天的数据
        while current_date < end:
            next_date = current_date + timedelta(days=1)
            
            # 格式化日期字符串 (YYYYMMDD HH:MM:SS)
            end_datetime = next_date.strftime('%Y%m%d 00:00:00')
            
            print(f"获取{current_date.strftime('%Y-%m-%d')}的数据...")
            
            try:
                # 请求历史数据 - 明确指定5分钟时间段
                bars = self.ib.reqHistoricalData(
                    contract=contract,
                    endDateTime=end_datetime,
                    durationStr='1 D',  # 获取1天的数据
                    barSizeSetting='5 mins',  # 5分钟K线
                    whatToShow='TRADES',
                    useRTH=use_rth,
                    formatDate=1  # 1表示为'YYYYMMDD HH:MM:SS'格式
                )
                
                if bars:
                    # 将数据转换为DataFrame
                    df = util.df(bars)
                    
                    # 标记交易日期
                    df['tradeDate'] = current_date.strftime('%Y-%m-%d')
                    
                    # 添加到结果集
                    all_data.append(df)
                    print(f"成功获取{len(df)}条5分钟K线数据")
                else:
                    print(f"警告: {current_date.strftime('%Y-%m-%d')}没有数据")
                
                # 防止请求过快被限制
                time.sleep(1)
                
            except Exception as e:
                print(f"获取{current_date.strftime('%Y-%m-%d')}数据时出错: {str(e)}")
            
            # 移至下一天
            current_date = next_date
        
        if not all_data:
            print("未能获取任何数据")
            return pd.DataFrame()
        
        # 合并所有数据
        combined_data = pd.concat(all_data)
        
        # 转换日期格式和时区
        combined_data['date'] = pd.to_datetime(combined_data['date'])
        combined_data['Date'] = combined_data['date'].dt.date
        
        print(f"总共获取{len(combined_data)}条5分钟K线数据")
        return combined_data

    def get_orb_data(self, symbol, start_date, end_date, use_rth=config.USE_RTH):
        """
        获取ORB策略所需的关键数据:
        1. 每天的前两个5分钟K线
        2. 每天的OHLC价格
        """
        print(f"获取{symbol}的ORB策略数据...")
        
        # 获取所有5分钟数据
        data = self.get_5min_data(symbol, start_date, end_date, use_rth)
        
        if data.empty:
            return pd.DataFrame()
        
        # 按日期分组
        grouped = data.groupby('Date')
        
        orb_data = []
        
        for date, group in grouped:
            # 按时间排序，确保按时间顺序
            day_data = group.sort_values('date')
            
            if len(day_data) < 2:
                print(f"警告: {date}的数据少于2根K线，跳过")
                continue
            
            # 提取前两根5分钟K线
            first_candle = day_data.iloc[0]
            second_candle = day_data.iloc[1]
            
            # 计算当天OHLC
            day_open = first_candle['open']
            day_high = day_data['high'].max()
            day_low = day_data['low'].min()
            day_close = day_data.iloc[-1]['close']
            
            # 创建该日的记录
            day_record = {
                'Date': date,
                'first_5min_open': first_candle['open'],
                'first_5min_high': first_candle['high'],
                'first_5min_low': first_candle['low'],
                'first_5min_close': first_candle['close'],
                'second_5min_open': second_candle['open'],
                'day_open': day_open,
                'day_high': day_high,
                'day_low': day_low,
                'day_close': day_close,
                'direction': 1 if first_candle['close'] > first_candle['open'] else -1,
                'is_doji': abs(first_candle['close'] - first_candle['open']) < 0.0001
            }
            
            orb_data.append(day_record)
        
        # 创建DataFrame
        orb_df = pd.DataFrame(orb_data)
        
        print(f"成功生成{len(orb_df)}天的ORB策略数据")
        return orb_df
    
    def save_data(self, data, file_path):
        """保存数据到CSV文件"""
        data.to_csv(file_path, index=False)
        print(f"数据已保存至: {file_path}")
        return file_path


def get_symbol_5min_data(symbol, start_date=config.START_DATE, end_date=config.END_DATE, output_dir=config.DATA_DIR):
    """获取指定股票的5分钟数据并保存"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    fetcher = IBKR5MinDataFetcher()
    
    try:
        # 获取完整5分钟数据
        full_data = fetcher.get_5min_data(symbol, start_date, end_date)
        if not full_data.empty:
            full_data_path = f"{output_dir}/{symbol}_5min_full_{start_date}_to_{end_date}.csv"
            fetcher.save_data(full_data, full_data_path)
            
            # 获取ORB策略所需数据
            orb_data = fetcher.get_orb_data(symbol, start_date, end_date)
            if not orb_data.empty:
                orb_data_path = f"{output_dir}/{symbol}_ORB_data_{start_date}_to_{end_date}.csv"
                fetcher.save_data(orb_data, orb_data_path)
                return {
                    'full_data': {'success': True, 'path': full_data_path, 'rows': len(full_data)},
                    'orb_data': {'success': True, 'path': orb_data_path, 'rows': len(orb_data)}
                }
        
        return {
            'full_data': {'success': False, 'error': 'No data returned'},
            'orb_data': {'success': False, 'error': 'No data returned'}
        }
    
    except Exception as e:
        print(f"获取数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'full_data': {'success': False, 'error': str(e)},
            'orb_data': {'success': False, 'error': str(e)}
        }
    finally:
        del fetcher  # 确保断开连接


def fetch_all_symbols():
    """获取配置中所有交易品种的数据"""
    results = {}
    for symbol in config.SYMBOLS:
        print(f"\n开始获取 {symbol} 数据...")
        symbol_result = get_symbol_5min_data(symbol)
        results[symbol] = symbol_result
    
    return results


def main():
    """主函数"""
    print(f"开始获取数据, 时间范围: {config.START_DATE} 至 {config.END_DATE}")
    print(f"交易品种: {', '.join(config.SYMBOLS)}")
    
    # 创建数据目录
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)
    
    # 获取所有品种的数据
    results = fetch_all_symbols()
    
    # 输出结果摘要
    print("\n==== 数据获取完成 ====")
    for symbol, result in results.items():
        print(f"\n{symbol}:")
        if result['full_data']['success']:
            print(f"  完整5分钟数据: {result['full_data']['rows']}行, 保存至{result['full_data']['path']}")
        else:
            print(f"  完整5分钟数据获取失败: {result['full_data']['error']}")
        
        if result['orb_data']['success']:
            print(f"  ORB策略数据: {result['orb_data']['rows']}行, 保存至{result['orb_data']['path']}")
        else:
            print(f"  ORB策略数据获取失败: {result['orb_data']['error']}")


if __name__ == "__main__":
    main() 