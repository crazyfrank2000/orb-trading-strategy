from ib_insync import *
import pandas as pd
from datetime import datetime, timedelta
import time
import os

class IBKRDataFetcher:
    """使用IBKR API获取历史数据的类"""
    
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        """
        初始化IBKR数据获取器
        
        参数:
        host (str): TWS/IB Gateway的主机地址，默认为本地
        port (int): TWS/IB Gateway的端口，默认为7497(模拟账户)
        client_id (int): 客户端ID，默认为1
        """
        self.ib = IB()
        try:
            self.ib.connect(host, port, clientId=client_id)
            print(f"成功连接到IBKR API (TWS/Gateway)，服务器版本: {self.ib.client.serverVersion()}")
        except Exception as e:
            print(f"连接IBKR API失败: {str(e)}")
            print("请确保TWS或IB Gateway正在运行，并已允许API连接")
            raise
    
    def __del__(self):
        """析构函数，确保断开连接"""
        if hasattr(self, 'ib') and self.ib.isConnected():
            self.ib.disconnect()
            print("已断开与IBKR API的连接")
    
    def get_contract(self, symbol, sec_type='STK', exchange='SMART', currency='USD'):
        """
        创建并返回一个合约对象
        
        参数:
        symbol (str): 股票代码
        sec_type (str): 证券类型，默认为股票
        exchange (str): 交易所，默认为SMART
        currency (str): 货币，默认为USD
        
        返回:
        Contract: 合格的合约对象
        """
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency
        
        # 确保合约有效
        qualified_contracts = self.ib.qualifyContracts(contract)
        
        if not qualified_contracts:
            raise ValueError(f"无法获取合格的合约: {symbol}")
        
        return qualified_contracts[0]
    
    def get_historical_data(self, symbol, start_date, end_date, bar_size='5 mins',
                           what_to_show='TRADES', use_rth=True):
        """
        获取历史数据 - 增强版
        """
        print(f"正在获取 {symbol} 的历史数据...")
        
        try:
            # 创建合约
            contract = self.get_contract(symbol)
            print(f"成功创建 {symbol} 合约")
            
            # 转换日期格式
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            # IBKR限制每次请求的数据量，需要分批获取
            all_data = []
            current_end = end
            
            while current_end >= start:
                # 向前推30天或到开始日期
                current_start = max(start, current_end - timedelta(days=30))
                
                # 格式化日期字符串
                end_str = current_end.strftime('%Y%m%d %H:%M:%S')
                
                print(f"获取 {symbol} 数据段: {current_start.strftime('%Y-%m-%d')} 到 {current_end.strftime('%Y-%m-%d')}")
                
                try:
                    # 请求历史数据 - 添加更多调试信息
                    print(f"发送请求: {contract.symbol}, 结束时间: {end_str}, 持续时间: {(current_end - current_start).days + 1} 天")
                    
                    # 请求历史数据
                    bars = self.ib.reqHistoricalData(
                        contract=contract,
                        endDateTime=end_str,
                        durationStr=f"{(current_end - current_start).days + 1} D",
                        barSizeSetting=bar_size,
                        whatToShow=what_to_show,
                        useRTH=use_rth,
                        formatDate=1
                    )
                    
                    print(f"请求完成，获取到 {len(bars) if bars else 0} 条数据")
                    
                    if bars:
                        # 转换为DataFrame
                        df = util.df(bars)
                        all_data.append(df)
                        print(f"成功获取 {len(df)} 条记录")
                    else:
                        print(f"警告: 该时间段未返回数据")
                    
                    # 防止请求过快被限制
                    print("等待1秒...")
                    time.sleep(1)
                
                except Exception as e:
                    print(f"获取时间段数据失败: {str(e)}")
                    # 继续尝试其他时间段
                
                # 更新结束日期为当前开始日期的前一天
                current_end = current_start - timedelta(days=1)
            
            if not all_data:
                print("未获取到任何数据")
                return pd.DataFrame()  # 返回空DataFrame而不是引发异常
            
            # 合并所有批次数据
            data = pd.concat(all_data)
            
            # 按日期升序排序
            data = data.sort_index()
            
            # 添加日期列
            data['date'] = pd.to_datetime(data['date'])
            data['Date'] = data['date'].dt.date
            
            print(f"成功获取 {len(data)} 条 {symbol} 的历史数据")
            return data
        
        except Exception as e:
            print(f"获取历史数据时发生错误: {str(e)}")
            print("详细错误信息:")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()  # 返回空DataFrame而不是引发异常
    
    def save_data_to_csv(self, data, symbol, output_dir='data'):
        """将数据保存为CSV文件"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        file_path = f"{output_dir}/{symbol}_5min_data.csv"
        data.to_csv(file_path)
        print(f"数据已保存至: {file_path}")
        
        return file_path

def fetch_and_save_data(symbols, start_date, end_date, output_dir='data'):
    """获取多个股票的历史数据并保存"""
    fetcher = IBKRDataFetcher()
    
    try:
        results = {}
        
        for symbol in symbols:
            try:
                data = fetcher.get_historical_data(symbol, start_date, end_date)
                file_path = fetcher.save_data_to_csv(data, symbol, output_dir)
                results[symbol] = {
                    'success': True,
                    'rows': len(data),
                    'file_path': file_path
                }
            except Exception as e:
                print(f"获取 {symbol} 数据失败: {str(e)}")
                results[symbol] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    finally:
        # 确保断开连接
        del fetcher

if __name__ == "__main__":
    # 测试数据获取
    symbols = ['QQQ']
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    results = fetch_and_save_data(symbols, start_date, end_date)
    
    for symbol, result in results.items():
        if result['success']:
            print(f"{symbol}: 成功获取 {result['rows']} 条数据，保存至 {result['file_path']}")
        else:
            print(f"{symbol}: 获取失败 - {result['error']}") 