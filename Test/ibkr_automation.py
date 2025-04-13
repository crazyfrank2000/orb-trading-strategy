from ib_insync import *
import pandas as pd
from datetime import datetime

class IBKRAutomation:
    def __init__(self):
        self.ib = IB()
        try:
            print("尝试连接到 TWS...")
            self.ib.connect('127.0.0.1', 7497, clientId=2)
            
            # 添加连接状态检查
            if self.ib.isConnected():
                print("成功连接到 TWS!")
            else:
                print("连接过程未报错，但未成功建立连接")
        except Exception as e:
            print(f"连接错误: {e}")
            print("请确保 TWS 已运行并且 API 连接已启用")
            raise
        
    def get_account_info(self):
        """获取账户信息"""
        try:
            accounts = self.ib.managedAccounts()
            if not accounts:
                print("警告: 未能获取账户列表")
                return {}
            
            account = accounts[0]  # 获取第一个账户
            print(f"正在获取账户 {account} 的信息...")
            
            # 获取账户摘要
            account_summary = self.ib.accountSummary(account)
            
            # 将数据整理成字典格式
            summary_dict = {}
            for item in account_summary:
                summary_dict[item.tag] = {
                    'value': item.value,
                    'currency': item.currency
                }
            
            return summary_dict
        except Exception as e:
            print(f"获取账户信息时出错: {e}")
            return {}
    
    def get_portfolio(self):
        """获取投资组合信息"""
        portfolio = self.ib.portfolio()
        
        # 将数据转换为更易读的格式
        portfolio_data = []
        for item in portfolio:
            portfolio_data.append({
                'symbol': item.contract.symbol,
                'position': item.position,
                'marketPrice': item.marketPrice,
                'marketValue': item.marketValue,
                'averageCost': item.averageCost,
                'unrealizedPNL': item.unrealizedPNL
            })
            
        return portfolio_data
    
    def get_market_data(self, symbols=['QQQ', 'SPY']):
        """获取实时市场数据 - 改进版"""
        market_data = {}
        
        for symbol in symbols:
            try:
                # 更详细的合约定义
                contract = Stock(symbol, 'SMART', 'USD')
                self.ib.qualifyContracts(contract)
                
                print(f"正在获取 {symbol} 的市场数据...")
                
                # 使用 reqMktData 替代方案
                # 方法1: 使用 reqTickByTickData
                self.ib.reqTickByTickData(contract, 'Last', 0, False)
                ticker = self.ib.reqMktData(contract, '', False, False)
                
                # 等待更长时间获取数据
                self.ib.sleep(3)  # 增加到3秒
                
                # 检查并打印原始ticker数据进行调试
                print(f"获取到的原始 {symbol} 数据: {ticker}")
                
                # 记录数据，处理空值
                market_data[symbol] = {
                    'last': ticker.last if hasattr(ticker, 'last') and ticker.last else '未获取',
                    'bid': ticker.bid if hasattr(ticker, 'bid') and ticker.bid else '未获取',
                    'ask': ticker.ask if hasattr(ticker, 'ask') and ticker.ask else '未获取',
                    'volume': ticker.volume if hasattr(ticker, 'volume') and ticker.volume else '未获取',
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # 取消数据请求
                self.ib.cancelMktData(contract)
                
            except Exception as e:
                print(f"获取 {symbol} 市场数据时出错: {str(e)}")
                market_data[symbol] = {'error': str(e)}
                
        return market_data
    
    def get_latest_price_from_history(self, symbols=['QQQ', 'SPY']):
        """通过历史数据获取最近价格"""
        market_data = {}
        
        for symbol in symbols:
            try:
                contract = Stock(symbol, 'SMART', 'USD')
                self.ib.qualifyContracts(contract)
                
                print(f"正在获取 {symbol} 的历史数据...")
                
                # 获取最近的历史数据
                bars = self.ib.reqHistoricalData(
                    contract,
                    endDateTime='',
                    durationStr='1 D',
                    barSizeSetting='1 min',
                    whatToShow='TRADES',
                    useRTH=True
                )
                
                if bars and len(bars) > 0:
                    latest_bar = bars[-1]  # 获取最新的一条数据
                    market_data[symbol] = {
                        'date': latest_bar.date,
                        'open': latest_bar.open,
                        'high': latest_bar.high,
                        'low': latest_bar.low,
                        'close': latest_bar.close,
                        'volume': latest_bar.volume
                    }
                else:
                    market_data[symbol] = {'error': '未获取到历史数据'}
                    
            except Exception as e:
                print(f"获取 {symbol} 历史数据时出错: {str(e)}")
                market_data[symbol] = {'error': str(e)}
                
        return market_data
    
    def close(self):
        """关闭连接"""
        self.ib.disconnect()

def main():
    try:
        ibkr = IBKRAutomation()
        
        # 1. 获取账户信息
        print("\n=== 账户信息 ===")
        account_info = ibkr.get_account_info()
        
        if account_info:
            for key, value in account_info.items():
                print(f"{key}: {value.get('value')} {value.get('currency')}")
        else:
            print("未能获取账户信息")
        
        # 2. 获取投资组合信息
        print("\n=== 投资组合信息 ===")
        portfolio = ibkr.get_portfolio()
        if portfolio:
            for position in portfolio:
                print(f"股票: {position['symbol']}")
                print(f"持仓: {position['position']}")
                print(f"市值: {position['marketValue']}")
                print(f"未实现盈亏: {position['unrealizedPNL']}\n")
        else:
            print("投资组合为空或无法获取投资组合信息")
        
        # 3. 首先尝试获取实时市场数据
        print("\n=== 尝试获取实时市场数据 ===")
        market_data = ibkr.get_market_data()
        
        # 检查是否成功获取了所有实时数据
        all_data_available = True
        for symbol, data in market_data.items():
            if data.get('last') == '未获取' and data.get('bid') == '未获取' and data.get('ask') == '未获取':
                all_data_available = False
                print(f"{symbol} 实时数据未能获取")
        
        # 如果实时数据获取失败，尝试获取历史数据
        if not all_data_available:
            print("\n=== 尝试从历史数据获取最近价格 ===")
            historical_data = ibkr.get_latest_price_from_history()
            
            for symbol, data in historical_data.items():
                if 'error' not in data:
                    print(f"\n{symbol} (历史数据):")
                    print(f"日期: {data['date']}")
                    print(f"开盘价: {data['open']}")
                    print(f"最高价: {data['high']}")
                    print(f"最低价: {data['low']}")
                    print(f"收盘价: {data['close']}")
                    print(f"成交量: {data['volume']}")
                else:
                    print(f"\n{symbol}: {data['error']}")
        
        # 打印获取到的实时数据
        print("\n=== 实时市场数据结果 ===")
        for symbol, data in market_data.items():
            if 'error' not in data:
                print(f"\n{symbol}:")
                print(f"最新价格: {data['last']}")
                print(f"买入价: {data['bid']}")
                print(f"卖出价: {data['ask']}")
                print(f"成交量: {data['volume']}")
                print(f"更新时间: {data['time']}")
            else:
                print(f"\n{symbol}: {data['error']}")
                
    except Exception as e:
        print(f"程序执行错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保程序结束时关闭连接
        if 'ibkr' in locals() and hasattr(ibkr, 'close'):
            ibkr.close()

if __name__ == "__main__":
    main() 