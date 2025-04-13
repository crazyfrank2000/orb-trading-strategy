"""
测试IBKR连接和数据获取
"""

from ib_insync import *
import time
import sys

def test_connection():
    """测试IBKR API连接"""
    ib = IB()
    
    try:
        print("尝试连接到IBKR API...")
        ib.connect('127.0.0.1', 7497, clientId=999)
        
        if ib.isConnected():
            print(f"连接成功! TWS/Gateway版本: {ib.client.serverVersion()}")
            print("检查连接状态...")
            time.sleep(1)
            
            # 获取账户列表
            accounts = ib.managedAccounts()
            print(f"账户列表: {accounts}")
            
            # 检查合约是否可用
            print("\n测试合约查询...")
            contract = Stock('QQQ', 'SMART', 'USD')
            qualified_contracts = ib.qualifyContracts(contract)
            
            if qualified_contracts:
                print(f"合约查询成功: {qualified_contracts[0]}")
            else:
                print("合约查询失败!")
            
            # 测试历史数据请求
            print("\n测试历史数据请求...")
            if qualified_contracts:
                bars = ib.reqHistoricalData(
                    qualified_contracts[0],
                    endDateTime='',
                    durationStr='1 D',
                    barSizeSetting='1 hour',
                    whatToShow='TRADES',
                    useRTH=True
                )
                
                if bars:
                    print(f"成功获取 {len(bars)} 条历史数据")
                    print(f"第一条数据: {bars[0]}")
                else:
                    print("未能获取历史数据!")
            
            return True
        
        else:
            print("连接失败!")
            return False
            
    except Exception as e:
        print(f"连接或数据请求错误: {str(e)}")
        return False
    
    finally:
        # 断开连接
        if ib.isConnected():
            ib.disconnect()
            print("已断开连接")

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1) 