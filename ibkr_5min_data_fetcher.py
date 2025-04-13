"""
专门为ORB策略获取IBKR 5分钟数据
使用配置文件中的参数设置
"""

from ib_insync import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import os
import time as time_module
import config  # 导入配置文件
import pytz
from zoneinfo import ZoneInfo
import logging
from market_calendar import MarketCalendar

class IBKR5MinDataFetcher:
    def __init__(self, host=config.IBKR_HOST, port=config.IBKR_PORT, client_id=config.IBKR_CLIENT_ID):
        # 设置日志
        self._setup_logging()
        
        # 初始化IB连接
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        
        # 初始化市场日历
        self.market_calendar = MarketCalendar(config.TIMEZONE)
        self.ny_tz = ZoneInfo(config.TIMEZONE)
        
        # 连接到TWS
        self._connect()
        
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=config.LOG_LEVEL,
            format=config.LOG_FORMAT
        )
        self.logger = logging.getLogger(__name__)
        
    def _connect(self):
        """连接到TWS/IB Gateway"""
        try:
            self.ib.connect(self.host, self.port, self.client_id)
            self.logger.info("成功连接到IBKR")
        except Exception as e:
            self.logger.error(f"连接IBKR失败: {str(e)}")
            raise
            
    def get_contract(self, symbol):
        """获取合约信息"""
        contract = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)
        return contract
        
    def get_historical_data_with_retry(self, contract, end_datetime, duration, 
                                     bar_size, use_rth):
        """带重试机制的历史数据获取"""
        for attempt in range(config.MAX_RETRIES):
            try:
                # 修改时间格式为 US/Eastern
                if isinstance(end_datetime, str):
                    end_datetime = pd.Timestamp(end_datetime, tz=self.ny_tz)
                
                # 格式化为IBKR要求的格式
                formatted_datetime = end_datetime.strftime('%Y%m%d %H:%M:%S US/Eastern')
                
                bars = self.ib.reqHistoricalData(
                    contract=contract,
                    endDateTime=formatted_datetime,  # 使用带时区的时间格式
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow='TRADES',
                    useRTH=use_rth,
                    formatDate=1
                )
                
                # 添加检查
                if bars is None or len(bars) == 0:
                    self.logger.warning(f"未获取到数据，尝试次数: {attempt + 1}")
                    if attempt < config.MAX_RETRIES - 1:
                        time_module.sleep(config.RETRY_DELAY)
                        continue
                return bars
                
            except Exception as e:
                self.logger.warning(f"第{attempt + 1}次获取数据失败: {str(e)}")
                if attempt < config.MAX_RETRIES - 1:
                    time_module.sleep(config.RETRY_DELAY)
                else:
                    raise
                    
    def get_5min_data(self, symbol, start_date, end_date, use_rth=True):
        """获取5分钟数据"""
        self.logger.info(f"获取{symbol}从{start_date}到{end_date}的5分钟数据...")
        
        contract = self.get_contract(symbol)
        
        # 转换日期并确保使用正确的时区
        start = pd.Timestamp(start_date).tz_localize(None).tz_localize(self.ny_tz)
        end = pd.Timestamp(end_date).tz_localize(None).tz_localize(self.ny_tz)
        
        # 获取交易日列表
        trading_days = self.market_calendar.get_trading_days(start, end)
        
        all_data = []
        for trading_day in trading_days:
            try:
                # 获取市场交易时间
                trading_day_ts = pd.Timestamp(trading_day).tz_localize(None)
                market_open, market_close = self.market_calendar.get_trading_hours(
                    trading_day_ts.tz_localize(self.ny_tz)
                )
                
                if market_open is None:
                    continue
                    
                # 确保使用正确的收盘时间
                end_time = market_close.strftime('%Y%m%d 16:00:00')
                
                # 获取当天数据
                bars = self.get_historical_data_with_retry(
                    contract=contract,
                    end_datetime=end_time,
                    duration='1 D',
                    bar_size='5 mins',
                    use_rth=use_rth
                )
                
                if bars and len(bars) > 0:
                    df = util.df(bars)
                    
                    # 处理时间戳
                    df['date'] = pd.to_datetime(df['date'])
                    if df['date'].dt.tz is None:
                        df['date'] = df['date'].dt.tz_localize('UTC')
                    df['date'] = df['date'].dt.tz_convert(self.ny_tz)
                    
                    # 过滤交易时段数据
                    df = df[
                        (df['date'] >= market_open) &
                        (df['date'] <= market_close)
                    ]
                    
                    if not df.empty:
                        # 添加交易日期
                        df['tradeDate'] = df['date'].dt.date
                        
                        all_data.append(df)
                        self.logger.info(f"成功获取{len(df)}条K线数据")
                    else:
                        self.logger.warning(f"过滤后数据为空: {trading_day}")
                else:
                    self.logger.warning(f"未获取到{trading_day}的数据")
                
            except Exception as e:
                self.logger.error(f"获取{trading_day}数据失败: {str(e)}")
                
            time_module.sleep(config.REQUEST_DELAY)
            
        if not all_data:
            return pd.DataFrame()
            
        # 合并数据
        combined_data = pd.concat(all_data)
        combined_data = combined_data.sort_values('date')
        
        return combined_data
        
    def get_orb_data(self, symbol, start_date, end_date, use_rth=True):
        """获取ORB策略所需数据"""
        data = self.get_5min_data(symbol, start_date, end_date, use_rth)
        
        if data.empty:
            return pd.DataFrame()
            
        orb_data = []
        for date, group in data.groupby('tradeDate'):
            try:
                # 修改这部分的时区处理
                date_ts = pd.Timestamp(date).tz_localize(None)  # 先移除时区
                market_open, _ = self.market_calendar.get_trading_hours(
                    date_ts.tz_localize(self.ny_tz)  # 再添加时区
                )
                
                if market_open is None:
                    continue
                    
                # 获取开盘后的数据
                day_data = group[group['date'] >= market_open].sort_values('date')
                
                if len(day_data) < 2:
                    self.logger.warning(f"{date}的有效数据少于2根K线，跳过")
                    continue
                    
                # 验证第一根K线时间
                first_bar_time = day_data.iloc[0]['date']
                if first_bar_time.time() != market_open.time():
                    self.logger.warning(
                        f"{date}的第一根K线时间（{first_bar_time.time()}）"
                        f"不是开盘时间（{market_open.time()}），跳过"
                    )
                    continue
                    
                # 获取前两根K线的高低点
                orb_high = day_data.iloc[:2]['high'].max()
                orb_low = day_data.iloc[:2]['low'].min()
                
                # 获取当天OHLC
                daily_data = {
                    'date': date,
                    'open': day_data.iloc[0]['open'],
                    'high': day_data['high'].max(),
                    'low': day_data['low'].min(),
                    'close': day_data.iloc[-1]['close'],
                    'orb_high': orb_high,
                    'orb_low': orb_low
                }
                
                orb_data.append(daily_data)
                
            except Exception as e:
                self.logger.error(f"处理{date}数据时出错: {str(e)}")
                continue
                
        if not orb_data:
            return pd.DataFrame()
            
        orb_df = pd.DataFrame(orb_data)
        return orb_df
        
    def __enter__(self):
        """上下文管理器入口"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.ib.disconnect()
        
    def __del__(self):
        """析构函数"""
        if self.ib.isConnected():
            self.ib.disconnect()

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