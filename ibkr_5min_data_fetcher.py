"""
增强版 IBKR 5分钟数据获取器 - 支持ATR数据完整性
扩展获取日期范围，确保ATR指标能完整计算
"""

from ib_insync import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time as time_module
import config
import pytz
from zoneinfo import ZoneInfo
import logging
from market_calendar import MarketCalendar
import re


class Enhanced_IBKR_Fetcher:
    def __init__(self, host=config.IBKR_HOST, port=config.IBKR_PORT, client_id=config.IBKR_CLIENT_ID):
        """初始化增强版数据获取器"""
        # 设置日志
        logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
        self.logger = logging.getLogger(__name__)
        
        # 初始化IB连接
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.connected = False
        
        # 初始化市场日历
        try:
            self.market_calendar = MarketCalendar(config.TIMEZONE)
            self.ny_tz = ZoneInfo(config.TIMEZONE)
        except Exception as e:
            self.logger.error(f"初始化市场日历失败: {e}")
            raise
        
        # 连接到TWS
        self._connect()
    
    def _connect(self):
        """连接到TWS/IB Gateway"""
        try:
            self.ib.connect(self.host, self.port, self.client_id)
            self.connected = True
            self.logger.info("成功连接到IBKR")
        except Exception as e:
            self.logger.error(f"连接IBKR失败: {str(e)}")
            raise
    
    def get_contract(self, symbol):
        """获取合约信息"""
        try:
            contract = Stock(symbol.upper(), 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            return contract
        except Exception as e:
            self.logger.error(f"获取{symbol}合约失败: {e}")
            raise
    
    def get_5min_data(self, symbol, start_date, end_date, use_rth=True):
        """获取5分钟数据"""
        self.logger.info(f"获取{symbol}从{start_date}到{end_date}的5分钟数据...")
        
        try:
            contract = self.get_contract(symbol)
            
            # 转换日期
            start = pd.Timestamp(start_date).tz_localize(None).tz_localize(self.ny_tz)
            end = pd.Timestamp(end_date).tz_localize(None).tz_localize(self.ny_tz)
            
            # 获取交易日列表
            trading_days = self.market_calendar.get_trading_days(start, end)
            
            if len(trading_days) == 0:
                self.logger.warning("指定日期范围内没有交易日")
                return pd.DataFrame()
            
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
                    
                    # 请求历史数据
                    end_time = market_close.strftime('%Y%m%d 16:00:00')
                    
                    bars = self.ib.reqHistoricalData(
                        contract=contract,
                        endDateTime=end_time,
                        durationStr='1 D',
                        barSizeSetting='5 mins',
                        whatToShow='TRADES',
                        useRTH=use_rth,
                        formatDate=1
                    )
                    
                    if bars and len(bars) > 0:
                        df = util.df(bars)
                        
                        # 处理时间戳
                        df['date'] = pd.to_datetime(df['date'])
                        if df['date'].dt.tz is None:
                            df['date'] = df['date'].dt.tz_localize('UTC')
                        df['date'] = df['date'].dt.tz_convert(self.ny_tz)
                        
                        # 过滤交易时段数据
                        df = df[(df['date'] >= market_open) & (df['date'] <= market_close)]
                        
                        if not df.empty:
                            df['tradeDate'] = df['date'].dt.date
                            all_data.append(df)
                            self.logger.info(f"成功获取{trading_day}: {len(df)}条K线")
                    
                    time_module.sleep(config.REQUEST_DELAY)
                    
                except Exception as e:
                    self.logger.error(f"获取{trading_day}数据失败: {str(e)}")
                    continue
            
            if len(all_data) == 0:
                return pd.DataFrame()
            
            # 合并所有数据
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.sort_values('date')
            return combined_data
            
        except Exception as e:
            self.logger.error(f"获取5分钟数据失败: {e}")
            raise
    
    def get_orb_data(self, symbol, start_date, end_date, use_rth=True):
        """获取ORB策略所需数据"""
        data = self.get_5min_data(symbol, start_date, end_date, use_rth)
        
        if data.empty:
            return pd.DataFrame()
        
        orb_data = []
        for date, group in data.groupby('tradeDate'):
            try:
                # 时区处理
                date_ts = pd.Timestamp(date).tz_localize(None)
                market_open, _ = self.market_calendar.get_trading_hours(
                    date_ts.tz_localize(self.ny_tz)
                )
                
                if market_open is None:
                    continue
                
                # 获取开盘后的数据
                day_data = group[group['date'] >= market_open].sort_values('date')
                
                if len(day_data) < 2:
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
        
        if len(orb_data) == 0:
            return pd.DataFrame()
        
        orb_df = pd.DataFrame(orb_data)
        return orb_df
    
    def disconnect(self):
        """断开连接"""
        try:
            if hasattr(self, 'ib') and self.ib.isConnected():
                self.ib.disconnect()
                self.connected = False
                self.logger.info("已断开IBKR连接")
        except Exception as e:
            self.logger.error(f"断开连接时出错: {e}")


def calculate_atr_for_orb_data(orb_df, period=14):
    """为ORB数据计算并添加ATR指标"""
    df = orb_df.copy()
    
    # 确保日期排序
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 计算真实波幅 (True Range)
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # 计算ATR (N日平均真实波幅)
    df['atr'] = df['tr'].rolling(window=period).mean()
    
    # 删除中间计算列
    df = df.drop(['prev_close', 'tr1', 'tr2', 'tr3', 'tr'], axis=1)
    
    print(f"计算ATR完成，有效ATR数据行数: {df['atr'].notna().sum()}")
    return df


def get_symbol_5min_data_enhanced(symbol, start_date=config.START_DATE, 
                                 end_date=config.END_DATE, output_dir=config.DATA_DIR, 
                                 atr_period=14):
    """增强版数据获取：扩展日期范围确保ATR数据完整"""
    
    fetcher = Enhanced_IBKR_Fetcher()
    try:
        # 确保目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 为了计算ATR，扩展数据获取范围
        extended_start_date = pd.to_datetime(start_date) - pd.Timedelta(days=atr_period + 10)
        extended_start_str = extended_start_date.strftime('%Y-%m-%d')
        
        print(f"为计算ATR，扩展数据获取范围: {extended_start_str} 至 {end_date}")
        print(f"实际策略日期范围: {start_date} 至 {end_date}")
        
        # 获取完整5分钟数据（使用扩展的日期范围）
        full_data = fetcher.get_5min_data(symbol, extended_start_str, end_date)
        
        if not full_data.empty:
            # 保存5分钟数据（使用原始日期范围命名）
            full_filename = f"{symbol}_5min_full_{start_date}_to_{end_date}.csv"
            full_data_path = os.path.join(output_dir, full_filename)
            full_data.to_csv(full_data_path, index=False)
            print(f"5分钟数据已保存: {full_data_path} ({len(full_data)}行)")
            
            # 获取ORB策略所需数据（使用扩展的日期范围）
            orb_data = fetcher.get_orb_data(symbol, extended_start_str, end_date)
            
            if not orb_data.empty:
                # 计算ATR并添加到ORB数据
                orb_data_with_atr = calculate_atr_for_orb_data(orb_data, atr_period)
                
                # 保存ORB数据
                orb_filename = f"{symbol}_ORB_data_{start_date}_to_{end_date}.csv"
                orb_data_path = os.path.join(output_dir, orb_filename)
                orb_data_with_atr.to_csv(orb_data_path, index=False, float_format='%.4f')
                print(f"ORB数据已保存: {orb_data_path} ({len(orb_data_with_atr)}行)")
                
                return {
                    'full_data': {'success': True, 'path': full_data_path, 'rows': len(full_data)},
                    'orb_data': {'success': True, 'path': orb_data_path, 'rows': len(orb_data_with_atr)}
                }
        
        return {
            'full_data': {'success': False, 'error': 'No data returned'},
            'orb_data': {'success': False, 'error': 'No data returned'}
        }
        
    except Exception as e:
        error_msg = f"获取数据时出错: {str(e)}"
        print(error_msg)
        return {
            'full_data': {'success': False, 'error': error_msg},
            'orb_data': {'success': False, 'error': error_msg}
        }
    finally:
        fetcher.disconnect()


def main():
    """主函数"""
    try:
        print(f"开始获取增强版数据, 时间范围: {config.START_DATE} 至 {config.END_DATE}")
        print(f"交易品种: {', '.join(config.SYMBOLS)}")
        
        # 获取所有品种的数据
        for symbol in config.SYMBOLS:
            print(f"\n开始获取 {symbol} 数据...")
            result = get_symbol_5min_data_enhanced(symbol)
            
            print(f"\n{symbol} 结果:")
            if result['full_data']['success']:
                print(f"  5分钟数据: {result['full_data']['rows']}行")
            else:
                print(f"  5分钟数据失败: {result['full_data']['error']}")
            
            if result['orb_data']['success']:
                print(f"  ORB数据: {result['orb_data']['rows']}行")
            else:
                print(f"  ORB数据失败: {result['orb_data']['error']}")
        
    except Exception as e:
        print(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 