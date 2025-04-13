import pandas_market_calendars as mcal
from datetime import datetime, time
import pytz
from zoneinfo import ZoneInfo
import pandas as pd

class MarketCalendar:
    def __init__(self, timezone='America/New_York'):
        self.nyse = mcal.get_calendar('NYSE')
        self.tz = ZoneInfo(timezone)
        
    def is_trading_day(self, date):
        """检查是否为交易日"""
        try:
            # 确保日期是datetime对象
            if isinstance(date, str):
                date = pd.Timestamp(date)
            schedule = self.nyse.schedule(
                start_date=date.date(),
                end_date=date.date()
            )
            return not schedule.empty
        except Exception as e:
            print(f"检查交易日时出错: {str(e)}")
            return False
    
    def get_trading_hours(self, date):
        """获取指定日期的交易时间"""
        try:
            # 确保日期是datetime对象
            if isinstance(date, str):
                date = pd.Timestamp(date)
            schedule = self.nyse.schedule(
                start_date=date.date(),
                end_date=date.date()
            )
            
            if schedule.empty:
                return None, None
                
            market_open = schedule.iloc[0]['market_open'].astimezone(self.tz)
            market_close = schedule.iloc[0]['market_close'].astimezone(self.tz)
            
            return market_open, market_close
        except Exception as e:
            print(f"获取交易时间时出错: {str(e)}")
            return None, None
        
    def is_early_close_day(self, date):
        """检查是否为提前收市日"""
        try:
            _, close_time = self.get_trading_hours(date)
            if close_time is None:
                return False
            return close_time.time() < time(16, 0)
        except Exception as e:
            print(f"检查提前收市日时出错: {str(e)}")
            return False
        
    def get_trading_days(self, start_date, end_date):
        """获取指定日期范围内的所有交易日"""
        try:
            # 确保日期是datetime对象
            if isinstance(start_date, str):
                start_date = pd.Timestamp(start_date)
            if isinstance(end_date, str):
                end_date = pd.Timestamp(end_date)
                
            schedule = self.nyse.schedule(
                start_date=start_date.date(),
                end_date=end_date.date()
            )
            return schedule.index.date
        except Exception as e:
            print(f"获取交易日列表时出错: {str(e)}")
            return [] 