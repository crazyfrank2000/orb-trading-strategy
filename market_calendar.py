"""
美股市场交易日历模块
提供交易日判断和交易时间获取功能
"""

import pandas as pd
from datetime import datetime, time, timedelta
import holidays
from zoneinfo import ZoneInfo
import logging


class MarketCalendar:
    """美股市场交易日历"""
    
    def __init__(self, timezone='America/New_York'):
        """初始化市场日历
        
        Args:
            timezone: 时区，默认为纽约时区
        """
        self.timezone = timezone
        self.tz = ZoneInfo(timezone)
        self.logger = logging.getLogger(__name__)
        
        # 美股常规交易时间
        self.market_open_time = time(9, 30)    # 09:30
        self.market_close_time = time(16, 0)   # 16:00
        
        # 美股节假日
        self.us_holidays = holidays.UnitedStates(years=range(2020, 2030))
        
        # 额外的市场休市日（半日交易等）
        self.additional_holidays = self._get_additional_holidays()
        
        # 半日交易日（提前收盘到13:00）
        self.early_close_days = self._get_early_close_days()
    
    def _get_additional_holidays(self):
        """获取额外的市场休市日"""
        additional = set()
        
        # 可以添加特殊的休市日，比如：
        # - 总统葬礼日
        # - 9/11等特殊事件导致的休市
        # additional.add(datetime(2001, 9, 11).date())
        
        return additional
    
    def _get_early_close_days(self):
        """获取提前收盘日（通常是13:00收盘）"""
        early_close = set()
        
        # 常见的提前收盘日：
        # - 感恩节后的黑色星期五
        # - 圣诞节前夕
        # - 独立日前夕（如果是平日）
        
        for year in range(2020, 2030):
            # 感恩节后的黑色星期五
            thanksgiving = self._get_thanksgiving(year)
            black_friday = thanksgiving + timedelta(days=1)
            early_close.add(black_friday.date())
            
            # 圣诞节前夕（如果是平日）
            christmas_eve = datetime(year, 12, 24).date()
            if christmas_eve.weekday() < 5:  # 周一到周五
                early_close.add(christmas_eve)
            
            # 独立日前夕（如果独立日是周一且前夜是平日）
            independence_day = datetime(year, 7, 4).date()
            if independence_day.weekday() == 0:  # 如果7月4日是周一
                july_3 = datetime(year, 7, 3).date()
                if july_3.weekday() < 5:  # 7月3日是平日
                    early_close.add(july_3)
        
        return early_close
    
    def _get_thanksgiving(self, year):
        """获取感恩节日期（11月第四个周四）"""
        november_first = datetime(year, 11, 1)
        # 找到11月第一个周四
        first_thursday = november_first + timedelta(days=(3 - november_first.weekday()) % 7)
        # 加三周得到第四个周四
        thanksgiving = first_thursday + timedelta(weeks=3)
        return thanksgiving
    
    def is_trading_day(self, date):
        """判断是否为交易日
        
        Args:
            date: 日期对象或字符串
            
        Returns:
            bool: 是否为交易日
        """
        if isinstance(date, str):
            date = pd.to_datetime(date).date()
        elif isinstance(date, pd.Timestamp):
            date = date.date()
        elif isinstance(date, datetime):
            date = date.date()
        
        # 周六周日不是交易日
        if date.weekday() >= 5:
            return False
        
        # 检查是否是节假日
        if date in self.us_holidays:
            return False
        
        # 检查额外的休市日
        if date in self.additional_holidays:
            return False
        
        return True
    
    def get_trading_days(self, start_date, end_date):
        """获取指定日期范围内的所有交易日
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            list: 交易日列表
        """
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # 确保去除时区信息，只保留日期
        if hasattr(start_date, 'tz') and start_date.tz is not None:
            start_date = start_date.tz_localize(None)
        if hasattr(end_date, 'tz') and end_date.tz is not None:
            end_date = end_date.tz_localize(None)
        
        trading_days = []
        current_date = start_date
        
        while current_date <= end_date:
            if self.is_trading_day(current_date):
                trading_days.append(current_date.date())
            current_date += timedelta(days=1)
        
        return trading_days
    
    def get_trading_hours(self, date):
        """获取指定日期的交易时间
        
        Args:
            date: 日期对象（带时区信息）
            
        Returns:
            tuple: (market_open, market_close) 时间戳，如果不是交易日则返回(None, None)
        """
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        # 确保有时区信息
        if hasattr(date, 'tz') and date.tz is None:
            date = date.tz_localize(self.tz)
        elif not hasattr(date, 'tz'):
            date = pd.to_datetime(date).tz_localize(self.tz)
        
        # 检查是否为交易日
        if not self.is_trading_day(date.date()):
            return None, None
        
        # 构建当天的开盘和收盘时间
        market_open = date.replace(
            hour=self.market_open_time.hour,
            minute=self.market_open_time.minute,
            second=0,
            microsecond=0
        )
        
        # 检查是否为提前收盘日
        if date.date() in self.early_close_days:
            market_close = date.replace(hour=13, minute=0, second=0, microsecond=0)
        else:
            market_close = date.replace(
                hour=self.market_close_time.hour,
                minute=self.market_close_time.minute,
                second=0,
                microsecond=0
            )
        
        return market_open, market_close
    
    def get_next_trading_day(self, date):
        """获取下一个交易日
        
        Args:
            date: 基准日期
            
        Returns:
            date: 下一个交易日
        """
        if isinstance(date, str):
            date = pd.to_datetime(date).date()
        elif isinstance(date, pd.Timestamp):
            date = date.date()
        elif isinstance(date, datetime):
            date = date.date()
        
        next_date = date + timedelta(days=1)
        while not self.is_trading_day(next_date):
            next_date += timedelta(days=1)
        
        return next_date
    
    def get_previous_trading_day(self, date):
        """获取前一个交易日
        
        Args:
            date: 基准日期
            
        Returns:
            date: 前一个交易日
        """
        if isinstance(date, str):
            date = pd.to_datetime(date).date()
        elif isinstance(date, pd.Timestamp):
            date = date.date()
        elif isinstance(date, datetime):
            date = date.date()
        
        prev_date = date - timedelta(days=1)
        while not self.is_trading_day(prev_date):
            prev_date -= timedelta(days=1)
        
        return prev_date


def test_market_calendar():
    """测试市场日历功能"""
    calendar = MarketCalendar()
    
    # 测试交易日判断
    print("=== 交易日测试 ===")
    test_dates = [
        '2024-01-01',  # 新年，休市
        '2024-01-02',  # 周二，交易日
        '2024-01-06',  # 周六，休市
        '2024-07-04',  # 独立日，休市
        '2024-12-25',  # 圣诞节，休市
    ]
    
    for date_str in test_dates:
        is_trading = calendar.is_trading_day(date_str)
        print(f"{date_str}: {'交易日' if is_trading else '休市'}")
    
    # 测试获取交易日列表
    print("\n=== 交易日列表测试 ===")
    trading_days = calendar.get_trading_days('2024-01-01', '2024-01-10')
    print(f"2024-01-01 到 2024-01-10 的交易日：")
    for day in trading_days:
        print(f"  {day}")
    
    # 测试交易时间
    print("\n=== 交易时间测试 ===")
    test_date = pd.to_datetime('2024-01-02').tz_localize('America/New_York')
    market_open, market_close = calendar.get_trading_hours(test_date)
    if market_open:
        print(f"2024-01-02 交易时间：{market_open} - {market_close}")
    else:
        print("2024-01-02 非交易日")


if __name__ == "__main__":
    test_market_calendar() 