"""
ORB (Opening Range Breakout) Strategy Backtest - QQQ Version
Based on IBKR data, implementing the ORB strategy from the paper "Can Day Trading Really Be Profitable?"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from ibkr_data_fetcher import IBKRDataFetcher, fetch_and_save_data

# Set matplotlib to use a font that supports both English and Chinese
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# Set higher DPI for better image quality
plt.rcParams['figure.dpi'] = 120
# Use a more professional style (compatible with different matplotlib versions)
try:
    # 新版matplotlib (3.6+)
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        # 旧版matplotlib
        plt.style.use('seaborn-whitegrid')
    except:
        # 如果都不支持，使用默认的网格样式
        plt.style.use('default')
        plt.rcParams['axes.grid'] = True
        print("注意: 使用默认样式代替seaborn-whitegrid")

class ORB_QQQ_Strategy:
    def __init__(self, start_date, end_date, initial_capital=25000):
        """
        Initialize ORB strategy backtest - QQQ specific
        
        Parameters:
        start_date (str): Backtest start date, format 'YYYY-MM-DD'
        end_date (str): Backtest end date, format 'YYYY-MM-DD'
        initial_capital (float): Initial capital, default $25,000
        """
        self.symbol = 'QQQ'
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_per_share = 0.0005  # Commission per share
        self.max_leverage = 4  # Maximum leverage
        self.risk_per_trade = 0.01  # Risk per trade (1%)
        
        # Trade records
        self.trades = []
        
        # Get historical data
        self.data = self._get_data()
        
        # Backtest results
        self.backtest_results = None
    
    def _get_data(self):
        """Get QQQ historical 5-minute candle data, preferably from IBKR"""
        data_dir = 'data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        csv_file = f"{data_dir}/{self.symbol}_5min_data.csv"
        
        # Check if CSV file exists
        if not os.path.exists(csv_file):
            print(f"Getting 5-minute historical data for {self.symbol} from IBKR...")
            
            try:
                # Get data and save
                result = fetch_and_save_data(
                    [self.symbol], 
                    self.start_date, 
                    self.end_date,
                    data_dir
                )
                
                if not result[self.symbol]['success']:
                    raise ValueError(f"Failed to get data: {result[self.symbol]['error']}")
                
            except Exception as e:
                print(f"Unable to get data from IBKR, will try alternative methods: {str(e)}")
                raise
        else:
            print(f"Found existing CSV data file: {csv_file}")
        
        # Load data from CSV file
        print(f"Loading {self.symbol} data from CSV file...")
        data = pd.read_csv(csv_file, parse_dates=['date'])
        
        # Ensure Date column is date type
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date']).dt.date
        else:
            data['Date'] = pd.to_datetime(data['date']).dt.date
        
        # Filter date range
        start = datetime.strptime(self.start_date, '%Y-%m-%d').date()
        end = datetime.strptime(self.end_date, '%Y-%m-%d').date()
        
        data = data[(data['Date'] >= start) & (data['Date'] <= end)]
        
        # Ensure column name consistency (IBKR data format may differ from what the backtest system needs)
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        # Check if columns exist, then rename
        for old_col, new_col in column_mapping.items():
            if old_col in data.columns and new_col not in data.columns:
                data = data.rename(columns={old_col: new_col})
        
        print(f"Successfully loaded {len(data)} 5-minute candles for {self.symbol}")
        return data
    
    def _calculate_position_size(self, entry_price, stop_loss_price):
        """
        Calculate position size, according to the formula in the paper:
        Shares = int[min((A * 0.01 / R), (4 * A / P))]
        
        A: Account capital
        R: Distance between entry price and stop loss
        P: Entry price
        """
        risk = abs(entry_price - stop_loss_price)
        risk_amount = self.current_capital * self.risk_per_trade
        
        # Risk-based position size
        risk_based_shares = int(risk_amount / risk)
        
        # Leverage-based position size
        leverage_based_shares = int((self.max_leverage * self.current_capital) / entry_price)
        
        # Take the smaller of the two
        shares = min(risk_based_shares, leverage_based_shares)
        
        return shares
    
    def run_backtest(self):
        """Run ORB strategy backtest for QQQ"""
        print(f"Starting ORB strategy backtest for QQQ ({self.start_date} to {self.end_date})...")
        
        # Group by date
        grouped_data = self.data.groupby('Date')
        
        # Track date processing status
        total_days = len(grouped_data)
        days_with_trades = 0
        days_skipped_cross = 0
        days_skipped_data = 0
        
        print(f"Total of {total_days} trading days")
        
        for date, day_data in grouped_data:
            # Ensure there are enough candles to trade
            if len(day_data) < 2:
                days_skipped_data += 1
                continue
            
            # Extract the first 5-minute data
            first_5min = day_data.iloc[0]
            
            # Check if it's a doji (open price equals close price)
            if abs(first_5min['Open'] - first_5min['Close']) < 0.0001:
                days_skipped_cross += 1
                continue  # If it's a doji, don't trade that day
            
            # Determine direction (if the first candle's close is higher than open, go long; otherwise, go short)
            direction = 1 if first_5min['Close'] > first_5min['Open'] else -1
            
            # Determine entry price (opening price of the second 5-minute candle)
            entry_price = day_data.iloc[1]['Open']
            
            # Determine stop loss price
            stop_loss_price = first_5min['Low'] if direction == 1 else first_5min['High']
            
            # Check if risk is zero (to prevent division by zero)
            risk = abs(entry_price - stop_loss_price)
            if risk < 0.0001:
                continue
            
            # Set take profit target (10R)
            take_profit_price = entry_price + (10 * risk * direction)
            
            # Calculate position size
            shares = self._calculate_position_size(entry_price, stop_loss_price)
            
            # Skip this trade if position size is 0
            if shares == 0:
                continue
            
            # Calculate commission
            commission = shares * self.commission_per_share
            
            # Record trade
            trade = {
                'date': date,
                'entry_time': day_data.iloc[1].name,
                'direction': direction,
                'entry_price': entry_price,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'shares': shares,
                'risk_amount': self.current_capital * self.risk_per_trade,
                'risk_in_points': risk,  # Store the risk in price points
                'commission': commission,
                'exit_price': None,
                'exit_time': None,
                'pnl': None,
                'pnl_pct': None,
                'pnl_in_r': None,  # PnL in risk units
                'exit_reason': None
            }
            
            # Simulate trading process (starting from the third candle, after entry)
            day_data_after_entry = day_data.iloc[2:]
            
            # Initialize exit flag
            exit_executed = False
            
            for idx, bar in day_data_after_entry.iterrows():
                # Check if stop loss is hit
                if (direction == 1 and bar['Low'] <= stop_loss_price) or \
                   (direction == -1 and bar['High'] >= stop_loss_price):
                    trade['exit_price'] = stop_loss_price
                    trade['exit_time'] = idx
                    trade['exit_reason'] = 'Stop Loss'
                    exit_executed = True
                    break
                
                # Check if take profit is hit
                if (direction == 1 and bar['High'] >= take_profit_price) or \
                   (direction == -1 and bar['Low'] <= take_profit_price):
                    trade['exit_price'] = take_profit_price
                    trade['exit_time'] = idx
                    trade['exit_reason'] = 'Take Profit'
                    exit_executed = True
                    break
            
            # If neither stop loss nor take profit is hit, close at the end of day
            if not exit_executed:
                trade['exit_price'] = day_data.iloc[-1]['Close']
                trade['exit_time'] = day_data.iloc[-1].name
                trade['exit_reason'] = 'End of Day'
            
            # Calculate trade P&L
            trade_pnl = (trade['exit_price'] - trade['entry_price']) * direction * shares - commission
            trade['pnl'] = trade_pnl
            trade['pnl_pct'] = trade_pnl / self.current_capital
            
            # Calculate PnL in risk units (R)
            if risk > 0:
                trade['pnl_in_r'] = ((trade['exit_price'] - trade['entry_price']) * direction) / risk
            else:
                trade['pnl_in_r'] = 0
            
            # Update account capital
            self.current_capital += trade_pnl
            
            # Add trade record
            self.trades.append(trade)
            days_with_trades += 1
        
        print(f"Processing complete: {days_with_trades} days with trade signals, {days_skipped_cross} days skipped due to doji, {days_skipped_data} days skipped due to insufficient data")
        
        # Calculate backtest results
        self._calculate_backtest_results()
        
        return self.backtest_results
    
    def _calculate_backtest_results(self):
        """Calculate backtest results statistics"""
        if not self.trades:
            self.backtest_results = {
                'total_trades': 0,
                'profitable_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_pnl_pct': 0,
                'avg_pnl_pct': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'initial_capital': self.initial_capital,
                'final_capital': self.current_capital
            }
            return
        
        # Convert trade records to DataFrame for analysis
        trades_df = pd.DataFrame(self.trades)
        
        # Calculate basic statistics
        total_trades = len(trades_df)
        profitable_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(trade['pnl'] for trade in self.trades)
        total_pnl_pct = (self.current_capital - self.initial_capital) / self.initial_capital
        
        avg_pnl_pct = trades_df['pnl_pct'].mean() if total_trades > 0 else 0
        
        # Calculate PnL in risk units statistics
        avg_pnl_in_r = trades_df['pnl_in_r'].mean() if total_trades > 0 else 0
        
        # Create complete date range, including non-trading days
        all_dates = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
        daily_capital = pd.Series(index=all_dates, dtype=float)
        daily_capital.iloc[0] = self.initial_capital
        
        # Sort trades by date for accurate equity curve
        sorted_trades = sorted(self.trades, key=lambda x: pd.to_datetime(x['date']))
        
        # Initialize current capital
        current_capital = self.initial_capital
        current_date_idx = 0
        
        for trade in sorted_trades:
            trade_date = pd.to_datetime(trade['date'])
            # Fill dates from last trading day to current trade date
            while current_date_idx < len(all_dates) and all_dates[current_date_idx] < trade_date:
                daily_capital[all_dates[current_date_idx]] = current_capital
                current_date_idx += 1
            
            # Update current day capital
            current_capital += trade['pnl']
            if current_date_idx < len(all_dates) and all_dates[current_date_idx] == trade_date:
                daily_capital[all_dates[current_date_idx]] = current_capital
        
        # Fill remaining dates
        while current_date_idx < len(all_dates):
            daily_capital[all_dates[current_date_idx]] = current_capital
            current_date_idx += 1
        
        # Create complete equity curve DataFrame
        equity_curve = pd.DataFrame({
            'capital': daily_capital
        })
        equity_curve = equity_curve.fillna(method='ffill')  # Fill missing values
        
        # Calculate drawdown
        equity_curve['peak'] = equity_curve['capital'].cummax()
        equity_curve['drawdown'] = (equity_curve['peak'] - equity_curve['capital']) / equity_curve['peak']
        max_drawdown = equity_curve['drawdown'].max()
        
        # Calculate annualized return
        days_in_market = (pd.to_datetime(self.end_date) - pd.to_datetime(self.start_date)).days
        annual_return = (self.current_capital / self.initial_capital) ** (365 / days_in_market) - 1
        
        # Simplified Sharpe ratio calculation (assuming risk-free rate of 0)
        if len(trades_df) > 1:
            returns = trades_df['pnl_pct'].values
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 / (days_in_market / len(returns)))
        else:
            sharpe_ratio = 0
        
        # Store backtest results
        self.backtest_results = {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'avg_pnl_pct': avg_pnl_pct,
            'avg_pnl_in_r': avg_pnl_in_r,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'annual_return': annual_return,
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'daily_returns': self._calculate_daily_returns(),
            'equity_curve': equity_curve
        }
    
    def _calculate_daily_returns(self):
        """Calculate daily returns for subsequent analysis"""
        if not self.trades:
            return pd.Series()
        
        # Create dictionary of date and daily P&L
        daily_pnl = {}
        daily_pnl_in_r = {}
        
        for trade in self.trades:
            date = trade['date']
            if date in daily_pnl:
                daily_pnl[date] += trade['pnl']
                daily_pnl_in_r[date] = daily_pnl_in_r.get(date, 0) + trade['pnl_in_r']
            else:
                daily_pnl[date] = trade['pnl']
                daily_pnl_in_r[date] = trade['pnl_in_r']
        
        # Convert to Series
        daily_pnl_series = pd.Series(daily_pnl)
        daily_pnl_in_r_series = pd.Series(daily_pnl_in_r)
        
        # Calculate daily returns
        initial_capital = self.initial_capital
        daily_returns = {}
        
        for date, pnl in daily_pnl.items():
            daily_returns[date] = pnl / initial_capital
            initial_capital += pnl
        
        # Create a DataFrame with all metrics
        daily_metrics = pd.DataFrame({
            'pnl': daily_pnl_series,
            'pnl_in_r': daily_pnl_in_r_series,
            'returns': pd.Series(daily_returns)
        })
        
        return daily_metrics
    
    def plot_pnl_in_risk_units(self, save_path=None):
        """Plot daily PnL expressed in risk units (R) as per the paper's Figure 3"""
        if not self.trades:
            print("No trade records, unable to plot chart")
            return
        
        # Extract PnL in risk units for each trade
        trade_pnl_in_r = [trade['pnl_in_r'] for trade in self.trades]
        
        # Calculate statistics
        avg_pnl_in_r = np.mean(trade_pnl_in_r)
        profitable_trades = [r for r in trade_pnl_in_r if r > 0]
        losing_trades = [r for r in trade_pnl_in_r if r <= 0]
        avg_profit = np.mean(profitable_trades) if profitable_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        max_profit = max(trade_pnl_in_r) if trade_pnl_in_r else 0
        max_loss = min(trade_pnl_in_r) if trade_pnl_in_r else 0
        
        # Create the figure
        fig, ax = plt.figure(figsize=(14, 8)), plt.gca()
        
        # Plot PnL in R units
        x = range(len(trade_pnl_in_r))
        ax.bar(x, trade_pnl_in_r, width=1, color='skyblue', edgecolor='blue', alpha=0.7)
        
        # Add horizontal lines for averages
        ax.axhline(y=avg_pnl_in_r, color='black', linestyle='-', alpha=0.7, label=f'Average PnL={avg_pnl_in_r:.2f}')
        ax.axhline(y=avg_profit, color='green', linestyle='-', alpha=0.7, label='Average Gain')
        ax.axhline(y=avg_loss, color='red', linestyle='-', alpha=0.7, label='Average Loss')
        
        # Add annotations
        ax.annotate(f'Max Profit', xy=(np.argmax(trade_pnl_in_r), max_profit), 
                   xytext=(np.argmax(trade_pnl_in_r)-100, max_profit+1),
                   arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5),
                   color='blue', fontsize=10)
        
        ax.annotate(f'Max Loss', xy=(np.argmin(trade_pnl_in_r), max_loss), 
                   xytext=(np.argmin(trade_pnl_in_r)+50, max_loss-1),
                   arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5),
                   color='blue', fontsize=10)
        
        ax.annotate(f'Average PnL={avg_pnl_in_r:.2f}', xy=(len(trade_pnl_in_r)-150, avg_pnl_in_r), 
                   xytext=(len(trade_pnl_in_r)-150, avg_pnl_in_r+0.3),
                   color='blue', fontsize=10)
        
        # Add text about losses
        ax.annotate('Some losses are only a fraction of 1R', 
                   xy=(len(trade_pnl_in_r)/2, -1.5),
                   color='blue', fontsize=12)
        
        # Set title and labels
        ax.set_title('History of PnL (in R)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Trade #', fontsize=12)
        ax.set_ylabel('R', fontsize=12)
        
        # Set limits for better visualization
        ax.set_ylim(-2, 11)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"PnL in risk units chart saved to {save_path}")
        
        plt.show()
    
    def plot_equity_curve_comparison(self, save_path=None):
        """Plot equity curve comparison as per the paper's Figure 1"""
        if not self.backtest_results:
            print("No backtest results, please run backtest first")
            return
        
        # Get QQQ daily closing prices for the same period
        daily_data = self.data.groupby('Date')['Close'].last().to_frame()
        daily_data.index = pd.to_datetime(daily_data.index)
        
        if daily_data.empty:
            print("Cannot get QQQ historical data")
            return
        
        # Calculate buy and hold equity curve - modified to use date index matching
        start_price = daily_data['Close'].iloc[0]
        buy_hold_shares = self.initial_capital / start_price
        daily_data['BuyHold'] = daily_data['Close'] * buy_hold_shares
        
        # Get ORB strategy equity curve - modified to complete date range
        equity_curve = self.backtest_results['equity_curve']
        
        # Create figure with log scale
        fig, ax = plt.figure(figsize=(14, 8)), plt.gca()
        
        # Plot equity curves
        ax.plot(equity_curve.index, equity_curve['capital'], 'k-', linewidth=1.5, label='ORB QQQ')
        ax.plot(daily_data.index, daily_data['BuyHold'], 'r--', linewidth=1.5, label='Buy&Hold QQQ')
        
        # Set logarithmic y-scale
        ax.set_yscale('log')
        
        # Format y-axis as dollars
        from matplotlib.ticker import FuncFormatter
        def dollar_formatter(x, pos):
            return f'${x:,.0f}'
        ax.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))
        
        # Set x-axis major ticks to years
        from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter('Jan %y'))
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(loc='upper left', frameon=True)
        
        # Highlight 2020 COVID period with a light gray rectangle
        import matplotlib.patches as patches
        covid_start = pd.to_datetime('2020-02-01')
        covid_end = pd.to_datetime('2020-08-01')
        ax.add_patch(patches.Rectangle((covid_start, ax.get_ylim()[0]), 
                                       covid_end - covid_start, 
                                       ax.get_ylim()[1] - ax.get_ylim()[0],
                                       alpha=0.2, facecolor='gray', edgecolor='none'))
        
        # Highlight 2022 bear market with a light gray rectangle
        bear_start = pd.to_datetime('2022-01-01')
        bear_end = pd.to_datetime('2023-01-01')
        ax.add_patch(patches.Rectangle((bear_start, ax.get_ylim()[0]), 
                                      bear_end - bear_start, 
                                      ax.get_ylim()[1] - ax.get_ylim()[0],
                                      alpha=0.2, facecolor='gray', edgecolor='none'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Equity curve comparison saved to {save_path}")
        
        plt.show()
    
    def compare_with_buy_and_hold(self):
        """Compare ORB strategy with buy and hold QQQ"""
        # Get QQQ daily closing prices for the same period
        daily_data = self.data.groupby('Date')['Close'].last().to_frame()
        daily_data.index = pd.to_datetime(daily_data.index)
        
        if daily_data.empty:
            print("Cannot get QQQ historical data")
            return None
        
        # Calculate buy and hold strategy returns
        start_price = daily_data['Close'].iloc[0]
        end_price = daily_data['Close'].iloc[-1]
        
        buy_hold_shares = self.initial_capital / start_price
        buy_hold_final = buy_hold_shares * end_price
        buy_hold_return = (buy_hold_final - self.initial_capital) / self.initial_capital
        
        # Calculate buy and hold strategy annualized return
        days_in_market = (daily_data.index[-1] - daily_data.index[0]).days
        buy_hold_annual_return = (buy_hold_final / self.initial_capital) ** (365 / days_in_market) - 1
        
        # Calculate buy and hold strategy maximum drawdown
        daily_data['Portfolio'] = daily_data['Close'] * buy_hold_shares
        daily_data['Cummax'] = daily_data['Portfolio'].cummax()
        daily_data['Drawdown'] = (daily_data['Cummax'] - daily_data['Portfolio']) / daily_data['Cummax']
        buy_hold_max_drawdown = daily_data['Drawdown'].max()
        
        # Calculate buy and hold strategy Sharpe ratio
        daily_data['Daily_Return'] = daily_data['Close'].pct_change()
        buy_hold_sharpe = daily_data['Daily_Return'].mean() / daily_data['Daily_Return'].std() * np.sqrt(252) if daily_data['Daily_Return'].std() > 0 else 0
        
        # Build comparison data
        comparison = {
            'orb': {
                'final_capital': self.backtest_results['final_capital'],
                'total_return': self.backtest_results['total_pnl_pct'],
                'annual_return': self.backtest_results['annual_return'],
                'max_drawdown': self.backtest_results['max_drawdown'],
                'sharpe_ratio': self.backtest_results['sharpe_ratio']
            },
            'buy_hold': {
                'final_capital': buy_hold_final,
                'total_return': buy_hold_return,
                'annual_return': buy_hold_annual_return,
                'max_drawdown': buy_hold_max_drawdown,
                'sharpe_ratio': buy_hold_sharpe
            }
        }
        
        return comparison
    
    def generate_report(self, include_comparison=True, save_path=None):
        """Generate backtest report"""
        if not self.backtest_results:
            print("No backtest results, please run backtest first")
            return
        
        report = f"""
=============================================
          QQQ ORB Strategy Backtest Report
=============================================

Backtest Period: {self.start_date} to {self.end_date}
Initial Capital: ${self.initial_capital:.2f}

---------------------------------------------
                Backtest Results
---------------------------------------------
Total Trades: {self.backtest_results['total_trades']}
Profitable Trades: {self.backtest_results['profitable_trades']}
Win Rate: {self.backtest_results['win_rate']:.2%}

Total P&L: ${self.backtest_results['total_pnl']:.2f}
Total Return: {self.backtest_results['total_pnl_pct']:.2%}
Annualized Return: {self.backtest_results['annual_return']:.2%}
Average Return per Trade: {self.backtest_results['avg_pnl_pct']:.2%}
Average PnL in Risk Units (R): {self.backtest_results['avg_pnl_in_r']:.2f}

Maximum Drawdown: {self.backtest_results['max_drawdown']:.2%}
Sharpe Ratio: {self.backtest_results['sharpe_ratio']:.2f}

Final Capital: ${self.backtest_results['final_capital']:.2f}
"""
        
        # Add comparison with buy and hold
        if include_comparison:
            comparison = self.compare_with_buy_and_hold()
            
            if comparison:
                report += f"""
---------------------------------------------
          Comparison with Buy & Hold QQQ
---------------------------------------------
                ORB Strategy    Buy & Hold QQQ    Difference
Final Capital:   ${comparison['orb']['final_capital']:.2f}    ${comparison['buy_hold']['final_capital']:.2f}    ${comparison['orb']['final_capital'] - comparison['buy_hold']['final_capital']:.2f}
Total Return:    {comparison['orb']['total_return']:.2%}    {comparison['buy_hold']['total_return']:.2%}    {comparison['orb']['total_return'] - comparison['buy_hold']['total_return']:.2%}
Annual Return:   {comparison['orb']['annual_return']:.2%}    {comparison['buy_hold']['annual_return']:.2%}    {comparison['orb']['annual_return'] - comparison['buy_hold']['annual_return']:.2%}
Max Drawdown:    {comparison['orb']['max_drawdown']:.2%}    {comparison['buy_hold']['max_drawdown']:.2%}    {comparison['orb']['max_drawdown'] - comparison['buy_hold']['max_drawdown']:.2%}
Sharpe Ratio:    {comparison['orb']['sharpe_ratio']:.2f}    {comparison['buy_hold']['sharpe_ratio']:.2f}    {comparison['orb']['sharpe_ratio'] - comparison['buy_hold']['sharpe_ratio']:.2f}
"""
        
        # Add trade details
        report += """
---------------------------------------------
                Trade Statistics
---------------------------------------------
"""
        
        # Calculate PnL in R statistics
        pnl_in_r = [trade['pnl_in_r'] for trade in self.trades]
        profitable_r = [r for r in pnl_in_r if r > 0]
        losing_r = [r for r in pnl_in_r if r <= 0]
        
        report += f"""
Total Trades in R-multiples: {len(pnl_in_r)}
Avg PnL in R: {np.mean(pnl_in_r):.2f}
Avg Profit in R: {np.mean(profitable_r) if profitable_r else 0:.2f}
Avg Loss in R: {np.mean(losing_r) if losing_r else 0:.2f}
Max Profit in R: {max(pnl_in_r) if pnl_in_r else 0:.2f}
Max Loss in R: {min(pnl_in_r) if pnl_in_r else 0:.2f}
"""
        
        report += "\n=============================================\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {save_path}")
        
        return report


def main():
    """Main function"""
    # Set backtest parameters to match the paper
    start_date = '2016-01-01'
    end_date = '2023-02-17'
    initial_capital = 25000
    
    # Create output directory
    output_dir = 'orb_backtest_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create and run QQQ ORB strategy backtest
    orb_qqq = ORB_QQQ_Strategy(start_date, end_date, initial_capital)
    results = orb_qqq.run_backtest()
    
    # Generate report
    report = orb_qqq.generate_report(include_comparison=True, save_path=f"{output_dir}/QQQ_ORB_report.txt")
    print(report)
    
    # Plot equity curve comparison (Figure 1 from the paper)
    orb_qqq.plot_equity_curve_comparison(save_path=f"{output_dir}/Figure1_QQQ_ORB_vs_BuyHold.png")
    
    # Plot PnL in risk units (Figure 3 from the paper)
    orb_qqq.plot_pnl_in_risk_units(save_path=f"{output_dir}/Figure3_QQQ_ORB_PnL_in_R.png")
    
    print(f"\nBacktest complete! All results saved to {output_dir} directory")


if __name__ == "__main__":
    main() 