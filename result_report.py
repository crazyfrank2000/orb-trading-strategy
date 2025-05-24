"""
result_report.py - 交易结果分析报告生成器

功能:
1. 分析CSV交易日志文件
2. 与买入持有(B&H)策略对比
3. 统计盈亏分布情况
4. 生成可视化图表和分析报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import glob
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class TradingResultAnalyzer:
    def __init__(self, log_dir='log', data_dir='data'):
        """初始化分析器"""
        self.log_dir = log_dir
        self.data_dir = data_dir
        self.trades_df = None
        self.symbol = None
        self.price_data = None
        self.strategy_returns = None
        self.bh_returns = None
        
        # 创建输出目录
        self.output_dir = 'reports'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"创建报告目录: {self.output_dir}")
    
    def load_trade_log(self, csv_file_path=None):
        """加载交易日志CSV文件"""
        if csv_file_path is None:
            # 自动查找最新的交易日志文件
            csv_files = glob.glob(os.path.join(self.log_dir, "*_trades_*.csv"))
            if not csv_files:
                raise FileNotFoundError("未找到交易日志文件")
            csv_file_path = max(csv_files, key=os.path.getctime)
        
        print(f"加载交易日志: {csv_file_path}")
        
        # 读取CSV文件
        self.trades_df = pd.read_csv(csv_file_path, encoding='utf-8-sig')
        
        # 从文件名提取品种信息
        filename = os.path.basename(csv_file_path)
        self.symbol = filename.split('_')[0]
        
        # 数据清理和类型转换
        self.trades_df['交易日期'] = pd.to_datetime(self.trades_df['交易日期'])
        self.trades_df['盈亏'] = pd.to_numeric(self.trades_df['盈亏'], errors='coerce')
        self.trades_df['R倍数'] = pd.to_numeric(self.trades_df['R倍数'], errors='coerce')
        self.trades_df['交易后资金'] = pd.to_numeric(self.trades_df['交易后资金'], errors='coerce')
        
        print(f"成功加载 {len(self.trades_df)} 条交易记录")
        return self.trades_df
    
    def load_price_data(self):
        """加载价格数据用于计算B&H基准"""
        # 查找对应的5分钟数据文件
        price_files = glob.glob(os.path.join(self.data_dir, f"{self.symbol}_5min_full_*.csv"))
        
        if not price_files:
            print(f"警告: 未找到 {self.symbol} 的价格数据文件")
            return None
        
        price_file = price_files[0]
        print(f"加载价格数据: {price_file}")
        
        try:
            self.price_data = pd.read_csv(price_file)
            
            # 处理日期列，设置utc=True避免混合时区警告
            self.price_data['date'] = pd.to_datetime(self.price_data['date'], utc=True)
            
            # 生成日线数据
            daily_prices = self.price_data.groupby(self.price_data['date'].dt.date).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).reset_index()
            
            daily_prices['date'] = pd.to_datetime(daily_prices['date'])
            self.price_data = daily_prices.sort_values('date')
            
            return self.price_data
            
        except Exception as e:
            print(f"加载价格数据时出错: {str(e)}")
            print("跳过B&H对比分析")
            return None
    
    def calculate_strategy_returns(self):
        """计算策略收益率"""
        if self.trades_df is None:
            raise ValueError("请先加载交易日志")
        
        # 按日期排序
        trades_sorted = self.trades_df.sort_values('交易日期')
        
        # 计算累计收益率
        initial_capital = trades_sorted['交易前资金'].iloc[0]
        trades_sorted['累计收益率'] = (trades_sorted['交易后资金'] / initial_capital - 1) * 100
        
        # 创建日期序列的收益率
        date_range = pd.date_range(
            start=trades_sorted['交易日期'].min(),
            end=trades_sorted['交易日期'].max(),
            freq='D'
        )
        
        strategy_returns = pd.DataFrame({'date': date_range})
        strategy_returns['return'] = 0.0
        
        for _, trade in trades_sorted.iterrows():
            mask = strategy_returns['date'] >= trade['交易日期']
            strategy_returns.loc[mask, 'return'] = trade['累计收益率']
        
        self.strategy_returns = strategy_returns
        return strategy_returns
    
    def calculate_bh_returns(self):
        """计算买入持有基准收益率"""
        if self.price_data is None:
            print("警告: 无价格数据，跳过B&H计算")
            return None
        
        if self.trades_df is None:
            raise ValueError("请先加载交易日志")
        
        # 获取策略开始和结束日期
        start_date = self.trades_df['交易日期'].min()
        end_date = self.trades_df['交易日期'].max()
        
        # 过滤价格数据到策略期间
        price_period = self.price_data[
            (self.price_data['date'] >= start_date) & 
            (self.price_data['date'] <= end_date)
        ].copy()
        
        if price_period.empty:
            print("警告: 策略期间无价格数据")
            return None
        
        # 计算B&H收益率
        initial_price = price_period['close'].iloc[0]
        price_period['bh_return'] = (price_period['close'] / initial_price - 1) * 100
        
        self.bh_returns = price_period[['date', 'bh_return']].copy()
        return self.bh_returns
    
    def analyze_win_loss(self):
        """分析盈亏情况"""
        if self.trades_df is None:
            raise ValueError("请先加载交易日志")
        
        # 基本统计
        total_trades = len(self.trades_df)
        winning_trades = self.trades_df[self.trades_df['盈亏'] > 0]
        losing_trades = self.trades_df[self.trades_df['盈亏'] < 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = win_count / total_trades * 100 if total_trades > 0 else 0
        
        # 盈亏统计
        total_profit = winning_trades['盈亏'].sum() if not winning_trades.empty else 0
        total_loss = losing_trades['盈亏'].sum() if not losing_trades.empty else 0
        net_profit = total_profit + total_loss
        
        avg_win = winning_trades['盈亏'].mean() if not winning_trades.empty else 0
        avg_loss = losing_trades['盈亏'].mean() if not losing_trades.empty else 0
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        # R倍数分析
        avg_r = self.trades_df['R倍数'].mean()
        win_r = winning_trades['R倍数'].mean() if not winning_trades.empty else 0
        loss_r = losing_trades['R倍数'].mean() if not losing_trades.empty else 0
        
        analysis = {
            'total_trades': total_trades,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_profit': net_profit,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_r': avg_r,
            'avg_win_r': win_r,
            'avg_loss_r': loss_r,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades
        }
        
        return analysis
    
    def plot_equity_curve_comparison(self):
        """绘制权益曲线对比图"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # 上图：收益率对比
        if self.strategy_returns is not None:
            ax1.plot(self.strategy_returns['date'], self.strategy_returns['return'], 
                    label=f'{self.symbol} ORB策略', linewidth=2, color='blue')
        
        if self.bh_returns is not None:
            ax1.plot(self.bh_returns['date'], self.bh_returns['bh_return'], 
                    label=f'{self.symbol} 买入持有', linewidth=2, color='red', alpha=0.7)
        
        ax1.set_title(f'{self.symbol} 策略收益率对比', fontsize=14, fontweight='bold')
        ax1.set_ylabel('累计收益率 (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # 下图：账户资金变化
        if self.trades_df is not None:
            trades_sorted = self.trades_df.sort_values('交易日期')
            ax2.plot(trades_sorted['交易日期'], trades_sorted['交易后资金'], 
                    label='账户资金', linewidth=2, color='green', marker='o', markersize=3)
            ax2.axhline(y=trades_sorted['交易前资金'].iloc[0], color='gray', 
                       linestyle='--', alpha=0.7, label='初始资金')
        
        ax2.set_title('账户资金变化', fontsize=14, fontweight='bold')
        ax2.set_xlabel('日期')
        ax2.set_ylabel('账户资金 ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'{self.symbol}_equity_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"权益曲线对比图已保存: {output_path}")
        plt.show()
    
    def plot_pnl_distribution(self, analysis):
        """绘制盈亏分布图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 盈亏分布直方图
        all_pnl = self.trades_df['盈亏'].dropna()
        ax1.hist(all_pnl, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax1.set_title('盈亏分布直方图', fontsize=12, fontweight='bold')
        ax1.set_xlabel('盈亏金额 ($)')
        ax1.set_ylabel('交易次数')
        ax1.grid(True, alpha=0.3)
        
        # R倍数分布
        r_values = self.trades_df['R倍数'].dropna()
        ax2.hist(r_values, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_title('R倍数分布', fontsize=12, fontweight='bold')
        ax2.set_xlabel('R倍数')
        ax2.set_ylabel('交易次数')
        ax2.grid(True, alpha=0.3)
        
        # 胜负比例饼图
        win_loss_data = [analysis['win_count'], analysis['loss_count']]
        win_loss_labels = [f"盈利 ({analysis['win_count']})", f"亏损 ({analysis['loss_count']})"]
        colors = ['lightgreen', 'lightcoral']
        
        ax3.pie(win_loss_data, labels=win_loss_labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('胜负比例', fontsize=12, fontweight='bold')
        
        # 月度盈亏趋势
        if not self.trades_df.empty:
            monthly_pnl = self.trades_df.groupby(self.trades_df['交易日期'].dt.to_period('M'))['盈亏'].sum()
            monthly_pnl.plot(kind='bar', ax=ax4, color='steelblue', alpha=0.7)
            ax4.set_title('月度盈亏趋势', fontsize=12, fontweight='bold')
            ax4.set_xlabel('月份')
            ax4.set_ylabel('盈亏 ($)')
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'{self.symbol}_pnl_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"盈亏分布图已保存: {output_path}")
        plt.show()
    
    def plot_trade_analysis(self, analysis):
        """绘制交易分析图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 出场原因统计
        exit_reasons = self.trades_df['出场原因'].value_counts()
        exit_reasons.plot(kind='pie', ax=ax1, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'orange'])
        ax1.set_title('出场原因分布', fontsize=12, fontweight='bold')
        ax1.set_ylabel('')
        
        # 盈利交易的R倍数分布
        if not analysis['winning_trades'].empty:
            win_r_values = analysis['winning_trades']['R倍数'].dropna()
            ax2.hist(win_r_values, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
            ax2.set_title('盈利交易R倍数分布', fontsize=12, fontweight='bold')
            ax2.set_xlabel('R倍数')
            ax2.set_ylabel('交易次数')
            ax2.grid(True, alpha=0.3)
        
        # 亏损交易的R倍数分布
        if not analysis['losing_trades'].empty:
            loss_r_values = analysis['losing_trades']['R倍数'].dropna()
            ax3.hist(loss_r_values, bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
            ax3.set_title('亏损交易R倍数分布', fontsize=12, fontweight='bold')
            ax3.set_xlabel('R倍数')
            ax3.set_ylabel('交易次数')
            ax3.grid(True, alpha=0.3)
        
        # 交易方向分析
        direction_count = self.trades_df['方向'].value_counts()
        direction_pnl = self.trades_df.groupby('方向')['盈亏'].sum()
        
        x_pos = range(len(direction_count))
        bars = ax4.bar(x_pos, direction_pnl.values, alpha=0.7, 
                       color=['green' if x > 0 else 'red' for x in direction_pnl.values])
        ax4.set_title('交易方向盈亏分析', fontsize=12, fontweight='bold')
        ax4.set_xlabel('交易方向')
        ax4.set_ylabel('总盈亏 ($)')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(direction_count.index)
        ax4.grid(True, alpha=0.3)
        
        # 在柱子上添加数值标签
        for i, (bar, count) in enumerate(zip(bars, direction_count.values)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + (50 if height > 0 else -100),
                    f'${height:.0f}\n({count}笔)', ha='center', va='bottom' if height > 0 else 'top')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'{self.symbol}_trade_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"交易分析图已保存: {output_path}")
        plt.show()
    
    def generate_summary_report(self, analysis):
        """生成分析报告摘要"""
        # 计算策略与B&H对比
        strategy_return = self.strategy_returns['return'].iloc[-1] if self.strategy_returns is not None else 0
        bh_return = self.bh_returns['bh_return'].iloc[-1] if self.bh_returns is not None else 0
        
        # 计算最大回撤
        if self.strategy_returns is not None:
            cumulative = (1 + self.strategy_returns['return'] / 100).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max * 100
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0
        
        report = f"""
{'='*60}
                    {self.symbol} 交易策略分析报告
{'='*60}

📊 基本统计
----------------------------------------
总交易次数:          {analysis['total_trades']:>10}
盈利次数:            {analysis['win_count']:>10}
亏损次数:            {analysis['loss_count']:>10}
胜率:                {analysis['win_rate']:>9.2f}%

💰 盈亏分析
----------------------------------------
总盈利:              ${analysis['total_profit']:>9.2f}
总亏损:              ${analysis['total_loss']:>9.2f}
净盈亏:              ${analysis['net_profit']:>9.2f}
盈亏比:              {analysis['profit_factor']:>10.2f}

📈 收益率对比
----------------------------------------
策略总收益率:        {strategy_return:>9.2f}%
买入持有收益率:      {bh_return:>9.2f}%
超额收益:            {strategy_return - bh_return:>9.2f}%
最大回撤:            {max_drawdown:>9.2f}%

🎯 风险收益指标
----------------------------------------
平均盈利:            ${analysis['avg_win']:>9.2f}
平均亏损:            ${analysis['avg_loss']:>9.2f}
平均R倍数:           {analysis['avg_r']:>10.2f}
盈利交易平均R:       {analysis['avg_win_r']:>10.2f}
亏损交易平均R:       {analysis['avg_loss_r']:>10.2f}

🔍 出场原因统计
----------------------------------------
"""
        # 添加出场原因统计
        exit_stats = self.trades_df['出场原因'].value_counts()
        for reason, count in exit_stats.items():
            percentage = count / analysis['total_trades'] * 100
            report += f"{reason}:           {count:>5} ({percentage:>5.1f}%)\n"
        
        report += f"\n{'='*60}\n报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        # 保存报告到文件
        report_path = os.path.join(self.output_dir, f'{self.symbol}_analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        print(f"\n详细报告已保存: {report_path}")
        
        return report
    
    def run_full_analysis(self, csv_file_path=None):
        """运行完整分析"""
        print("开始交易结果分析...")
        
        # 加载数据
        self.load_trade_log(csv_file_path)
        self.load_price_data()
        
        # 计算收益率
        self.calculate_strategy_returns()
        self.calculate_bh_returns()
        
        # 分析盈亏
        analysis = self.analyze_win_loss()
        
        # 生成图表
        print("\n生成分析图表...")
        self.plot_equity_curve_comparison()
        self.plot_pnl_distribution(analysis)
        self.plot_trade_analysis(analysis)
        
        # 生成报告
        print("\n生成分析报告...")
        self.generate_summary_report(analysis)
        
        print(f"\n分析完成！所有文件已保存到 {self.output_dir} 目录")


def main():
    """主函数"""
    analyzer = TradingResultAnalyzer()
    
    # 检查是否有交易日志文件
    log_files = glob.glob(os.path.join(analyzer.log_dir, "*_trades_*.csv"))
    
    if not log_files:
        print("错误: 未找到交易日志文件，请先运行策略生成交易记录")
        return
    
    print(f"找到 {len(log_files)} 个交易日志文件:")
    for i, file in enumerate(log_files):
        print(f"{i+1}. {os.path.basename(file)}")
    
    # 选择要分析的文件
    if len(log_files) == 1:
        selected_file = log_files[0]
    else:
        try:
            choice = input(f"\n请选择要分析的文件 (1-{len(log_files)}, 默认最新): ").strip()
            if choice == "":
                selected_file = max(log_files, key=os.path.getctime)
            else:
                selected_file = log_files[int(choice) - 1]
        except (ValueError, IndexError):
            print("无效选择，使用最新文件")
            selected_file = max(log_files, key=os.path.getctime)
    
    # 运行分析
    analyzer.run_full_analysis(selected_file)


if __name__ == "__main__":
    main() 