from ibkr_5min_data_fetcher import IBKR5MinDataFetcher
import config
import os
import pandas as pd

def main():
    # 创建数据目录
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)
    
    # 使用上下文管理器确保正确断开连接
    try:
        with IBKR5MinDataFetcher() as fetcher:
            # 获取QQQ的数据
            data = fetcher.get_orb_data(
                symbol='QQQ',
                start_date='2024-01-01',
                end_date='2024-02-28',
                use_rth=True
            )
            
            if not data.empty:
                # 构建输出文件路径
                output_file = os.path.join(config.DATA_DIR, 'qqq_orb_data.csv')
                # 保存数据
                data.to_csv(output_file, index=False)
                print(f"数据已保存到 {output_file}")
                print(f"获取到 {len(data)} 条数据")
            else:
                print("未获取到数据")
    except Exception as e:
        print(f"运行过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 