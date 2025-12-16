import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建可视化文件夹
output_dir = 'data_visualization'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取数据
data = pd.read_csv('nflx_2014_2023.csv')

# 数据处理
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# 1. 价格走势可视化
plt.figure(figsize=(12, 6))
plt.plot(data['close'], label='收盘价')
plt.plot(data['next_day_close'], label='次日收盘价')
plt.title('Netflix股价走势 (2014-2023)')
plt.xlabel('日期')
plt.ylabel('价格 ($)')
plt.legend()
plt.grid(True)
plt.savefig(f'{output_dir}/price_trend.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. 成交量可视化
plt.figure(figsize=(12, 6))
plt.bar(data.index, data['volume'], color='skyblue')
plt.title('Netflix成交量走势 (2014-2023)')
plt.xlabel('日期')
plt.ylabel('成交量')
plt.grid(True)
plt.savefig(f'{output_dir}/volume_trend.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. 技术指标可视化 - RSI
plt.figure(figsize=(12, 6))
plt.plot(data['rsi_7'], label='RSI 7')
plt.plot(data['rsi_14'], label='RSI 14')
plt.axhline(y=70, color='r', linestyle='--', label='超买线')
plt.axhline(y=30, color='g', linestyle='--', label='超卖线')
plt.title('Netflix RSI指标 (2014-2023)')
plt.xlabel('日期')
plt.ylabel('RSI值')
plt.legend()
plt.grid(True)
plt.savefig(f'{output_dir}/rsi_indicators.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. 技术指标可视化 - CCI
plt.figure(figsize=(12, 6))
plt.plot(data['cci_7'], label='CCI 7')
plt.plot(data['cci_14'], label='CCI 14')
plt.axhline(y=100, color='r', linestyle='--', label='超买线')
plt.axhline(y=-100, color='g', linestyle='--', label='超卖线')
plt.title('Netflix CCI指标 (2014-2023)')
plt.xlabel('日期')
plt.ylabel('CCI值')
plt.legend()
plt.grid(True)
plt.savefig(f'{output_dir}/cci_indicators.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. 相关性热力图
correlation_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi_7', 'rsi_14', 'cci_7', 'cci_14', 'next_day_close']
correlation_matrix = data[correlation_cols].corr()

plt.figure(figsize=(12, 10))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.title('Netflix股票数据相关性热力图')
plt.xticks(np.arange(len(correlation_cols)), correlation_cols, rotation=45)
plt.yticks(np.arange(len(correlation_cols)), correlation_cols)
plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print('数据处理完成！')
print(f'可视化图表已保存到 {output_dir} 文件夹')