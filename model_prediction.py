import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建可视化文件夹
output_dir = 'prediction_visualization'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取数据
data = pd.read_csv('nflx_2014_2023.csv')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# 选择特征和目标
features = ['open', 'high', 'low', 'close', 'volume', 'rsi_7', 'rsi_14', 'cci_7', 'cci_14']
target = 'next_day_close'

# 准备数据
train_data = data[features + [target]]

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(train_data)

# 创建序列数据
sequence_length = 60

def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, :-1])
        y.append(data[i, -1])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, sequence_length)

# 转换为PyTorch张量
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 模型参数
input_size = len(features)
hidden_size = 50
num_layers = 2
output_size = 1

# 初始化模型
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# 加载训练好的模型
model.load_state_dict(torch.load('saved_model/lstm_model.pth'))
model.eval()
print('模型加载完成！')

# 进行预测
with torch.no_grad():
    y_pred = model(X)

# 将PyTorch张量转换为NumPy数组
y_pred = y_pred.numpy()
y = y.numpy()

# 反归一化预测结果和实际值
# 创建一个全零数组，大小与原始缩放数据相同
scaled_pred = np.zeros((len(y_pred), scaled_data.shape[1]))
scaled_actual = np.zeros((len(y), scaled_data.shape[1]))

# 将预测结果和实际值放入最后一列（对应target）
scaled_pred[:, -1] = y_pred.flatten()
scaled_actual[:, -1] = y.flatten()

# 反归一化
actual_prices = scaler.inverse_transform(scaled_actual)[:, -1]
predicted_prices = scaler.inverse_transform(scaled_pred)[:, -1]

# 计算评估指标
mae = mean_absolute_error(actual_prices, predicted_prices)
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))

print(f'\n预测评估指标：')
print(f'平均绝对误差 (MAE): {mae:.4f}')
print(f'均方根误差 (RMSE): {rmse:.4f}')

# 可视化预测结果
plt.figure(figsize=(12, 6))
plt.plot(data.index[sequence_length:], actual_prices, label='实际次日收盘价', color='blue')
plt.plot(data.index[sequence_length:], predicted_prices, label='预测次日收盘价', color='red', alpha=0.7)
plt.title('Netflix股票价格预测 (LSTM模型)')
plt.xlabel('日期')
plt.ylabel('价格 ($)')
plt.legend()
plt.grid(True)
plt.savefig(f'{output_dir}/price_prediction.png', dpi=300, bbox_inches='tight')
plt.close()

# 可视化预测与实际值的散点图
plt.figure(figsize=(10, 10))
plt.scatter(actual_prices, predicted_prices, alpha=0.7)
plt.plot([actual_prices.min(), actual_prices.max()], [actual_prices.min(), actual_prices.max()], 'r--', lw=2)
plt.title('预测值 vs 实际值散点图')
plt.xlabel('实际次日收盘价 ($)')
plt.ylabel('预测次日收盘价 ($)')
plt.grid(True)
plt.savefig(f'{output_dir}/prediction_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

# 可视化预测误差分布
errors = actual_prices - predicted_prices
plt.figure(figsize=(12, 6))
plt.hist(errors, bins=50, alpha=0.7, color='green')
plt.title('预测误差分布直方图')
plt.xlabel('误差 ($)')
plt.ylabel('频数')
plt.grid(True)
plt.savefig(f'{output_dir}/prediction_error.png', dpi=300, bbox_inches='tight')
plt.close()

# 保存预测结果到CSV文件
prediction_df = pd.DataFrame({
    'date': data.index[sequence_length:],
    'actual': actual_prices,
    'predicted': predicted_prices,
    'error': errors
})
prediction_df.to_csv(f'{output_dir}/predictions.csv', index=False)
print(f'\n预测结果已保存到 {output_dir}/predictions.csv')
print(f'预测可视化已保存到 {output_dir} 文件夹')
print('\n预测完成！')