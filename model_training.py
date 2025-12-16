import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建可视化文件夹
output_dir = 'model_training_visualization'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 创建模型保存文件夹
model_dir = 'saved_model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 读取数据
data = pd.read_csv('nflx_2014_2023.csv')

# 数据处理
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# 选择特征和目标
features = ['open', 'high', 'low', 'close', 'volume', 'rsi_7', 'rsi_14', 'cci_7', 'cci_14']
target = 'next_day_close'

# 准备训练数据
train_data = data[features + [target]]

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(train_data)

# 创建序列数据
def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, :-1])
        y.append(data[i, -1])
    return np.array(X), np.array(y)

# 设置序列长度
sequence_length = 60

# 创建训练序列
X, y = create_sequences(scaled_data, sequence_length)

# 转换为PyTorch张量
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# 划分训练集和验证集
train_size = int(0.8 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

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

# 初始化模型、损失函数和优化器
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练参数
epochs = 100
early_stop_patience = 5  # 增加耐心值，让模型有更多学习机会
best_val_loss = float('inf')
patience_counter = 0
history = {'train_loss': [], 'val_loss': []}

# 训练模型
for epoch in range(epochs):
    # 训练模式
    model.train()
    train_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_X.size(0)
    
    train_loss /= len(train_loader.dataset)
    history['train_loss'].append(train_loss)
    
    # 验证模式
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_X.size(0)
    
    val_loss /= len(val_loader.dataset)
    history['val_loss'].append(val_loss)
    
    # 打印训练信息
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    # 早停检查 - 添加损失改进阈值，只有当改善超过阈值时才重置计数器
    loss_improvement = best_val_loss - val_loss
    improvement_threshold = 1e-4  # 降低阈值，使早停机制更敏感
    
    if val_loss < best_val_loss and loss_improvement > improvement_threshold:
        best_val_loss = val_loss
        patience_counter = 0
        # 保存最佳模型
        torch.save(model.state_dict(), f'{model_dir}/lstm_model.pth')
        print(f'验证损失改善 {loss_improvement:.6f}，保存最佳模型')
    else:
        patience_counter += 1
        print(f'验证损失未显著改善，耐心值: {patience_counter}/{early_stop_patience}')
        if patience_counter >= early_stop_patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

# 加载最佳模型
model.load_state_dict(torch.load(f'{model_dir}/lstm_model.pth'))
print(f'模型已保存到 {model_dir}/lstm_model.pth')

# 可视化训练过程
plt.figure(figsize=(12, 6))
plt.plot(history['train_loss'], label='训练损失')
plt.plot(history['val_loss'], label='验证损失')
plt.title('LSTM模型训练损失曲线')
plt.xlabel('训练轮次')
plt.ylabel('均方误差 (MSE)')
plt.legend()
plt.grid(True)
plt.savefig(f'{output_dir}/training_loss.png', dpi=300, bbox_inches='tight')
plt.close()

# 计算和保存训练过程中的指标
print('\n训练完成！')
print(f'最佳验证损失: {best_val_loss:.6f}')

# 保存训练历史
np.save(f'{model_dir}/training_history.npy', history)
print(f'训练历史已保存到 {model_dir}/training_history.npy')

# 模型结构
print('\n模型结构：')
print(model)

print('\n模型训练可视化完成！')
print(f'训练信息可视化已保存到 {output_dir} 文件夹')