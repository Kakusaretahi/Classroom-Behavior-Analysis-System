import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# =========================
# 1. 读取时间序列数据
# =========================
data = pd.read_csv("collective_timeseries.csv")
ratios = data["ratio"].values.reshape(-1, 1)

scaler = MinMaxScaler()
ratios_scaled = scaler.fit_transform(ratios)

# =========================
# 2. 构造序列数据
# =========================
SEQ_LEN = 20  # 用过去20个时间点预测下一个

X = []
y = []

for i in range(len(ratios_scaled) - SEQ_LEN):
    X.append(ratios_scaled[i:i+SEQ_LEN])
    y.append(ratios_scaled[i+SEQ_LEN])

X = np.array(X)
y = np.array(y)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# =========================
# 3. 定义LSTM模型
# =========================
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# =========================
# 4. 训练模型
# =========================
EPOCHS = 100

for epoch in range(EPOCHS):
    model.train()
    output = model(X)
    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# =========================
# 5. 预测
# =========================
model.eval()
predicted = model(X).detach().numpy()

# 反归一化
predicted = scaler.inverse_transform(predicted)
real = scaler.inverse_transform(y.numpy())

# =========================
# 6. 可视化
# =========================
plt.figure()
plt.plot(real, label="Real")
plt.plot(predicted, label="Predicted")
plt.legend()
plt.title("LSTM Abnormal Ratio Prediction")
plt.show()