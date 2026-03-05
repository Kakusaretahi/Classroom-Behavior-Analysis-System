import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("output/collective_timeseries.csv")
ratios = data["ratio"].values.reshape(-1,1)
scaler = MinMaxScaler()
ratios_scaled = scaler.fit_transform(ratios)
SEQ_LEN = 20
X=[]
y=[]

for i in range(len(ratios_scaled)-SEQ_LEN):
    X.append(ratios_scaled[i:i+SEQ_LEN])
    y.append(ratios_scaled[i+SEQ_LEN])

X=np.array(X)
y=np.array(y)

X=torch.tensor(X,dtype=torch.float32)
y=torch.tensor(y,dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm=nn.LSTM(1,32,batch_first=True)
        self.fc=nn.Linear(32,1)

    def forward(self,x):
        out,_=self.lstm(x)
        out=out[:,-1,:]
        out=self.fc(out)
        return out

model=LSTMModel()
criterion=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
for epoch in range(100):
    output=model(X)
    loss=criterion(output,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
model.eval()
pred=model(X).detach().numpy()
pred=scaler.inverse_transform(pred)
real=scaler.inverse_transform(y.numpy())
plt.figure()
plt.plot(real,label="Real")
plt.plot(pred,label="Predicted")
plt.legend()
plt.title("Abnormal Ratio Prediction")
plt.savefig("output/predict.png")