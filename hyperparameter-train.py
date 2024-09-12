import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn import preprocessing
# 加载CSV文件
data = pd.read_csv('mse_data_1.csv')

# 提取特征列和目标列
X = data[['num_leaves', 'max_bin', 'max_depth', 'learning_rate', 'colsample_bytree', 'bagging_fraction', 'min_child_samples']]
y = data['mse']  # 假设MSE是目标列的名称
#归一化
min_max_scaler = preprocessing.MinMaxScaler()
X=min_max_scaler.fit_transform(X)

# 划分数据集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("==========X_train=========")
print(X_train,X_train.shape)
print("==========X_val===========")
print(X_val)
print("==========y_train=========")
print(y_train)
print("==========y_val===========")
print(y_val)

class MyNet(torch.nn.Module):
    def __init__(self,in_put,hidden,hidden1,out_put):
        super().__init__()
        self.linear1=torch.nn.Linear(in_put,hidden)
        self.linear2=torch.nn.Linear(hidden,hidden1)
        self.linear3=torch.nn.Linear(hidden1,out_put)

    def forward(self,data):
        x=self.linear1(data)
        x=torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear3(x)
        return x
in_features=X_train.shape[1]
print(in_features)
hidden=256
hidden1=128
out_put=1
model=MyNet(in_features,hidden,hidden1,out_put)

criterion=torch.nn.MSELoss()

learn_rate=0.0001
optimizer=torch.optim.Adam(model.parameters(),learn_rate)

print(model)

# 将数据转换为PyTorch张量
#X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)
#y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).reshape(-1, 1)

epochs=2000

for epoch in range(epochs):
    model.train()

    outputs=model(X_train_tensor)
    loss=criterion (outputs,y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.10f}')

