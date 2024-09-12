import torch
import torch.nn as nn
import pandas as pd
# 加载已保存的模型参数
loaded_model = NeuralNetwork(input_dim)  # 创建一个新的模型实例
loaded_model.load_state_dict(torch.load('hyperparameter-model.pth'))  # 加载参数字典
loaded_model.eval()  # 设置模型为评估模式

csv_file_path='mse_data_1.csv'
date=pd.read_csv(csv_file_path)
print(date)
