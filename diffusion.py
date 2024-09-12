import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
"""
initial_params = [
    torch.tensor([30, 0.014491855350176669]),
    torch.tensor([40, 0.014140589822092303]),
    torch.tensor([50, 0.014029538870042987]),
    torch.tensor([60, 0.014027916680373783]),
    torch.tensor([70, 0.013975871679439497])
]
"""
initial_params = [
    torch.tensor([30, 0.014491855350176669,10]),
    torch.tensor([40, 0.015140589822092303,20]),
    torch.tensor([50, 0.016029538870042987,30]),
    torch.tensor([60, 0.017027916680373783,40]),
    torch.tensor([70, 0.018975871679439497,50])
]

num_steps = 10000

betas = torch.linspace(-6, 6, num_steps)
betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, 0)
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

assert alphas.shape == alphas_prod.shape == alphas_prod_p.shape == alphas_bar_sqrt.shape == one_minus_alphas_bar_sqrt.shape == one_minus_alphas_bar_log.shape


#3 确定扩散过程任意时刻的采样值
"""
def q_x(x_0, t):
    noise = torch.randn_like(x_0[0])
    alphas_t = alphas_bar_sqrt[t]
    alphas_l_m_t = one_minus_alphas_bar_sqrt[t]
    return alphas_t * x_0 + alphas_l_m_t * noise
"""

"""
def q_x(x_0, t):
    alphas_t = alphas_bar_sqrt[t]
    alphas_l_m_t = one_minus_alphas_bar_sqrt[t]
    noisy_x_list = []
    for x in x_0:
        noise = torch.randn_like(x)
        noisy_x = alphas_t * x + alphas_l_m_t * noise
        noisy_x_list.append(noisy_x)
    return noisy_x_list
#4 加噪步之后的效果
"""
def q_x(x_0, t):
    alphas_t = alphas_bar_sqrt[t]
    alphas_l_m_t = one_minus_alphas_bar_sqrt[t]
    noisy_x_list = []
    for x in x_0:
        noise = torch.randn_like(x, dtype=torch.float)  # 明确指定生成的随机噪声的数据类型为浮点数类型
        noisy_x = alphas_t * x + alphas_l_m_t * noise * torch.cos(betas[t])
        noisy_x_list.append(noisy_x)
    return noisy_x_list
expand_params=[]
for t in range(num_steps):
    q_t = q_x(initial_params, torch.tensor(t))
    expand_params.append(q_t)
    #print("Step", t + 1, ":")
    #for i, params in enumerate(q_t):
        #print(f"  Parameter {i + 1}: {params}")


expand_params = [item for sublist in expand_params for item in sublist]

sorted_expand_params=sorted(expand_params,key=lambda x:x[0])
print(sorted_expand_params)

# 提取 X 轴和 Y 轴的值
x_values = [item[0].item() for item in sorted_expand_params]
y_values = [item[1].item() for item in sorted_expand_params]
z_values = [item[2].item() for item in sorted_expand_params]



# 绘制三维图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_values, y_values, z_values, marker='o')

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('Three-dimensional Scatter Plot of Sorted Expand Params')

plt.savefig('C:\\Users\\10553\\Desktop\\1\\三维\\10000steps')  # 保存图片
plt.show()
