import argparse
import logging
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.io
import pickle
torch.autograd.set_detect_anomaly(True)
os.environ['KMP_DUPLICATE_LIB_OK']='True'
with open('./US101_Lane1to5_t1.5s30.pickle', 'rb') as f:
    data_pickle = pickle.load(f)

rhoMat = np.array([np.array(ele) for ele in data_pickle['rhoMat']])
uMat = np.array([np.array(ele) for ele in data_pickle['vMat']])
Exact_rho = rhoMat.T
Exact_u = uMat.T
rho_star = Exact_rho.flatten()[:,None]
u_star = Exact_u.flatten()[:,None]

plt.figure()
plt.scatter(rho_star,u_star,color='r',s=0.1)
plt.xlabel('Density(veh/m)',fontsize=14)
plt.ylabel('Speed(m/s)',fontsize=14)

x = rho_star
y = u_star
num_bins = 180
bin_edges = np.linspace(0.05, 0.5, num_bins + 1)

# 计算每个小区间x的均值
bin_means_x = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(num_bins)]

# 统计每个小区间内的散点数值和计算均值
bin_counts = np.histogram(x, bins=bin_edges)[0]
bin_means = [np.mean(y[(x >= bin_edges[i]) & (x < bin_edges[i + 1])]) for i in range(num_bins)]

# 计算方差
bin_variances = [np.var(y[(x >= bin_edges[i]) & (x < bin_edges[i + 1])]) for i in range(num_bins)]
print(len(bin_variances))
# plt.figure()
# plt.xlabel(r'$\rho$')
# plt.ylabel(r"u")
# plt.legend()
plt.plot(bin_means_x,bin_means,label='Mean', linewidth=1, color='blue')
plt.plot(bin_means_x,bin_variances,color='blue',label='Variance',linestyle='--',linewidth=1 )


import numpy as np
from scipy.optimize import minimize

def func2(k, vf, kc, m):
    return vf/(1+(k/kc)**m)**(2/m)
# 定义目标函数
def objective_lsm(params, x_data, y_data):
    vf, kc,m = params
    y_hat = func2(x_data, vf, kc, m)
    error = y_data - y_hat
    weighted_error = error
    cost = np.sum(weighted_error ** 2)
    return cost

# 初始参数猜测值
initial_guess = [20.0, 0.2, 1.5]
# 使用优化算法找到最优参数
result_lsm = minimize(objective_lsm, initial_guess, args=(bin_means_x, bin_means))
vf_optimal_lsm, kc_optimal_lsm, m = result_lsm.x
print("最优参数 vf_lsm:", vf_optimal_lsm)
print("最优参数 kc_lsm:", kc_optimal_lsm)
print("最优参数 m_lsm:", m)

xx2 = np.arange(0.03, 0.5, 0.001)
x2 = bin_means_x
y2 = bin_means
yy2_mean = func2(xx2,vf_optimal_lsm,kc_optimal_lsm,m)
# plt.figure()
# plt.plot(bin_means_x,bin_means,label='Mean')
plt.plot(xx2, yy2_mean,linewidth=1,color='k',label='S3 fundamental diagram')
# plt.xlabel(r'$\rho$')
# plt.ylabel(r"u")
plt.legend()
# plt.show()

def func3(k, miu, sigma, y, A):
    return y + A/(k*sigma*np.sqrt(2*np.pi))*(np.e**(-(np.log(k)/miu)**2/(2*sigma**2)))
# 定义目标函数
def objective_lsm(params, x_data, y_data):
    miu, sigma, y, A = params
    y_hat = func3(x_data, miu, sigma, y, A)
    error = y_data - y_hat
    weighted_error = error
    cost = np.sum(weighted_error ** 2)
    return cost

# 初始参数猜测值
initial_guess = [0.1, 9.0, -0.2,1]
bin_means_x1 = np.array(bin_means_x)
# 使用优化算法找到最优参数
result_lsm = minimize(objective_lsm, initial_guess, args=(bin_means_x1, bin_variances))
miu_optimal_lsm, sigma_optimal_lsm, y_optimal_lsm,A_optimal_lsm = result_lsm.x
print("最优参数 miu_lsm:", miu_optimal_lsm)
print("最优参数 sigma_lsm:", sigma_optimal_lsm)
print("最优参数 y_lsm:", y_optimal_lsm)
print("最优参数 A_lsm:", A_optimal_lsm)

xx2 = np.arange(0.03, 0.5, 0.001)
x2 = bin_means_x
y2 = bin_variances

yy2_variance = func3(xx2,miu_optimal_lsm,sigma_optimal_lsm,y_optimal_lsm,A_optimal_lsm)

# plt.plot(bin_means_x,bin_variances,label='Variance')
plt.plot(xx2, yy2_variance,'k',linestyle='--',label='Log-normal density function',linewidth=1)
# plt.xlabel(r'$\rho$')
# plt.ylabel(r"u")
# plt.title('curve_fit')

plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.xlim(0,0.6)
plt.ylim(0,25)
plt.savefig('Stochastic_process.png', dpi=400)
plt.show()
