import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取Excel文件
file_path = 'D:/competition/数学建模/china_influenza2.xlsx'
data = pd.read_excel(file_path)

# 转换日期格式
data['日期'] = pd.to_datetime(data['日期'])

# 定义SEIR模型
def SEIR_model(y, t, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# 定义拟合函数
def fit_SEIR(t, beta, sigma, gamma):
    y0 = S0, E0, I0, R0
    ret = odeint(SEIR_model, y0, t, args=(beta, sigma, gamma))
    return ret[:, 2]  # 只返回I（感染者）用于拟合

# 初始条件
N = 1000000  # 总人口数，可以调整
I0 = data['甲型流感总样本数'].dropna().values[0]
E0 = I0 / 2  # 初始潜伏者数量，假设为感染者的一半
R0 = 0
S0 = N - I0 - E0 - R0

# 时间序列
t = np.arange(len(data))

# 实际感染人数（甲型）
cases_A = data['甲型流感总样本数'].values

# 去除缺失值以进行拟合
t_fit = t[np.isfinite(cases_A)]
cases_A_fit = cases_A[np.isfinite(cases_A)]

# 拟合模型参数
popt, pcov = curve_fit(fit_SEIR, t_fit, cases_A_fit, bounds=(0, [1, 1, 1]))
beta, sigma, gamma = popt
print(f"Fitted parameters: beta={beta}, sigma={sigma}, gamma={gamma}")

# 使用拟合的参数预测感染人数
fitted = fit_SEIR(t, beta, sigma, gamma)

# 用预测值填充缺失值
cases_A_filled = np.where(np.isfinite(cases_A), cases_A, fitted)

# 预测未来几年
future_days = 365  # 预测未来3年
t_future = np.arange(len(data) + future_days)
future_predictions = fit_SEIR(t_future, beta, sigma, gamma)

# 绘制实际数据、填充后的数据和未来预测数据
plt.figure(figsize=(12, 6))
plt.plot(t, cases_A, 'o', label='实际甲型流感样本数', markersize=5)
plt.plot(t, cases_A_filled, '-', label='填充后的甲型流感样本数', color='blue')
plt.plot(t_future, future_predictions, '--', label='未来预测甲型流感样本数', color='red')
plt.xlabel('时间（天）')
plt.ylabel('样本数')
plt.title('SEIR模型预测甲型流感感染人数')

# 调整横轴刻度
plt.xticks(np.arange(0, len(t_future), step=50))

plt.legend(loc='upper left')
plt.grid(True)
plt.show()
