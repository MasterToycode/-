import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlsxwriter
from matplotlib.dates import AutoDateLocator, DateFormatter

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取Excel文件
file_path = 'D:/competition/数学建模/china_influenza2.xlsx'
df = pd.read_excel(file_path)

# 获取甲型流感总样本数数据
value = df['甲型流感总样本数'].values
dates = pd.to_datetime(df['日期'].values)

# 从日期中提取月份和季度
months = dates.month.values
quarters = dates.quarter.values

from sklearn.preprocessing import StandardScaler

# 数据标准化
scaler = StandardScaler()
scaled_value = scaler.fit_transform(value.reshape(-1, 1)).flatten()

# 数据预处理
x = []
y = []
seq = 3
for i in range(len(value) - seq - 1):
    x_seq = value[i:i + seq]
    x_month = months[i:i + seq]
    x_quarter = quarters[i:i + seq]
    x.append(np.column_stack((x_seq, x_month, x_quarter)))
    y.append(value[i + seq])

x = np.array(x)
y = np.array(y)

train_x = (torch.tensor(x[:50]).float() / 1000.).reshape(-1, seq, 3)
train_y = (torch.tensor(y[:50]).float() / 1000.).reshape(-1, 1)
test_x = (torch.tensor(x[50:]).float() / 1000.).reshape(-1, seq, 3)
test_y = (torch.tensor(y[50:]).float() / 1000.).reshape(-1, 1)

# 定义改进的LSTM模型
class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=32, num_layers=3, batch_first=True, dropout=0.2, bidirectional=True)
        self.linear = nn.Linear(32 * 2 * seq, 1)  # 乘以2因为是双向LSTM

    def forward(self, x):
        x, (h, c) = self.lstm(x)
        x = x.reshape(-1, 32 * 2 * seq)  # 乘以2因为是双向LSTM
        x = self.linear(x)
        return x

# 模型训练
model = BiLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_func = nn.MSELoss()
model.train()
l = []
# 学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
for epoch in range(1000):
    output = model(train_x)
    loss = loss_func(output, train_y)
    l.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        test_loss = loss_func(model(test_x), test_y)
        print("epoch:{}, train_loss:{:.4f}, test_loss:{:.4f}".format(epoch, loss.item(), test_loss.item()))

# 模型预测、画图
model.eval()
train_pred = model(train_x).data.reshape(-1) * 1000
test_pred = model(test_x).data.reshape(-1) * 1000
prediction = list(train_pred) + list(test_pred)

# 修正预测长度
dates_prediction = dates[3:len(prediction) + 3]

# 绘制图表
plt.figure(figsize=(12, 6))
plt.plot(dates[3:], value[3:], label='实际感染人数', marker='o')
plt.plot(dates_prediction[:50], prediction[:50], label='LSTM拟合曲线', color='blue')
plt.plot(dates_prediction[50:], prediction[50:], label='LSTM预测曲线', color='red', linestyle='--')
plt.xlabel('时间')
plt.ylabel('感染人数')
plt.title('LSTM预测甲型流感感染人数')

# 调整横轴日期显示
locator = AutoDateLocator()
formatter = DateFormatter('%Y-%m-%d')
plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(formatter)
plt.gcf().autofmt_xdate()  # 自动旋转日期标签

plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# 写数据到Excel
workbook = xlsxwriter.Workbook('LSTM_result_data.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write('A1', '日期')
worksheet.write('B1', '实际感染人数')
worksheet.write('C1', 'LSTM拟合值')
worksheet.write('D1', 'LSTM预测值')

# 写入数据
for i in range(len(value[3:])):
    row = i + 1
    date = dates[i + 3].strftime('%Y-%m-%d')
    actual_cases = value[i + 3]
    fitted_cases = prediction[i] if i < 50 and i < len(prediction) else None
    predicted_cases = prediction[i] if i >= 50 and i < len(prediction) else None

    # 处理NaN和INF值
    actual_cases = 0 if np.isnan(actual_cases) or np.isinf(actual_cases) else actual_cases
    fitted_cases = 0 if fitted_cases is not None and (np.isnan(fitted_cases) or np.isinf(fitted_cases)) else fitted_cases
    predicted_cases = 0 if predicted_cases is not None and (np.isnan(predicted_cases) or np.isinf(predicted_cases)) else predicted_cases

    worksheet.write(row, 0, date)
    worksheet.write(row, 1, actual_cases)
    worksheet.write(row, 2, fitted_cases if fitted_cases is not None else '')
    worksheet.write(row, 3, predicted_cases if predicted_cases is not None else '')

workbook.close()
