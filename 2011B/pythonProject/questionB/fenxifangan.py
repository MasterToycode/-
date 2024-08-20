import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

# 配置字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据读取
file_path = 'D:/competition/2024数学建模国赛/2011B/cumcm2011B附件2_全市六区交通网路和平台设置的数据表.xls'
nodes_data = pd.read_excel(file_path, sheet_name='全市交通路口节点数据')
platforms_data = pd.read_excel(file_path, sheet_name='全市交巡警平台')
districts_data = pd.read_excel(file_path, sheet_name='六城区的基本数据')

# 计算人口密度
districts_data['人口密度'] = districts_data['城区的人口'] / districts_data['城区的面积']

# 选择所需的列
nodes_data_all = nodes_data[['全市路口节点标号', '路口的横坐标X', '路口的纵坐标Y', '发案率(次数)', '路口所属区域']]
platforms_data = platforms_data[['交巡警平台编号', '交巡警平台位置标号']]

# 获取平台位置函数
def get_platform_positions(platform_ids, all_nodes):
    platform_positions = []
    for platform_id in platform_ids:
        platform_data = all_nodes[all_nodes['全市路口节点标号'] == platform_id]
        if not platform_data.empty:
            platform_positions.append(platform_data[['路口的横坐标X', '路口的纵坐标Y']].values[0])
    return np.array(platform_positions)

# 获取平台的位置
platform_ids = platforms_data['交巡警平台位置标号'].values
platform_positions = get_platform_positions(platform_ids, nodes_data_all)

# 计算路口节点坐标
node_coords = nodes_data_all[['路口的横坐标X', '路口的纵坐标Y']].values
# 计算距离矩阵
dist_matrix = distance_matrix(node_coords, platform_positions)
# 模拟交通拥堵因子的影响
def apply_traffic_conditions(distance_matrix, traffic_factors):
    adjusted_distances = distance_matrix.copy()
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            adjusted_distances[i, j] *= traffic_factors[j]  # 交通因子影响
    return adjusted_distances
# 道路条件假设（简化示例）
def apply_road_conditions(distances, road_conditions):
    adjusted_distances = distances.copy()
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            adjusted_distances[i, j] *= road_conditions[j]  # 道路条件影响
    return adjusted_distances

# 假设整体交通情况影响，例如，假设所有平台受到10%的交通拥堵影响
traffic_factors = np.ones(platform_positions.shape[0]) * 1.1

# 道路条件因子（假设值）
# 假设整体道路条件影响，例如，假设所有平台受到5%的道路条件影响
road_conditions = np.ones(platform_positions.shape[0]) * 1.05

# 假设1毫米代表100米，60km/h代表每秒行驶16.67米
response_time_matrix = dist_matrix * 100 / 16.67  # 计算响应时间，单位秒

# 应用交通拥堵和道路条件因子的距离调整
adjusted_response_time_matrix = apply_traffic_conditions(response_time_matrix, traffic_factors)
adjusted_response_time_matrix = apply_road_conditions(adjusted_response_time_matrix, road_conditions)
# 计算每个区域的调整后的平均响应时间
def calculate_weighted_crime_rate(x):
    area = x['发案率(次数)'].sum()
    weighted_crime_rate = (x['发案率(次数)'] * x['发案率(次数)'].sum()).sum() / area
    return pd.Series({
        '平均响应时间': adjusted_response_time_matrix[x.index].min(axis=1).mean(),
        '加权总发案率': weighted_crime_rate,
        '路口数量': len(x)
    })

coverage_by_district = nodes_data_all.groupby('路口所属区域').apply(calculate_weighted_crime_rate).reset_index()









# 合并人口密度信息
coverage_by_district = coverage_by_district.merge(districts_data[['全市六个城区', '人口密度']],
                                                  left_on='路口所属区域',
                                                  right_on='全市六个城区').drop(columns='全市六个城区')
# 输出各区的响应时间和发案率
print(coverage_by_district)

# 可视化
fig, ax1 = plt.subplots(figsize=(12, 8))

ax1.set_xlabel('城区')
ax1.set_ylabel('平均响应时间 (秒)', color='tab:red')
ax1.bar(coverage_by_district['路口所属区域'], coverage_by_district['平均响应时间'], color='tab:red', alpha=0.6)
ax1.tick_params(axis='y', labelcolor='tab:red')

ax2 = ax1.twinx()
ax2.set_ylabel('加权总发案率', color='tab:blue')
ax2.plot(coverage_by_district['路口所属区域'], coverage_by_district['加权总发案率'], color='tab:blue', marker='o')
ax2.tick_params(axis='y', labelcolor='tab:blue')

plt.title('各城区的平均响应时间与加权总发案率')
plt.show()

#加权总发案率：我们通过将每个路口的发案率乘以该路口的发案率总和，然后求和
# ，最后除以总发案率，得到加权后的总发案率。这样考虑了发案率的分布情况。
