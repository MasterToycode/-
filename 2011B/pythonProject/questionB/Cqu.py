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
