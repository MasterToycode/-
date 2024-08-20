import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import Voronoi, voronoi_plot_2d

file_path = 'D:/competition/2024数学建模国赛/2011B/cumcm2011B附件2_全市六区交通网路和平台设置的数据表.xls'
nodes_data = pd.read_excel(file_path, sheet_name='全市交通路口节点数据')
platforms_data = pd.read_excel(file_path, sheet_name='全市交巡警平台')
routes_data = pd.read_excel(file_path, sheet_name='全市交通路口的路线')
nodes_data_A = nodes_data[nodes_data['路口所属区域'] == 'A']
platforms_data_A = platforms_data.head(20)


# 获取平台的坐标
platform_positions = nodes_data_A[['路口的横坐标X', '路口的纵坐标Y']].iloc[:20].values
all_nodes_positions = nodes_data_A[['路口的横坐标X', '路口的纵坐标Y']].values

# 每个平台的管辖最大半径为3 km = 3000 mm
radius = 3000

# 获取A区的路口连线数
route_lines = routes_data[routes_data['路线起点(节点）标号'].isin(nodes_data_A['全市路口节点标号']) & routes_data['路线终点（节点）标号'].isin(nodes_data_A['全市路口节点标号'])]

# 获取连线的坐标
route_positions = []
for _, row in route_lines.iterrows():
    start_node = nodes_data_A[nodes_data_A['全市路口节点标号'] == row['路线起点(节点）标号']]
    end_node = nodes_data_A[nodes_data_A['全市路口节点标号'] == row['路线终点（节点）标号']]
    if not start_node.empty and not end_node.empty:
        start_pos = start_node[['路口的横坐标X', '路口的纵坐标Y']].values[0]
        end_pos = end_node[['路口的横坐标X', '路口的纵坐标Y']].values[0]
        route_positions.append((start_pos, end_pos))


# 创建Voronoi图
vor = Voronoi(platform_positions)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

# 可视化Voronoi图并显示所有路口
plt.figure(figsize=(14, 14))
voronoi_plot_2d(vor, show_vertices=False, line_colors='orange', line_width=2)

# 绘制所有路口的位置
plt.scatter(all_nodes_positions[:, 0], all_nodes_positions[:, 1], color='blue', label='路口位置')
# 绘制平台的位置
plt.scatter(platform_positions[:, 0], platform_positions[:, 1], color='red', label='交巡警服务平台', zorder=5)

# 绘制A区的路口连线
for start_pos, end_pos in route_positions:
    plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], color='green', linestyle='-', linewidth=1, zorder=2)

# 绘制每个平台的管辖区域
for idx, pos in enumerate(platform_positions):
    circle = Circle(pos, radius, facecolor='purple', edgecolor='black', fill=False, alpha=0.3, linestyle='-', linewidth=1.5)
    plt.gca().add_patch(circle)
    plt.text(pos[0], pos[1], f'平台 {idx+1}', fontsize=9, ha='center', va='center', color='black', weight='bold')


plt.title('A区交巡警服务平台管辖范围示意图')
plt.xlabel('X坐标 (毫米)')
plt.ylabel('Y坐标 (毫米)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()