import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

file_path = 'D:/competition/2024数学建模国赛/2011B/cumcm2011B附件2_全市六区交通网路和平台设置的数据表.xls'
nodes_data = pd.read_excel(file_path, sheet_name='全市交通路口节点数据')
platforms_data = pd.read_excel(file_path, sheet_name='全市交巡警平台')
nodes_data_all = nodes_data[['全市路口节点标号', '路口的横坐标X', '路口的纵坐标Y']]
platforms_data = platforms_data[['交巡警平台位置标号']]
# 获取案发地P点的数据
incident_node = nodes_data_all[nodes_data_all['全市路口节点标号'] == 32]
incident_position = incident_node[['路口的横坐标X', '路口的纵坐标Y']].values[0]

# 获取平台的坐标
def get_platform_positions(platform_ids, all_nodes):
    platform_positions = []
    for platform_id in platform_ids:
        platform_data = all_nodes[all_nodes['全市路口节点标号'] == platform_id]
        if not platform_data.empty:
            platform_positions.append(platform_data[['路口的横坐标X', '路口的纵坐标Y']].values[0])
    return np.array(platform_positions)

platform_ids = platforms_data['交巡警平台位置标号'].values
platform_positions = get_platform_positions(platform_ids, nodes_data_all)

# 地图到实际的比例尺
scale = 100

# 逃犯速度 (km/h) 转换为 m/min
fugitive_speed_kmh = 60
fugitive_speed_m_per_min = fugitive_speed_kmh * 1000 / 60
fugitive_escape_time = 3
fugitive_max_distance_m = fugitive_speed_m_per_min * fugitive_escape_time

# 计算案发点到每个路口的距离
def compute_distance(pos1, pos2, scale):
    distance_mm = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
    distance_m = distance_mm * scale
    return distance_m

# 找出逃犯可能到达的路口节点
possible_escape_nodes = nodes_data_all[nodes_data_all.apply(
    lambda row: compute_distance([row['路口的横坐标X'], row['路口的纵坐标Y']], incident_position,
                                 scale) <= fugitive_max_distance_m,
    axis=1)].copy()


outer_radius = fugitive_max_distance_m * 1.5  # 包围圈半径扩展至1.5倍，视为外围

# 确定逃逸位置是否在包围圈内
def is_within_circle(center, point, radius):
    return compute_distance(center, point, scale) <= radius

# 计算每个逃逸节点的威胁等级（威胁等级定义为距离案发地点的实际距离）
possible_escape_nodes['威胁等级'] = possible_escape_nodes.apply(
    lambda row: compute_distance([row['路口的横坐标X'], row['路口的纵坐标Y']], incident_position, scale),
    axis=1)

# 按威胁等级排序（从高到低），选择威胁等级前3的节点
top_threat_nodes = possible_escape_nodes.nlargest(3, '威胁等级')

# 计算每个平台到每个逃逸节点的距离
def compute_platform_escape_distances(platform_positions, escape_nodes_positions, scale):
    distances = []
    for platform_pos in platform_positions:
        for escape_node_pos in escape_nodes_positions:
            distance_m = compute_distance(platform_pos, escape_node_pos, scale)
            distances.append((platform_pos, escape_node_pos, distance_m))
    return distances

# 获取包围圈内和包围圈外的平台节点
within_circle_platforms = [pos for pos in platform_positions if is_within_circle(incident_position, pos, outer_radius)]
outside_circle_platforms = [pos for pos in platform_positions if not is_within_circle(incident_position, pos, outer_radius)]
escape_positions = top_threat_nodes[['路口的横坐标X', '路口的纵坐标Y']].values

# 计算平台到逃逸节点的距离
distances = compute_platform_escape_distances(outside_circle_platforms, escape_positions, scale)

# 将结果按距离排序，选择最近的前五个平台进行堵截
distances_sorted = sorted(distances, key=lambda x: x[2])
top_5_blocking_platforms = []
for platform_pos, _, _ in distances_sorted:
    # 确保不重复添加
    if len(top_5_blocking_platforms) < 5 and not any(np.array_equal(platform_pos, p) for p in top_5_blocking_platforms):
        top_5_blocking_platforms.append(platform_pos)

# 找到相应的平台标号
def get_platform_ids(positions, all_nodes):
    ids = []
    for pos in positions:
        matched_platforms = all_nodes[(all_nodes[['路口的横坐标X', '路口的纵坐标Y']] == pos).all(axis=1)]
        if not matched_platforms.empty:
            ids.extend(matched_platforms['全市路口节点标号'].values)
    return ids

# 打印追击和堵截平台标号
def print_platform_ids(platform_positions, strategy):
    platform_ids = get_platform_ids(platform_positions, nodes_data_all)
    print(f"策略: {strategy}")
    for platform_id in platform_ids:
        print(f"平台标号: {platform_id}")

print("追击平台:")
print_platform_ids(within_circle_platforms, "追击")

print("堵截平台:")
print_platform_ids(top_5_blocking_platforms, "堵截")

# 可视化
plt.figure(figsize=(12, 8))
plt.scatter(*incident_position, c='red', label='案发地点', s=200, marker='*')
plt.scatter(platform_positions[:, 0], platform_positions[:, 1], c='blue', label='交巡警平台', s=100)
plt.scatter(top_threat_nodes['路口的横坐标X'], top_threat_nodes['路口的纵坐标Y'], c='purple', edgecolor='black', s=200, marker='o', label='威胁前3位置')
outer_circle = plt.Circle(incident_position, outer_radius / scale, color='green', fill=False, linestyle='--',
                          label='包围圈')

plt.gca().add_artist(outer_circle)
# 绘制包围圈外的前5个平台到包围圈边界的箭头
for platform_pos in top_5_blocking_platforms:
    circle_edge_x = incident_position[0] + (outer_radius / scale) * (platform_pos[0] - incident_position[0]) / np.linalg.norm(platform_pos - incident_position)
    circle_edge_y = incident_position[1] + (outer_radius / scale) * (platform_pos[1] - incident_position[1]) / np.linalg.norm(platform_pos - incident_position)
    plt.arrow(*platform_pos, circle_edge_x - platform_pos[0], circle_edge_y - platform_pos[1],
              head_width=5, head_length=5, fc='blue', ec='blue')

# 绘制包围圈内的部分平台到威胁等级最高的逃逸节点的箭头
num_arrows_to_draw = 5
for platform_pos in within_circle_platforms[:num_arrows_to_draw]:
    for _, escape_node_pos in top_threat_nodes[['路口的横坐标X', '路口的纵坐标Y']].iterrows():
        plt.arrow(*platform_pos, *(escape_node_pos['路口的横坐标X'] - platform_pos[0], escape_node_pos['路口的纵坐标Y'] - platform_pos[1]),
                  head_width=5, head_length=5, fc='red', ec='red')

plt.xlabel('横坐标X')
plt.ylabel('纵坐标Y')
plt.title('围堵方案 - 威胁等级前3')
plt.legend()
plt.grid(True)
plt.show()
