import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
file_path = 'D:/competition/2024数学建模国赛/2011B/cumcm2011B附件2_全市六区交通网路和平台设置的数据表.xls'
nodes_data = pd.read_excel(file_path, sheet_name='全市交通路口节点数据')
platforms_data = pd.read_excel(file_path, sheet_name='全市交巡警平台')

# 提取A区的节点数据
nodes_data_A = nodes_data[nodes_data['路口所属区域'] == 'A']

# 获取13条交通要道的路口数据
exit_nodes_A = [12, 14, 16, 21, 22, 23, 24, 28, 29, 30, 38, 48, 62]

# 获取13条交通要道在A区的路口坐标
exit_node_positions = nodes_data_A[nodes_data_A['全市路口节点标号'].isin(exit_nodes_A)][['全市路口节点标号', '路口的横坐标X', '路口的纵坐标Y']]

# 获取A区的20个交巡警平台的位置标号
platform_ids = platforms_data['交巡警平台位置标号'].head(20)

# 根据平台位置标号获取对应的坐标
platform_positions = nodes_data_A[nodes_data_A['全市路口节点标号'].isin(platform_ids)][['全市路口节点标号', '路口的横坐标X', '路口的纵坐标Y']]

# 计算每个平台到每个路口的距离矩阵
distance_matrix = np.zeros((len(platform_positions), len(exit_node_positions)))

for i, platform_pos in enumerate(platform_positions[['路口的横坐标X', '路口的纵坐标Y']].values):
    for j, exit_pos in enumerate(exit_node_positions[['路口的横坐标X', '路口的纵坐标Y']].values):
        distance_matrix[i, j] = np.sqrt((platform_pos[0] - exit_pos[0])**2 + (platform_pos[1] - exit_pos[1])**2)

# 构建网络流图
G = nx.DiGraph()

# 添加源点和汇点
source = 'source'
sink = 'sink'
G.add_node(source)
G.add_node(sink)

# 添加平台节点和路口节点
platform_nodes = [f'platform_{i}' for i in range(len(platform_positions))]
exit_nodes = [f'exit_{j}' for j in range(len(exit_node_positions))]

# 从源点到平台节点的边，容量为1
for platform_node in platform_nodes:
    G.add_edge(source, platform_node, capacity=1)

# 从平台节点到路口节点的边，容量为1
for i in range(len(platform_positions)):
    for j in range(len(exit_node_positions)):
        G.add_edge(platform_nodes[i], exit_nodes[j], capacity=1)

# 从路口节点到汇点的边，容量为1
for exit_node in exit_nodes:
    G.add_edge(exit_node, sink, capacity=1)

# 计算最大流
flow_value, flow_dict = nx.maximum_flow(G, source, sink)

# 输出流值（成功封锁的路口数量）
if flow_value == len(exit_node_positions):
    print("所有路口都已被封锁。")
else:
    print(f"仅封锁了 {flow_value} 个路口，无法封锁所有路口。")

# 输出分配结果
assignments = {j: [] for j in range(len(exit_node_positions))}
for platform_node in platform_nodes:
    for exit_node in exit_nodes:
        if flow_dict[platform_node][exit_node] > 0:
            platform_index = platform_nodes.index(platform_node)
            exit_index = exit_nodes.index(exit_node)
            assignments[exit_index].append(platform_index)

# 计算未使用的平台
used_platforms = set([i for sublist in assignments.values() for i in sublist])
remaining_platforms = set(range(len(platform_positions))) - used_platforms

# 将未使用的平台分配给距离较大的路口
for platform_index in remaining_platforms:
    platform_x = platform_positions.iloc[platform_index]['路口的横坐标X']
    platform_y = platform_positions.iloc[platform_index]['路口的纵坐标Y']
    max_distance = -1
    best_exit_index = None
    for exit_index in range(len(exit_node_positions)):
        exit_x = exit_node_positions.iloc[exit_index]['路口的横坐标X']
        exit_y = exit_node_positions.iloc[exit_index]['路口的纵坐标Y']
        distance = np.sqrt((platform_x - exit_x)**2 + (platform_y - exit_y)**2)
        if distance > max_distance:
            max_distance = distance
            best_exit_index = exit_index
    if best_exit_index is not None:
        assignments[best_exit_index].append(platform_index)

# 打印详细的分配信息
print("\n详细分配信息:")
for exit_index, platforms in assignments.items():
    exit_node_id = exit_node_positions.iloc[exit_index]['全市路口节点标号']
    print(f"路口 {exit_node_id} 被分配到的平台:", end=" ")
    for platform_index in platforms:
        platform_id = platform_positions.iloc[platform_index]['全市路口节点标号']
        print(platform_id, end=" ")
    print()

# 绘制最终分配结果
plt.figure(figsize=(10, 8))

# 画出平台的点
plt.scatter(platform_positions['路口的横坐标X'], platform_positions['路口的纵坐标Y'], c='blue', label='交巡警平台', s=100)

# 画出13条交通要道路口的点
plt.scatter(exit_node_positions['路口的横坐标X'], exit_node_positions['路口的纵坐标Y'], c='red', label='交通要道路口', s=100)

# 连接平台和路口
for j, platform_list in assignments.items():
    exit_x = exit_node_positions.iloc[j]['路口的横坐标X']
    exit_y = exit_node_positions.iloc[j]['路口的纵坐标Y']
    for i in platform_list:
        platform_x = platform_positions.iloc[i]['路口的横坐标X']
        platform_y = platform_positions.iloc[i]['路口的纵坐标Y']
        plt.plot([platform_x, exit_x], [platform_y, exit_y], 'g--', lw=2)
        plt.text(platform_x, platform_y, f'P{platform_positions.iloc[i]["全市路口节点标号"]}', fontsize=12, ha='right')
    plt.text(exit_x, exit_y, f'R{exit_node_positions.iloc[j]["全市路口节点标号"]}', fontsize=12, ha='right')

plt.title('交巡警平台与交通要道封锁调度图')
plt.xlabel('横坐标X')
plt.ylabel('纵坐标Y')
plt.legend()
plt.grid(True)
plt.show()
