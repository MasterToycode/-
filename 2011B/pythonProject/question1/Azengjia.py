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
exit_node_positions = nodes_data_A[nodes_data_A['全市路口节点标号'].isin(exit_nodes_A)][
    ['全市路口节点标号', '路口的横坐标X', '路口的纵坐标Y']]

# 获取A区的20个交巡警平台的位置标号
platform_ids = platforms_data['交巡警平台位置标号'].head(20)

# 根据平台位置标号获取对应的坐标
platform_positions = nodes_data_A[nodes_data_A['全市路口节点标号'].isin(platform_ids)][
    ['全市路口节点标号', '路口的横坐标X', '路口的纵坐标Y']]


def compute_distance_matrix(platform_positions, exit_node_positions):
    # 使用 numpy 数组来计算距离矩阵
    platform_coords = platform_positions[['路口的横坐标X', '路口的纵坐标Y']].values
    exit_coords = exit_node_positions[['路口的横坐标X', '路口的纵坐标Y']].values
    distance_matrix = np.sqrt(((platform_coords[:, np.newaxis] - exit_coords) ** 2).sum(axis=2))
    return distance_matrix


def build_graph(platform_positions, exit_node_positions, distance_matrix):
    G = nx.DiGraph()
    source = 'source'
    sink = 'sink'
    G.add_node(source)
    G.add_node(sink)

    platform_nodes = [f'platform_{i}' for i in range(len(platform_positions))]
    exit_nodes = [f'exit_{j}' for j in range(len(exit_node_positions))]

    for platform_node in platform_nodes:
        G.add_edge(source, platform_node, capacity=1)

    for i in range(len(platform_positions)):
        for j in range(len(exit_node_positions)):
            G.add_edge(platform_nodes[i], exit_nodes[j], capacity=1)

    for exit_node in exit_nodes:
        G.add_edge(exit_node, sink, capacity=1)

    return G, platform_nodes, exit_nodes


def allocate_platforms(G, platform_nodes, exit_nodes):
    flow_value, flow_dict = nx.maximum_flow(G, 'source', 'sink')
    assignments = {j: [] for j in range(len(exit_node_positions))}
    for platform_node in platform_nodes:
        for exit_node in exit_nodes:
            if flow_dict[platform_node].get(exit_node, 0) > 0:
                platform_index = platform_nodes.index(platform_node)
                exit_index = exit_nodes.index(exit_node)
                assignments[exit_index].append(platform_index)
    return assignments


def add_and_reassign(num_additional):
    best_assignments = None
    min_max_distance = float('inf')
    new_platform_positions = []

    for _ in range(10):  # 尝试多个位置以找到最佳配置
        # 生成新平台位置
        new_positions = np.random.uniform(
            low=[platform_positions['路口的横坐标X'].min(), platform_positions['路口的纵坐标Y'].min()],
            high=[platform_positions['路口的横坐标X'].max(), platform_positions['路口的纵坐标Y'].max()],
            size=(num_additional, 2))
        new_platform_positions = new_positions

        all_positions = np.vstack([platform_positions[['路口的横坐标X', '路口的纵坐标Y']].values, new_positions])
        new_distance_matrix = compute_distance_matrix(
            pd.DataFrame(all_positions, columns=['路口的横坐标X', '路口的纵坐标Y']), exit_node_positions)

        G, platform_nodes, exit_nodes = build_graph(
            pd.DataFrame(all_positions, columns=['路口的横坐标X', '路口的纵坐标Y']), exit_node_positions,
            new_distance_matrix)
        assignments = allocate_platforms(G, platform_nodes, exit_nodes)

        # 计算最大响应时间
        max_distance = max([new_distance_matrix[i, j] for i in range(len(platform_positions), len(all_positions))
                            for j in assignments.keys() if i in assignments[j]] or [0])

        if max_distance < min_max_distance:
            min_max_distance = max_distance
            best_assignments = assignments

    return best_assignments, new_platform_positions


# 初始分配
distance_matrix = compute_distance_matrix(platform_positions, exit_node_positions)
G, platform_nodes, exit_nodes = build_graph(platform_positions, exit_node_positions, distance_matrix)
assignments = allocate_platforms(G, platform_nodes, exit_nodes)

# 打印初始分配结果
print("\n初始分配信息:")
for exit_index, platforms in assignments.items():
    exit_node_id = exit_node_positions.iloc[exit_index]['全市路口节点标号']
    print(f"路口 {exit_node_id} 被分配到的平台:", end=" ")
    for platform_index in platforms:
        platform_id = platform_positions.iloc[platform_index]['全市路口节点标号']
        print(platform_id, end=" ")
    print()

# 增加2到5个新平台
num_additional = 5  # 你可以设置为2到5之间的任意数
print(f"\n模拟增加 {num_additional} 个平台")
best_assignments, new_platform_positions = add_and_reassign(num_additional)

# 打印新增平台的位置
print("\n新增平台的位置:")
for pos in new_platform_positions:
    print(f"新平台坐标: {pos}")

# 绘制最终分配结果
plt.figure(figsize=(12, 10))

# 画出原有平台的点
plt.scatter(platform_positions['路口的横坐标X'], platform_positions['路口的纵坐标Y'], c='blue', label='原有交巡警平台',
            s=100)

# 画出新增平台的点
if new_platform_positions.size > 0:
    plt.scatter(new_platform_positions[:, 0], new_platform_positions[:, 1], c='green', label='新增交巡警平台', s=100)

# 画出13条交通要道路口的点
plt.scatter(exit_node_positions['路口的横坐标X'], exit_node_positions['路口的纵坐标Y'], c='red', label='交通要道路口',
            s=100)

# 连接平台和路口
for j, platform_list in best_assignments.items():
    exit_x = exit_node_positions.iloc[j]['路口的横坐标X']
    exit_y = exit_node_positions.iloc[j]['路口的纵坐标Y']
    for i in platform_list:
        if i < len(platform_positions):
            platform_x = platform_positions.iloc[i]['路口的横坐标X']
            platform_y = platform_positions.iloc[i]['路口的纵坐标Y']
        else:
            platform_x = new_platform_positions[i - len(platform_positions)][0]
            platform_y = new_platform_positions[i - len(platform_positions)][1]

        plt.plot([platform_x, exit_x], [platform_y, exit_y], 'g--', lw=2)
        plt.text(platform_x, platform_y,
                 f'P{platform_positions.iloc[i]["全市路口节点标号"] if i < len(platform_positions) else "N" + str(i - len(platform_positions))}',
                 fontsize=12, ha='right')
    plt.text(exit_x, exit_y, f'R{exit_node_positions.iloc[j]["全市路口节点标号"]}', fontsize=12, ha='right')

plt.title('交巡警平台与交通要道封锁调度图')
plt.xlabel('横坐标X')
plt.ylabel('纵坐标Y')
plt.legend()
plt.grid(True)
plt.show()
