import heapq
import MAZE
import numpy as np


def astar(maze, start, end, statu, pointdata):
    """A*算法实现，用于在迷宫中找到从起点到终点的最短路径。"""
    start_node = MAZE.Node(None, start)  # 创建起始节点
    end_node = MAZE.Node(None, end)  # 创建终点节点
    open_list = []  # 开放列表用于存储待访问的节点
    closed_list = []  # 封闭列表用于存储已访问的节点
    heapq.heappush(open_list, (start_node.f, start_node))  # 将起始节点添加到开放列表
    # print("添加起始节点到开放列表。")

    # 当开放列表非空时，循环执行
    while open_list:
        current_node = heapq.heappop(open_list)[1]
        if current_node in closed_list:
            continue  # 弹出并返回开放列表中 f 值最小的节点
        closed_list.append(current_node)  # 将当前节点添加到封闭列表
        # print(f"当前节点: {current_node.position}", current_node.g)

        # 如果当前节点是目标节点，则回溯路径
        if current_node == end_node:
            total_g = current_node.g
            path = []
            while current_node:
                maze[current_node.position[0], current_node.position[1]] = 0.999
                if current_node.parent and abs(current_node.position[0] - current_node.parent.position[0]) + abs(
                        current_node.position[1] - current_node.parent.position[1]) == 2:
                    maze[current_node.parent.position[0], current_node.position[1]] = 0.999
                    maze[current_node.position[0], current_node.parent.position[1]] = 0.999
                path.append(current_node.position)
                current_node = current_node.parent
            path.append(total_g)
            # print("找到目标节点，返回路径。")
            return path[::-1]  # 返回反向路径，即从起点到终点的路径

        # 获取当前节点周围的相邻节点
        (x, y) = current_node.position
        neighbors = [(x - 1, y), (x - 1, y + 1), (x, y + 1), (x + 1, y + 1), (x + 1, y), (x + 1, y - 1), (x, y - 1),
                     (x - 1, y - 1)]
        point_list = check_start_or_end(maze, (x, y))
        # neighbors.sort(key=lambda X: (MAZE.manhattan_distance_nodepair(X, (end_node.position[0], end_node.position[1])) + 1 + ((1 + x + y + X[0] + X[1]) % 2) * (np.sqrt(2) - 1)))# 考虑到直线斜线每步路径消耗有差异，补上再排序
        # neighbors.sort(key=lambda X: (MAZE.manhattan_distance_nodepair(X, (end_node.position[0], end_node.position[1])) - 0 * MAZE.distance_nodepair(check_start_end(maze, (x, y)), X)))  # 考虑到直线斜线每步路径消耗有差异，补上再排序
        neighbors.sort(
            key=lambda X: (MAZE.manhattan_distance_nodepair(X, (end_node.position[0], end_node.position[1]))))
        neighbors = neighbors[:statu]

        # 远离密集区域
        IsResortDense = check_start_or_end(maze, current_node.position)
        if IsResortDense:
            neighbors.sort(key=lambda X: abs(X[0] * IsResortDense[0] + X[1] * IsResortDense[1]))  # 垂直路径远离

        # 该线路可能布线的平行四边形范围内存在其他起点终点时，分配线路
        IsResortCross = 0

        # print(neighbors)
        # print(end_node.position)
        # 遍历相邻节点
        for next in neighbors:  # 可优化 rubin
            # 确保相邻节点在迷宫范围内，且不是障碍物
            neighbor = MAZE.Node(current_node, next)  # 创建相邻节点
            if 0 <= next[0] < maze.shape[0] and 0 <= next[1] < maze.shape[1]:
                if abs(next[0] - current_node.position[0]) + abs(next[1] - current_node.position[1]) == 1:
                    if not maze[next[0], next[1]] == 1:
                        continue
                else:
                    if (not maze[next[0], next[1]] == 1) or (not maze[next[0], current_node.position[1]] == 1) or not (
                            maze[current_node.position[0], next[1]] == 1):
                        continue
                # 如果相邻节点已在封闭列表中，跳过不处理
                if neighbor in closed_list:
                    continue
                neighbor.g = current_node.g + MAZE.distance_node(current_node, neighbor)  # 计算相邻节点的 G 值
                neighbor.h = 0 * (((end_node.position[0] - next[0]) ** 2) + (
                        (end_node.position[1] - next[1]) ** 2))  # 计算 H 值.启发式函数设计，有很大操作空间
                neighbor.f = neighbor.g + neighbor.h  # 计算 F 值

                # 如果相邻节点的新 F 值较小，则将其添加到开放列表
                if add_to_open(open_list, neighbor) == 1:
                    heapq.heappush(open_list, (neighbor.f, neighbor))
                    # print(f"添加节点 {neighbor.position} 到开放列表。")
            # else:
            # print(f"节点 {next} 越界或为障碍。")

            # 如果当前节点是目标节点，则回溯路径
            if neighbor == end_node:
                total_g = neighbor.g
                path = []
                while neighbor:
                    maze[neighbor.position[0], neighbor.position[1]] = 0.999
                    if neighbor.parent and abs(
                            neighbor.position[0] - neighbor.parent.position[0]) + abs(
                        neighbor.position[1] - neighbor.parent.position[1]) == 2:
                        maze[neighbor.parent.position[0], neighbor.position[1]] = 0.999
                        maze[neighbor.position[0], neighbor.parent.position[1]] = 0.999
                    path.append(neighbor.position)
                    neighbor = neighbor.parent
                path.append(total_g)
                # print("找到目标节点，返回路径。")
                return path[::-1]  # 返回反向路径，即从起点到终点的路径

    return None  # 如果没有找到路径，返回 None


def add_to_open(open_list, neighbor):
    """检查并添加节点到开放列表。"""
    for node in open_list:
        # 如果开放列表中已存在相同位置的节点且 G 值更低，不添加该节点
        if neighbor == node[1]:
            if neighbor.g >= node[1].g:
                return 0
            else:
                node[1].g = neighbor.g
                return 0  # 添加，且删除原节点
    return 1  # 如果不存在，则返回 True 以便添加该节点到开放列表


def check_start_or_end(maze, position):  # 用于应对起点终点密集
    point_edge = None
    counter = 0
    for i in range(position[0] - 3, position[0] + 4):
        for j in range(position[1] - 3, position[1] + 4):
            if i < maze.shape[0] and j < maze.shape[1] and maze[i][j] == 0.99:
                counter += 1
                if not point_edge or MAZE.distance_nodepair(point_edge, position) < MAZE.distance_nodepair((i, j),
                                                                                                           position):
                    point_edge = (i, j)
    if counter > 4:
        return point_edge
    return None


'''
def check_start_end_cross(maze, position, pointdata):  # 用于避免线路穿过较小范围内的起终点连线，使得该线路绕路
    for i in range(len(pointdata)):

    return None
'''
