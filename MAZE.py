import random
import numpy as np


def randomcolor():
    colorarr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = "#" + ''.join([random.choice(colorarr) for i in range(6)])
    return color


class Node:
    """节点类表示搜索树中的每一个点。"""

    def __init__(self, parent=None, position=None):
        self.parent = parent  # 该节点的父节点
        self.position = position  # 节点在迷宫中的坐标位置
        self.g = 0  # G值：从起点到当前节点的成本
        self.h = 0  # H值：当前节点到目标点的估计成本
        self.f = 0  # F值：G值与H值的和，即节点的总评估成本

    # 比较两个节点位置是否相同
    def __eq__(self, other):
        return self.position == other.position

    # 定义小于操作，以便在优先队列中进行比较
    def __lt__(self, other):
        return self.f < other.f


class NodePair:
    def __init__(self, startnode=None, endnode=None, value=None):
        self.StartNode = startnode
        self.EndNode = endnode
        self.Value = value


def distance_node(self, other):
    return np.sqrt((self.position[0] - other.position[0]) ** 2 + (self.position[1] - other.position[1]) ** 2)


def manhattan_distance_nodepair(self, other):
    deltax = abs(self[0] - other[0])
    deltay = abs(self[1] - other[1])
    return np.sqrt(2) * min(deltax, deltay) + abs(deltax - deltay)


def distance_nodepair(self, other):
    if self and other:
        return np.sqrt((self[0] - other[0]) ** 2 + (self[1] - other[1]) ** 2)
    return 0


def DrawCircle(maze, x, y, r):
    for j in range(y - r, y + r):
        for i in range(round(x - np.sqrt(r * r - (y - j) * (y - j))), round(x + np.sqrt(r * r - (y - j) * (y - j)))):
            maze[i, j] = 0


def DrawRectangle(maze, x, y, a, b):
    for i in range(round(x - a / 2), round(x + a / 2)):
        for j in range(round(y - b / 2), round(y + b / 2)):
            maze[i, j] = 0


def DrawDiagRectangle(maze, x, y, a, b):  # 右上到左下为长边
    y_ceil = int(y + np.sqrt(0.5) * (a + b))
    y_up = int(y + np.sqrt(0.5) * a)
    y_down = int(y - np.sqrt(0.5) * a)
    y_floor = int(y - np.sqrt(0.5) * (a + b))
    for j in range(y_up, y_ceil):
        for i in range(round(x - np.sqrt(0.5) * (a - b) / 2 - (y_ceil - j)),
                       round(x - np.sqrt(0.5) * (a - b) / 2 + (y_ceil - j))):
            maze[i, j] = 0
    for j in range(y_down, y_up):
        for i in range(round(x + np.sqrt(0.5) * (a + b) - (y_up - j) - np.sqrt(2) * b) - 3,
                       round(x + np.sqrt(0.5) * (a + b) - (y_up - j)) - 2):
            maze[i, j] = 0
    for j in range(y_floor, y_down):
        for i in range(round(x + np.sqrt(0.5) * (a - b) / 2 - (j - y_floor)) - 5,
                       round(x + np.sqrt(0.5) * (a - b) / 2 + (j - y_floor)) - 5):
            maze[i, j] = 0


def init_maze(MazeSizeX, MazeSizeY, StartData, EndData, ValueData, NodePairSet, ColorData, num):
    # 设定起始点和终点
    maze = np.ones((MazeSizeX, MazeSizeY))
    '''
# 定义几个障碍物区块，每个障碍物区块是一个矩形
    obstacle_blocks = [
        (5, 20, 10, 10),  # (y起始, x起始, 高度, 宽度)
        (20, 20, 10, 10),
        (35, 20, 10, 10),
    ]

# 在迷宫中设置障碍物
    for y_start, x_start, height, width in obstacle_blocks:
        maze[y_start:y_start + height, x_start:x_start + width] = 1
    '''
    for i in range(num):
        NodePairSet.append([StartData[i], EndData[i], ValueData[i], ColorData[i]])
    '''
    DrawCircle(maze, 16 // 5, 16 // 5, 10 // 5)
    DrawCircle(maze, 215 // 5, 16 // 5, 10 // 5)
    DrawCircle(maze, 16 // 5, 215 // 5, 10 // 5)
    DrawCircle(maze, 215 // 5, 215 // 5, 10 // 5)
    DrawCircle(maze, 102 // 5, 43 // 5, 17 // 5)
    DrawCircle(maze, 87 // 5, 140 // 5, 17 // 5)
    DrawCircle(maze, 149 // 5, 53 // 5, 5 // 5)
    DrawCircle(maze, 149 // 5, 129 // 5, 5 // 5)
    DrawCircle(maze, 111 // 5, 91 // 5, 5 // 5)
    DrawCircle(maze, 187 // 5, 91 // 5, 5 // 5)

    DrawRectangle(maze, 32 // 5, 20 // 5, 15 // 5, 8 // 5)
    DrawRectangle(maze, 32 // 5, 77 // 5, 10 // 5, 18 // 5)
    DrawRectangle(maze, 32 // 5, 133 // 5, 10 // 5, 18 // 5)
    DrawRectangle(maze, 32 // 5, 188 // 5, 10 // 5, 18 // 5)
    DrawRectangle(maze, 89 // 5, 189 // 5, 10 // 5, 18 // 5)
    DrawRectangle(maze, 144 // 5, 189 // 5, 10 // 5, 18 // 5)
    DrawRectangle(maze, 200 // 5, 189 // 5, 10 // 5, 18 // 5)
    DrawRectangle(maze, 131 // 5, 92 // 5, 10 // 5, 15 // 5)
    DrawRectangle(maze, 167 // 5, 92 // 5, 10 // 5, 15 // 5)

    #DrawDiagRectangle(maze, 127 // 5, 18 // 5, 9 // 5, 21 // 5)
    #DrawDiagRectangle(maze, 75 // 5, 71 // 5, 9 // 5, 21 // 5)
    #DrawDiagRectangle(maze, 62 // 5, 166 // 5, 9 // 5, 21 // 5)
    #DrawDiagRectangle(maze, 115 // 5, 113 // 5, 9 // 5, 21 // 5)
    '''
    # maze[MazeSizeX // 2, MazeSizeY // 2] = 0
    # maze[MazeSizeX // 2, MazeSizeY // 2 - 1] = 0

    # 确保起始点和终点是障碍物
    for i in range(num):
        maze[NodePairSet[i][0]] = 0.99
        maze[NodePairSet[i][1]] = 0.99

    return maze
