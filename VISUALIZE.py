import numpy as np
import matplotlib.pyplot as plt


def visualize_path(maze, pathset, nodepairset, epoch):
    """将找到的路径可视化在迷宫上。"""

    maze_copy = np.array(maze)
    #plt.imshow(maze_copy, cmap='gray', interpolation='nearest')
    plt.gca().invert_yaxis()

    for i in range(len(nodepairset)):
        start_x, start_y = nodepairset[i][0][1], nodepairset[i][0][0]
        end_x, end_y = nodepairset[i][1][1], nodepairset[i][1][0]
        plt.scatter([start_x], [start_y], color=nodepairset[i][3], s=3, label='Start', zorder=1)  # 起点为绿色圆点
        plt.scatter([end_x], [end_y], color=nodepairset[i][3], s=3, label='End', marker='x')  # 终点为红色圆点
        if not i:
            plt.legend()

    for i in range(len(pathset)):
        if pathset[i]:
            # previous_path = pathset[i]
            pathset[i].pop(0)
            # print(pathset[i])
            # maze_copy = np.array(maze)
            # for step in pathset[i]:
            #    maze_copy[step] = 0.5  # 标记路径上的点
            # plt.figure(figsize=(10, 10))
            # 将迷宫中的通道显示为黑色，障碍物为白色
            # plt.imshow(maze_copy, cmap='hot', interpolation='nearest')
            # 提取路径上的x和y坐标
            path_x = [p[1] for p in pathset[i]]  # 列坐标
            path_y = [p[0] for p in pathset[i]]  # 行坐标
            # 绘制路径
            plt.plot(path_x, path_y, color=nodepairset[i][3], linewidth=1)
            # 绘制起点和终点

            # print(nodepairset)
            # 添加图例
            # # 隐藏坐标轴
            # plt.axis('off')
            # 显示图像
    plt.savefig(str(epoch) + ".jpg", dpi=2000)
    plt.clf()
