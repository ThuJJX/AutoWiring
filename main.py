import time
import numpy as np
import random  # 定义随机生成颜色函数
import torch
from torch import nn, optim
import NET
import pandas as pd
import MAZE
import ASTAR
import VISUALIZE

# 获取数据
num = 0
mazesizeX = 160
mazesizeY = 250
StartData = []
EndData = []
ValueData = []

'''
data = pd.read_csv('data.csv')
for i in range(data.shape[0]):
    StartData.append((data.loc[i][0], data.loc[i][1]))
    EndData.append((data.loc[i][2], data.loc[i][3]))
    ValueData.append(random.random())
    num += 1
'''

data = pd.read_csv('坐标拾取结果.csv')
# print(data.shape[0])
for i in range(data.shape[0] // 2):
    StartData.append((int(data.iloc[2 * i].iloc[1]) // 5, int(data.iloc[2 * i].iloc[0]) // 5))
    EndData.append((int(data.iloc[2 * i + 1].iloc[1]) // 5, int(data.iloc[2 * i + 1].iloc[0]) // 5))
    # print(StartData, EndData)
    ValueData.append(MAZE.distance_nodepair(StartData[i], EndData[i]))
    # ValueData.append(i)
    num += 1

ColorData = [MAZE.randomcolor() for i in range(num)]

# 选择模型
# model = net.SimpleNet(num, 4 * num, 4 * num, 4 * num, num)
model = NET.ActivationNet(num, 2 * num, 4 * num, 4 * num, 4 * num, 2 * num, num)
# model = net.BatchNet(num, 4 * num, 4 * num, 4 * num, num)
# if torch.cuda.is_available():
#    model = model.cuda()

# 定义一些超参数
# batch_size = 64
learning_rate = 0.001
# num_epoches = 20

# 定义损失函数和优化器
criterion1 = nn.MSELoss()
criterion2 = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9, eps=1e-08, weight_decay=0, momentum=0,
                          centered=False)

# 训练模型
epoch = 0

PathSet = []
outputbias = [0 for i in range(num)]
output = ValueData
train_output = output
time1 = time.time()
time2 = time.time()
Loss2 = []
while epoch < 200:
    if epoch and epoch % 10 == 0:
        time2 = time.time()
        print(str(epoch) + "个epoch完成时间为" + str(time2 - time1) + "s")
        time1 = time.time()
    NodePairSet = []
    PathSet = []
    extraloss = 0
    label_data = []
    temp_data = []
    # print(output + outputbias)
    maze = MAZE.init_maze(mazesizeX, mazesizeY, StartData, EndData, list(np.add(train_output, outputbias)), NodePairSet,
                          ColorData, num)
    outputbias = [outputbias[i] / 2 for i in range(num)]  # 布线不通的惩罚
    maze_copy = np.array(maze)
    # print(NodePairSet)
    NodePairSet.sort(key=lambda x: x[2])
    for i in range(len(NodePairSet)):
        StartData[i] = NodePairSet[i][0]
        EndData[i] = NodePairSet[i][1]
        ColorData[i] = NodePairSet[i][3]
        train_output[i] = NodePairSet[i][2]
    for i in range(num):
        label_data.append(MAZE.manhattan_distance_nodepair(NodePairSet[i][0], NodePairSet[i][1]))
    # print(NodePairSet)
    PathCost = 0
    # print(NodePairSet)
    statu = 0
    Iter = 0
    num_not_found = 0
    while Iter < num:
        maze[NodePairSet[Iter][0]] = 1
        maze[NodePairSet[Iter][1]] = 1
        statu += 2
        temp_time1 = time.time()
        path = ASTAR.astar(maze, NodePairSet[Iter][0], NodePairSet[Iter][1], statu, NodePairSet)
        temp_time2 = time.time()
        # print(path)
        if path:
            # print("路径已找到：", path, )
            temp_data.append(int(path[0]))
            PathSet.append(path)
            PathCost += path[0]
            # if Iter % 10:
            #    print("已完成" + str(Iter) + "条")
            # VISUALIZE.visualize_path(maze, PathSet, NodePairSet, Iter)
            statu = 0
            maze[NodePairSet[Iter][0]] = 0.999
            maze[NodePairSet[Iter][1]] = 0.999
            Iter += 1
        else:
            if statu < 4 and temp_time2 - temp_time1 < 20:
                # print("扩大寻找范围。")
                continue
            outputbias[Iter] -= 5 / np.sqrt(epoch + 1)
            extraloss += 10
            # print("没有找到路径。")
            num_not_found += 1
            temp_data.append(0)
            PathSet.append(None)
            statu = 0
            Iter += 1

    img = torch.tensor(temp_data, dtype=torch.float)
    label = torch.tensor(label_data, dtype=torch.float)
    # img = torch.from_numpy(temp_data).float()
    # label = torch.from_numpy(label_data).float()
    # for data in train_loader:
    # img, label = data
    # img = img.view(img.size(0), -1)
    # if torch.cuda.is_available():
    #    img = img.cuda()
    #    label = label.cuda()
    # else:
    #    img = Variable(img)
    #    label = Variable(label)
    tempoutput = model(img).tolist()
    loss1 = 0.1 * criterion1(torch.tensor(sum(temp_data), dtype=torch.float),
                             torch.tensor(sum(label_data), dtype=torch.float)) / np.sqrt(mazesizeX * mazesizeY)
    loss2 = criterion1(torch.tensor(output, dtype=torch.float), torch.tensor(tempoutput, dtype=torch.float))
    loss = loss1 + loss2 + extraloss
    output = tempoutput
    if ((not epoch % 5) and num_not_found > 2) or not num_not_found:
        train_output = output
    print_loss = loss.data.item()

    optimizer.zero_grad()
    loss.requires_grad_(True)
    loss.backward()
    optimizer.step()
    epoch += 1
    print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))
    print('与上一布线方案之间的差异值为{:.4}'.format(loss2.data.item()))
    Loss2.append(loss2.data.item())
    if num_not_found:
        print("共有" + str(num_not_found) + "条线路不通，优化中......\n")
    else:
        print("全部布通，可参考" + str(epoch) + ".jpg\n")
        # 存储路径
    VISUALIZE.visualize_path(maze, PathSet, NodePairSet, epoch)
