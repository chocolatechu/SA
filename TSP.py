import scipy.io as scio
import numpy as np
import math
import random
import copy
import matplotlib.pyplot as plt

data = scio.loadmat("F:\code\SA\citys_data.mat")
city = data["citys"].astype(int)
city = city[0:15]


n = len(city)

# 计算两个地方之间的距离
D = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i == j:
            D[i, j] = 1e-4
        else:
            D[i, j] = np.sqrt((city[i, 0]-city[j, 0])**2 + (city[i, 1]-city[j, 1])**2)

# 初始化一些参数
T0 = 1e200   # 初始问题
Tend = 1e-30    # 终止温度
L = 2   # 各温度迭代的次数
q = 0.9     # 降温速率
time = math.log(Tend/T0, 0.9)   # 找出要迭代多少次才能达到Tend
time = int(np.ceil(time))
count = 0   # 迭代记录器
Obj = np.zeros((time, 1))   # 目标值矩阵初始化
track = np.zeros((time, n))  # 每代最优的路径


# 随机产生一个初始路线
S1 = []
i = 0
while i < n:
    a = random.randint(0, n-1)
    if a not in S1:
        S1.append(a)
    else:
        i = i-1
    i += 1
# 计算目前路线的距离
RLength = 0
for i in range(n-1):
    RLength += D[S1[i], S1[i+1]]

while T0 > Tend:
    count += 1
    # temp = np.zeros(L, n+1)
    # 产生一个新的解（不过这里只有两个路径相互交换）
    a = random.randint(0, n-1)
    b = random.randint(0, n-1)
    while 1:
        if a != b:
            break
        else:
            b = random.randint(0, n-1)
    S2 = copy.deepcopy(S1)
    S2[a], S2[b] = S2[b], S2[a]
    # 判断是否接受新的解
    R = 0
    for i in range(n - 1):
        R += D[S2[i], S2[i + 1]]
    if R < RLength:
        RLength = R
        S1 = copy.deepcopy(S2)
    else:
        P = np.exp(-1*(R-RLength)/T0)
        rand = random.random()
        if P >= rand:
            RLength = R
            S1 = copy.deepcopy(S2)

    Obj[count-1] = RLength
    track[count-1, :] = S1
    T0 = q * T0


# plt.plot(range(len(Obj)), Obj)
# plt.show()

fig, ax = plt.subplots()
ax.plot(city[S1, 0], city[S1, 1], "r")
for i, txt in enumerate(S1):
    if i == 0:
        ax.annotate(("begin", txt + 1), (city[S1[i], 0], city[S1[i], 1]))
    elif i == n-1:
        ax.annotate(("end", txt + 1), (city[S1[i], 0], city[S1[i], 1]))
    else:
        ax.annotate((txt + 1), (city[S1[i], 0], city[S1[i], 1]))
plt.show()
