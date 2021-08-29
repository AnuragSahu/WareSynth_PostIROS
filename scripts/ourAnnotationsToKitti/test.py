import numpy as np

K = [[5, 0, 2],
     [0, 3, 2],
     [0, 0, 1]]

R = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]

T = [0, 0, 0]
T = [[i] for i in T]

RT = np.hstack((R,T))
P = np.dot(K,RT)

point = [1,1,1,1]
point = [[i] for i in point]
res = np.dot(P, point)
res[0] = res[0]/res[-1]
res[1] = res[1]/res[-1]

print(res)
