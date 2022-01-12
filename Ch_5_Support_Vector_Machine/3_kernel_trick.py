# 2022 code by @metr0jw
# https://metr0jw.studio
# Not implemented yet

import numpy as np
import time


def compute(x_i, x_j, y_i, y_j):
    mat1 = np.dot(x_i, x_j)
    mat2 = np.dot(mat1, y_i)
    return np.dot(mat2, y_j)


def kernel_trick(x, y, trick=1):
    if trick == 1:
        x = x.T
    k = np.zeros([x.shape[0], x.shape[0]])
    for i in range(x.shape[0]):
        x_i = x[i, :]
        y_i = y[i, :]
        for j in range(x.shape[0]):
            x_j = x[j, :]
            y_j = y[j, :]
            k[i, j] = compute(x_i, x_j, y_i, y_j)
    return k


print("start")
x = np.random.rand(500, 500)
y = np.random.rand(500, 500)

start = time.time()
trick = kernel_trick(x, y)
print(time.time() - start)

start = time.time()
no_trick = kernel_trick(x, y, 0)
print(time.time() - start)
