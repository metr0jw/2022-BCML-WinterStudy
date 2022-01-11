import numpy as np
import time


def compute_RBF(mat1, mat2, sigma):
    trnorms1 = np.mat([(v * v.T)[0, 0] for v in mat1]).T
    trnorms2 = np.mat([(v * v.T)[0, 0] for v in mat2]).T

    k1 = trnorms1 * np.mat(np.ones((mat2.shape[0], 1), dtype=np.float64)).T
    k2 = np.mat(np.ones((mat1.shape[0], 1), dtype=np.float64)) * trnorms2.T

    k = k1 + k2
    k -= 2 * np.mat(mat1 * mat2.T)
    k *= - 1./(2 * np.power(sigma, 2))
    return np.exp(k)


x = np.random.rand(300, 300)
y = np.random.rand(300, 300)

start = time.time()
rbf_trick = compute_RBF(x, y, 0.001)
print(time.time() - start)

start = time.time()
without_trick = np.dot(x, y)
print(time.time() - start)
