# 2022 code by @metr0jw
# https://metr0jw.studio

import numpy as np
import time
import psutil


def memory_usage(message: str = 'debug'):
    # current process RAM usage
    p = psutil.Process()
    rss = p.memory_info().rss / 2 ** 20  # Bytes to MB
    print(f"[{message}] memory usage: {rss: 10.5f} MB")


def build_kernel(option):
    global x, y

    x = x.T
    K = np.zeros([x.shape[0], x.shape[0]])
    for i in range(x.shape[0]):
        xi = x[i, :]
        yi = y[i, :]
        for j in range(x.shape[0]):
            xj = x[j, :]
            yj = y[j, :]
            K[i, j] = radial_basis(xi, xj, yi, yj, option)

    return K


def radial_basis(xi, xj, yi, yj, gamma):
    r = (np.exp(-gamma*(np.linalg.norm(xi-xj)**2)))
    return r


def with_trick():
    start = time.time()
    result = build_kernel(0.001)
    print("With Kernel Trick:" + str(time.time() - start))
    print(memory_usage('After compute(Kernel Trick)'))


def without_trick():
    global x, y

    start = time.time()
    np.inner(x, y)
    print("Without Kernel Trick:" + str(time.time() - start))
    print(memory_usage('After compute'))


x = np.random.rand(300, 300)
y = np.random.rand(300, 300)

print(memory_usage('Before compute(Kernel Trick)'))

with_trick()
without_trick()
