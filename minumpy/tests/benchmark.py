import time

import minumpy as np


def benchmark():
    N = 512
    a = np.randint(shape=(N, N))
    b = np.randint(shape=(N, N))

    start = time.time()
    c = np.dot(a, b)
    end = time.time()

    print("elapsed time", end - start, np.sum(np.sum(c, 1), 0))


if __name__ == "__main__":
    benchmark()
