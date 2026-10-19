import matplotlib.pyplot as plt
import numpy as np


def compute_deltas(row_matrix, col_matrix, row_strategy, col_strategy):
    r = row_matrix @ col_strategy
    c = row_strategy @ col_matrix
    return np.array([np.max(r) - row_strategy @ r, np.max(c) - c @ col_strategy], dtype=np.float64)


def compute_nash_conv(row_matrix, col_matrix, row_strategy, col_strategy):
    return np.sum(compute_deltas(row_matrix, col_matrix, row_strategy, col_strategy))


def compute_exploitability(row_matrix, col_matrix, row_strategy, col_strategy):
    return compute_nash_conv(row_matrix, col_matrix, row_strategy, col_strategy) / 2


def fictitious_play(row_matrix, col_matrix, num_iters, naive):
    m, n = row_matrix.shape
    x = np.zeros(m)
    y = np.zeros(n)
    x[np.argmax(row_matrix @ np.full(n, 1 / n))] = 1
    y[np.argmax(np.full(m, 1 / m) @ col_matrix)] = 1
    lx, ly = x, y
    res = [(x.copy(), y.copy())]
    for t in range(1, num_iters):
        bx = np.zeros(m)
        by = np.zeros(n)
        bx[np.argmax(row_matrix @ (ly if naive else y))] = 1
        by[np.argmax((lx if naive else x) @ col_matrix)] = 1
        x = x + (bx - x) / (t + 1)
        y = y + (by - y) / (t + 1)
        lx, ly = bx, by
        res.append((x.copy(), y.copy()))
    return res


def plot_exploitability(row_matrix, col_matrix, strategies, label):
    e = [compute_exploitability(row_matrix, col_matrix, x, y) for x, y in strategies]
    plt.plot(e, label=label)
    plt.xlabel('iteration')
    plt.ylabel('exploitability')
    plt.legend()
    plt.show()
    return e
