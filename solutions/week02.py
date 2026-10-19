import itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog


def plot_best_response_value_function(row_matrix, step_size):
    xs = np.arange(0, 1 + step_size / 2, step_size)
    vals = [np.min(np.array([x, 1 - x]) @ row_matrix) for x in xs]
    plt.plot(xs, vals)
    plt.xlabel('probability of the first action')
    plt.ylabel('best response value')
    plt.show()


def verify_support(matrix, row_support, col_support):
    n = len(col_support)
    M = matrix[np.ix_(row_support, col_support)]
    A_eq = np.zeros((len(row_support) + 1, n + 1))
    A_eq[:-1, :n] = M
    A_eq[:-1, n] = -1
    A_eq[-1, :n] = 1
    b_eq = np.zeros(len(row_support) + 1)
    b_eq[-1] = 1
    res = linprog(np.zeros(n + 1), A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * n + [(None, None)])
    if not res.success:
        return None
    return res.x[:n].astype(np.float64)


def support_enumeration(row_matrix, col_matrix):
    m, n = row_matrix.shape
    eqs = []
    for k in range(1, min(m, n) + 1):
        for rs in itertools.combinations(range(m), k):
            for cs in itertools.combinations(range(n), k):
                rs, cs = np.array(rs), np.array(cs)
                y = verify_support(row_matrix, rs, cs)
                x = verify_support(col_matrix.T, cs, rs)
                if x is None or y is None:
                    continue
                row = np.zeros(m)
                col = np.zeros(n)
                row[rs] = x
                col[cs] = y
                if np.max(row_matrix @ col) > row @ row_matrix @ col + 1e-9:
                    continue
                if np.max(row @ col_matrix) > row @ col_matrix @ col + 1e-9:
                    continue
                if any(np.allclose(row, a) and np.allclose(col, b) for a, b in eqs):
                    continue
                eqs.append((row, col))
    return eqs
