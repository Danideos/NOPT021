import numpy as np


def evaluate_general_sum(row_matrix, col_matrix, row_strategy, col_strategy):
    row_value = row_strategy @ row_matrix @ col_strategy
    col_value = row_strategy @ col_matrix @ col_strategy
    return np.array([row_value, col_value])


def evaluate_zero_sum(row_matrix, row_strategy, col_strategy):
    return evaluate_general_sum(row_matrix, -row_matrix, row_strategy, col_strategy)


def calculate_best_response_against_row(col_matrix, row_strategy):
    values = row_strategy @ col_matrix
    br = np.zeros(len(values))
    br[np.argmax(values)] = 1
    return br


def calculate_best_response_against_col(row_matrix, col_strategy):
    values = row_matrix @ col_strategy
    br = np.zeros(len(values))
    br[np.argmax(values)] = 1
    return br


def evaluate_row_against_best_response(row_matrix, col_matrix, row_strategy):
    col_strategy = calculate_best_response_against_row(col_matrix, row_strategy)
    return row_strategy @ row_matrix @ col_strategy


def evaluate_col_against_best_response(row_matrix, col_matrix, col_strategy):
    row_strategy = calculate_best_response_against_col(row_matrix, col_strategy)
    return row_strategy @ col_matrix @ col_strategy


def find_strictly_dominated_actions(matrix):
    dominated = []
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if np.all(matrix[j] > matrix[i]):
                dominated.append(i)
                break
    return np.array(dominated, dtype=np.int64)


def iterated_removal_of_dominated_strategies(row_matrix, col_matrix):
    rows = np.arange(row_matrix.shape[0])
    cols = np.arange(row_matrix.shape[1])

    while True:
        bad_rows = find_strictly_dominated_actions(row_matrix)
        bad_cols = find_strictly_dominated_actions(col_matrix.T)
        if len(bad_rows) == 0 and len(bad_cols) == 0:
            break

        row_matrix = np.delete(row_matrix, bad_rows, axis=0)
        col_matrix = np.delete(col_matrix, bad_rows, axis=0)
        rows = np.delete(rows, bad_rows)

        row_matrix = np.delete(row_matrix, bad_cols, axis=1)
        col_matrix = np.delete(col_matrix, bad_cols, axis=1)
        cols = np.delete(cols, bad_cols)

    return row_matrix, col_matrix, rows, cols
