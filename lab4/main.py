import numpy as np
from itertools import product, combinations
from _pydecimal import Decimal
from scipy.stats import f, t
from random import random

np.set_printoptions(formatter={'float_kind': lambda x: "%.2f" % x})

gt = {1: 0.6798, 2: 0.5157, 3: 0.4377, 4: 0.3910, 5: 0.3595, 6: 0.3362, 7:
    0.3185, 8: 0.3043, 9: 0.2926, 10: 0.2829}

tt = {16: 2.120, 24: 2.064, 32: 1.96}  # m = [3, 6]

ft = {1: {16: 4.5, 24: 4.3, 32: 4.2},
      2: {16: 3.6, 24: 3.4, 32: 3.3},
      3: {16: 3.2, 24: 3.0, 32: 2.9},
      4: {16: 3.0, 24: 2.8, 32: 2.7},
      5: {16: 2.9, 24: 2.6, 32: 2.5},
      6: {16: 2.7, 24: 2.5, 32: 2.4}}


def get_Gt(size_of_selections, qty_of_selections, significance):
    size_of_selections += 1
    partResult1 = significance / (size_of_selections - 1)
    params = [partResult1, qty_of_selections, (size_of_selections - 1 - 1) * qty_of_selections]
    fisher = f.isf(*params)
    result = fisher / (fisher + (size_of_selections - 1 - 1))
    return Decimal(result).quantize(Decimal('.0001')).__float__()


def get_tt(f3, significance):
    return Decimal(abs(t.ppf(significance / 2, f3))).quantize(Decimal('.0001')).__float__()


def get_Ft(f3, f4, significance):
    return Decimal(abs(f.isf(significance, f4, f3))).quantize(Decimal('.0001')).__float__()


def make_norm_plan_matrix(plan_matrix, matrix_of_min_and_max_x):
    X0 = np.mean(matrix_with_min_max_x, axis=1)
    interval_of_change = np.array([(matrix_of_min_and_max_x[i, 1] - X0[i])
                                   for i in range(len(plan_matrix[0]))])
    X_norm = np.array(
        [[round((plan_matrix[i, j] - X0[j]) / interval_of_change[j], 3) for j
          in range(len(plan_matrix[i]))]
         for i in range(len(plan_matrix))])
    return X_norm


def cochran_check(Y_matrix):
    mean_Y = np.mean(Y_matrix, axis=1)
    dispersion_Y = np.mean((Y_matrix.T - mean_Y) ** 2, axis=0)
    Gp = np.max(dispersion_Y) / (np.sum(dispersion_Y))
    return Gp < gt[m - 1]


def get_student_value_with_scipy(f_, q_):
    return Decimal(abs(t.ppf(q_ / 2, f_))).quantize(Decimal('.0001')).__float__()


def students_t_test(norm_matrix, Y_matrix):
    mean_Y = np.mean(Y_matrix, axis=1)
    dispersion_Y = np.mean((Y_matrix.T - mean_Y) ** 2, axis=0)
    mean_dispersion = np.mean(dispersion_Y)
    sigma = np.sqrt(mean_dispersion / (N * m))
    betta = np.mean(norm_matrix.T * mean_Y, axis=1)
    ts = np.abs(betta) / sigma

# Розрахунок критерію Студента за допомогою scipy
    f_ = (m-1)*N
    q_ = 0.05
    t_ = get_student_value_with_scipy(f_, q_)


    if (m - 1) * N > 32:
        return np.where(ts > t_)
    return np.where(ts > tt[(m - 1) * N])


def phisher_criterion(Y_matrix, d):
    if d == N:
        return False
    Sad = m / (N - d) * np.mean(check1 - mean_Y)
    mean_dispersion = np.mean(np.mean((Y_matrix.T - mean_Y) ** 2, axis=0))
    Fp = Sad / mean_dispersion
    if (m - 1) * N > 32:
        if N - d > 6:
            return Fp < ft[6][32]
        return Fp < ft[N - d][32]
    if N - d > 6:
        return Fp < ft[6][(m - 1) * N]
    return Fp < ft[N - d][(m - 1) * N]


matrix_with_min_max_x = np.array([[-20, 30], [30, 80], [30, 45]])
m = 6
N = 8
f1 = m - 1
f2 = N
f3 = f1 * f2
f4 = N - 2
q = 0.05
norm_matrix = np.array(list(product("01", repeat=3)), dtype=np.int)
norm_matrix[norm_matrix == 0] = -1
norm_matrix = np.insert(norm_matrix, 0, 1, axis=1)
plan_matrix = np.empty((8, 3))
Gt = get_Gt(f2, f1, q)
t_tab = get_tt(f3, q)
Ft = get_Ft(f3, f4, q)

for i in range(len(norm_matrix)):
    for j in range(1, len(norm_matrix[i])):
        if j == 1:
            if norm_matrix[i, j] == -1:
                plan_matrix[i, j - 1] = 10
            elif norm_matrix[i, j] == 1:
                plan_matrix[i, j - 1] = 60
        elif j == 2:
            if norm_matrix[i, j] == -1:
                plan_matrix[i, j - 1] = -30
            elif norm_matrix[i, j] == 1:
                plan_matrix[i, j - 1] = 45
        elif j == 3:
            if norm_matrix[i, j] == -1:
                plan_matrix[i, j - 1] = -30
            elif norm_matrix[i, j] == 1:
                plan_matrix[i, j - 1] = 45

plan_matr = np.insert(plan_matrix, 0, 1, axis=1)
x_min_mean, x_max_mean = np.mean(matrix_with_min_max_x, axis=0)

y_min = 200 + x_min_mean
y_max = 200 + x_max_mean
diff_y = y_max - y_min

Y_matrix = np.array([[y_min + random() * diff_y for _ in range(m)] for _ in range(N)])

mean_Y = np.mean(Y_matrix, axis=1)
combination = list(combinations(range(1, 4), 2))
for i in combination:
    plan_matr = np.append(plan_matr, np.reshape(plan_matr[:, i[0]] *
                                                plan_matr[:, i[1]], (8, 1)), axis=1)
    norm_matrix = np.append(norm_matrix, np.reshape(norm_matrix[:, i[0]] *
                                                    norm_matrix[:, i[1]], (8, 1)), axis=1)
plan_matr = np.append(plan_matr, np.reshape(plan_matr[:, 1] * plan_matr[:, 2]
                                            * plan_matr[:, 3], (8, 1)), axis=1)
norm_matrix = np.append(norm_matrix, np.reshape(norm_matrix[:, 1] * norm_matrix[:, 2] * norm_matrix[:, 3], (8, 1)),
                        axis=1)

if cochran_check(Y_matrix):
    b_natura = np.linalg.lstsq(plan_matr, mean_Y, rcond=None)[0]
    b_norm = np.linalg.lstsq(norm_matrix, mean_Y, rcond=None)[0]
    check1 = np.sum(b_natura * plan_matr, axis=1)
    check2 = np.sum(b_norm * norm_matrix, axis=1)
    indexes = students_t_test(norm_matrix, Y_matrix)

    print("\nМатриця плану експерименту: \n", plan_matr)
    print("\nНормована матриця: \n", norm_matrix)
    print("\nМатриця відгуків: \n", Y_matrix)
    print("\nGt:", Gt)
    print("tt:", t_tab)
    print("Ft:", Ft)
    print("Середні значення У: ", mean_Y)
    print("Натуралізовані коефіціенти: ", b_natura)
    print("Нормовані коефіціенти: ", b_norm)
    print("Перевірка 1: ", check1)
    print("Перевірка 2: ", check2)
    print("Індекси коефіціентів, які задовольняють критерію Стьюдента: ",
          np.array(indexes)[0])
    print("Критерій Стьюдента: ", np.sum(b_natura[indexes] *
                                         np.reshape(plan_matr[:, indexes], (N, np.size(indexes))), axis=1))
    if phisher_criterion(Y_matrix, np.size(indexes)):
        print("\nРівняння регресії адекватно оригіналу.")
    else:
        print("\nРівняння регресії неадекватно оригіналу.")
else:
    print("\n\nДисперсія неоднорідна!")
