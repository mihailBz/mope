from random import random
import numpy as np
np.set_printoptions(precision=5, suppress=True)

x1 = (15, 45)
x2 = (-25, 10)
x3 = (45, 50)
m = 3
N = 4

x_avg_min = (x1[0] + x2[0] + x3[0]) / 3
x_avg_max = (x1[1] + x2[1] + x3[1]) / 3

y_min = 200 + x_avg_min
y_max = 200 + x_avg_max
diff_y = y_max - y_min

ys = [[y_min + random() * diff_y for _ in range(4)] for _ in range(m)]

exp_matrix_norm = [
    [1, 1, 1, 1],
    [-1, -1, 1, 1],
    [-1, 1, -1, 1],
    [-1, 1, 1, -1],
]
exp_matrix_norm.extend(ys)
exp_matrix_norm = np.array(exp_matrix_norm)
exp_matrix_norm = exp_matrix_norm.T
print('Матриця планування експерименту з нормованими X:\n')
print(exp_matrix_norm)




exp_matrix = [
    [x1[0], x1[0], x1[1], x1[1]],
    [x2[0], x2[1], x2[0], x2[1]],
    [x3[0], x3[1], x3[1], x3[0]]
]
exp_matrix.extend(ys)
exp_matrix = np.array(exp_matrix)
exp_matrix = exp_matrix.T

print('\nМатриця планування експерименту:\n')
print(exp_matrix)

## test data
#
# exp_matrix = np.array([
#     [-25, 5, 15, 15, 18, 16],
#     [-25, 40, 25, 10, 19, 13],
#     [75, 5, 25, 11, 14, 12],
#     [75, 40, 15, 16, 19, 16]
# ])
#
# exp_matrix_norm = [
#     [1, -1, -1, -1, 15, 18, 16],
#     [1, -1, 1, 1, 10, 19, 13],
#     [1, 1, -1, 1, 11, 14, 12],
#     [1, 1, 1, -1, 16, 19, 16]
# ]
# exp_matrix_norm = np.array(exp_matrix_norm)

##


mean_ys = []
for row in exp_matrix:
    mean_ys.append(np.mean(row[3:]))

print('\n\nСередні значення функції відгуку:\n')
print(mean_ys)



mxs = np.mean(exp_matrix, axis=0)[:3]
my = np.mean(mean_ys)

a1_to_a3 = []

for i in range(3):
    res = 0
    for j in range(len(mean_ys)):
        res += mean_ys[j] * exp_matrix[j, i]
    a1_to_a3.append(res / 4)

a1, a2, a3 = a1_to_a3

a_ij = []
for i in range(3):
    column = exp_matrix[:, i]

    res = 0
    for item in column:
        res += item ** 2
    a_ij.append(res / 4)

a11, a22, a33 = a_ij

a12 = (exp_matrix[0, 0] * exp_matrix[0, 1] + exp_matrix[1, 0] * exp_matrix[1, 1] + exp_matrix[2, 0] * exp_matrix[2, 1] +
       exp_matrix[3, 0] * exp_matrix[3, 1]) / 4
a21 = a12

a13 = (exp_matrix[0, 0] * exp_matrix[0, 2] + exp_matrix[1, 0] * exp_matrix[1, 2] + exp_matrix[2, 0] * exp_matrix[2, 2] +
       exp_matrix[3, 0] * exp_matrix[3, 2]) / 4
a31 = a13

a23 = (exp_matrix[0, 1] * exp_matrix[0, 2] + exp_matrix[1, 1] * exp_matrix[1, 2] + exp_matrix[2, 1] * exp_matrix[2, 2] +
       exp_matrix[3, 1] * exp_matrix[3, 2]) / 4
a32 = a23

vec = [my, a1, a2, a3]

matrix = [
    [1, mxs[0], mxs[1], mxs[2]],
    [mxs[0], a11, a12, a13],
    [mxs[1], a12, a22, a32],
    [mxs[2], a13, a23, a33]
]

bs = np.linalg.solve(matrix, vec)
print('\n\nЗначення коефіцієнтів рівняння регресії:\n')
print(bs)

exp_ys = []

for row in exp_matrix:
    exp_ys.append(bs[0] + bs[1] * row[0] + bs[2] * row[1] + bs[3] * row[2])

print('\n\nПідставимо значення факторів з матриці планування'
      ' і порівняємо результат з середніми значеннями функції відгуку за рядками:\n')
print(exp_ys)


sigmas = []

tmp_i = 0
for row in exp_matrix_norm:
    sigmas.append(
        ((row[4] - mean_ys[tmp_i]) ** 2 + (row[5] - mean_ys[tmp_i]) ** 2 + (row[6] - mean_ys[tmp_i]) ** 2) / 3
    )
    tmp_i += 1

print('\n\nДисперсії по рядках:\n')
print(sigmas)

gp = max(sigmas) / sum(sigmas)
print(f'\n\nЗначення для критерію Кохрена gp = {gp}')

if gp > 0.7679:
    print('\nДисперсія неоднорідна')
else:
    print('\nДисперсія однорідна')


Sb = sum(sigmas) / len(sigmas)

s_sqrd_beta = Sb / (m * len(sigmas))

s_beta = s_sqrd_beta ** 0.5

betas = []

for i in range(4):
    res = 0
    for j in range(4):
        res += mean_ys[j] * exp_matrix_norm[j, i]
    betas.append(res / 4)

ts = [abs(beta) / s_beta for beta in betas]
important_bs = []

for t in ts:
    if t > 2.306:
        important_bs.append(ts.index(t))

print(f'\n\nКоефіцієнти з цими індексами {important_bs} залишаються в рівнянні регресії, інші приймаємо незначними згідно з критерієм Стьюдента')

exp_res_after_student = []

for row in exp_matrix:
    res = 0
    for index in important_bs:
        if index == 0:
            res += bs[index]
        else:
            res += bs[index]*row[index-1]
    exp_res_after_student.append(res)

print('\n\nВизначаємо значення функції після застосвання критерію Стьюдента')
print(exp_res_after_student)

S_ad = 0

for i in range(4):
    S_ad += (exp_res_after_student[i] - mean_ys[i])**2

S_ad *= (m/(N-2))

F_p = S_ad/s_sqrd_beta

print(f'\n\nЗначення критерію Фішера {F_p}')

if F_p > 4.5:
    print("\nРівняння регресії неадекватно оригіналу при рівні значимості 0.05")
else:
    print("\nРівняння регресії адекватно оригіналу при рівні значимості 0.05")




print()
