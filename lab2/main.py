import numpy as np
from random import random
import pandas as pd

VAR_NUMBER = 104

x1_min = (-1, 15)
x1_max = (1, 45)
x2_min = (-1, -25)
x2_max = (1, 10)

R_kr = {
    6: 2.0,
    8: 2.17,
    10: 2.29,
    12: 2.39,
    15: 2.49,
    20: 2.62
}


def main():
    xs = [
        (x1_min[0], x2_min[0]),
        (x1_max[0], x2_min[0]),
        (x1_min[0], x2_max[0]),
    ]

    y_max = (30 - VAR_NUMBER) * 10
    y_min = (20 - VAR_NUMBER) * 10

    m = 5
    ys = [[y_min + random() * 100 for _ in range(m)] for i in range(3)]

    flag = check_by_romanovsky_kr(xs, ys)
    while not flag:
        m += 1
        ys = [[y_min + random() * 100 for _ in range(m)] for i in range(3)]
        flag = check_by_romanovsky_kr(xs, ys, m)

    df = make_df(xs, ys, m)
    ys_avg = [(sum(values) / len(values)) for values in ys]

    mx1 = df['X1'].mean()
    mx2 = df['X2'].mean()
    my = sum(ys_avg)/len(ys_avg)

    a1 = sum(list(map(lambda x: x**2, df['X1'])))/len(df['X1'])
    a2 = (xs[0][0]*xs[0][1]+xs[1][0]*xs[1][1] + xs[2][0]*xs[2][1])/3
    a3 = sum(list(map(lambda x: x**2, df['X2'])))/len(df['X2'])

    a11 = 0
    a22 = 0
    for i in range(3):
        a11 += xs[i][0]*ys_avg[i]
        a22 += xs[i][1]*ys_avg[i]
    a11 /= 3
    a22 /=3

    matrix1 = np.array([
        [1, mx1, mx2],
        [mx1, a1, a2],
        [mx2, a2, a3]
    ])
    v = np.array([my, a11, a22])
    bs = np.linalg.solve(matrix1, v)

    print('\nКоефіцієнти b')
    print(bs)

    test = []
    for i in range(3):
        test.append(bs[0] + xs[i][0]*bs[1] + xs[i][1]*bs[2])

    print('\nПеревірка')
    print(test)

    dx1 = abs(x1_max[1]-x1_min[1])/2
    dx2 = abs(x2_max[1]-x2_min[1])/2
    x10 = (x1_max[1] + x1_min[1])/2
    x20 = (x2_max[1]+x2_min[1])/2

    a_0 = bs[0]-bs[1]*(x10/dx1)-bs[2]*(x20/dx2)
    a_1 = bs[1]/dx1
    a_2 = bs[2]/dx2

    print('\nПеревірка натуралізованого рівняння регресії')
    res = []
    res.append(
        a_0+a_1*x1_min[1]+a_2*x2_min[1]
    )
    res.append(
        a_0 + a_1*x1_max[1] + a_2*x2_min[1]
    )
    res.append(
        a_0 + a_1*x1_min[1] + a_2*x2_max[1]
    )
    print(res)




def check_by_romanovsky_kr(xs, ys, m=5):


    print('\nМатриця експерименту')
    df = make_df(xs, ys, m)
    print(df)
    print('\nЗнаходимо середні значення та дисперсію')

    ys_avg = [(sum(values) / len(values)) for values in ys]
    disp = []
    for i in range(3):
        disp.append(np.var(ys[i]))
    #     tmp_value = [((y - ys_avg[i]) ** 2) for y in ys[i]]
    #     disp.append(sum(tmp_value) / len(tmp_value))

    df['Y середнє'] = ys_avg
    df['дисперсія'] = disp
    print(df)

    sigma_teta = ((2 * (2 * m - 2)) / (m * (m - 4))) ** (1 / 2)

    f_uv_s = [
        disp[0] / disp[1],
        disp[2] / disp[0],
        disp[2] / disp[1],
    ]

    teta_uv_s = [(((m - 2) / m) * f_uv) for f_uv in f_uv_s]

    r_uv_s = [((abs(teta_uv - 1)) / sigma_teta) for teta_uv in teta_uv_s]

    print('\nЕкспериментальне значення критерію Романовського')
    df['R uv'] = r_uv_s
    print(df)

    if m <= 20:
        tmp_m = m
        while tmp_m not in R_kr:
            tmp_m += 1
    else:
        tmp_m = 20

    for r_uv in r_uv_s:
        if r_uv > R_kr[tmp_m]:
            print('Гипотеза про однорідність дисперсій не підтвердилася. Збільшуємо кількість дослідів')
            return False
    return True



def make_df(xs, ys, m):
    matrix = []
    for i in range(3):
        matrix.append([*xs[i], *ys[i]])

    df = pd.DataFrame(
        matrix,
        columns=['X1', 'X2'] + list(['Y' + str(i) for i in range(1, m + 1)])
    )
    return df

if __name__ == '__main__':
    main()
