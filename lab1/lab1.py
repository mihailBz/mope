import numpy as np

# визначаємо коефіцієнти А[0-3]
as_ = np.random.randint(100, size=4)
print(f'Рівняння регресії: Y = {as_[0]} + {as_[1]} * x1 + {as_[2]} * x2 + {as_[3]} * x3')

# генеруємо матрицю планування
xs = np.random.randint(21, size=(8, 3))
print('\nМатриця планування:\n', xs)

# визначаємо значення функції відгуків для кожної точки плану
ys = [as_[0] + as_[1] * x[0] + as_[2] * x[1] + as_[3] * x[2] for x in xs]
print('\nЗначення функції відгуків для кожної точки плану:')
for i in range(8):
    print(f'Y{i+1} = {ys[i]}')

x0s = []
dxs = []

for i in range(3):
    column = xs[:, i]
    x_max = max(column)
    x_min = min(column)
    x0 = (x_max+x_min)/2
    dx = x0 - x_min
    assert (x0 - x_min) == x_max - x0

    # визначаємо нульовий рівень
    x0s.append(x0)
    if i == 0:
        print('\nВизначаємо нульовий рівень')
    print(f'X0{i+1} = {x0}')
    dxs.append(dx)

    # виконуємо нормування факторів
    xns = [((x-x0)/dx) for x in xs]

print('\nВиконуємо нормування факторів')
for el in xns:
    print(el)

# визначеємо У еталонне
y_et = as_[0] + as_[1] * x0s[0] + as_[2] * x0s[1] + as_[3] * x0s[2]
print(f'\nВизначаємо еталонний Y\nYет = {y_et}')

# шукаємо точку плану, що задовольняє критерію вибору оптимальності
tmp_array = [y - y_et for y in ys]
min_el = min(filter(lambda x: x >= 0, tmp_array))
index = tmp_array.index(min_el)

print(f'\nТочка плану, що задовольняє заданий критерій оптимальності: {index + 1}')
for i in range(3):
    print(f'X{i+1} = {xs[index][i]}')



