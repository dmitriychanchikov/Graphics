import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# загрузка изображения
img = cv2.imread('example.jpg')

# преобразование в черно-белое изображение
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# отображение изображения thresh
cv2.imshow('thresh', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# определяем интервал между точками
interval = 1

# определяем координаты начальной точки
start_x = 0
start_y = 0

#
lim_x = 431  # from 22.02.2022 to 28.04.2023
lim_y = 3778.20

#
k_x = lim_x / img.shape[1]
k_y = lim_y / img.shape[0]

# создаем пустой список для хранения координат точек
points = []

# проходим по изображению слева направо с заданным интервалом и ищем черные точки
for x in range(start_x, img.shape[1], interval):
    if x < 14:
        points.append((k_x * x, k_y * 24))
        #print(x, 24)
        print(k_x * x, k_y * 24)
        continue
    for y in range(img.shape[0] - 1, start_y, -interval):
        if thresh[y, x] == 255:  # если нашли черную точку, то добавляем ее координаты в список
            points.append(( k_x * x, k_y * abs(y - img.shape[0] + 1) ))
            #print(x, abs(y - img.shape[0] + 1))
            print( k_x * x, k_y * abs(y - img.shape[0] + 1) )
            break  # переходим к следующей строке

print(img.shape[1])
print(img.shape[0])

# преобразуем список в массив numpy
points = np.array(points)

# построение графика
#plt.plot(points[:, 0], points[:, 1])

# Создание фигуры с названием
fig = plt.figure(num='Сравнение графиков')

# отображение изображения и графика
ax = fig.subplots(2, 1)
#ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax[0].imshow(img)
ax[0].set_title('Изображение')

ax[1].plot(points[:, 0], points[:, 1])
ax[1].set_title('График')
ax[1].set_aspect(0.032)
ax[1].set_xlim([0, 431])
ax[1].set_ylim([0, 3749])
ax[1].grid(True)

plt.show()

# создаем DataFrame из массива points
df = pd.DataFrame(data=points, columns=['x', 'y'])

# сохраняем DataFrame в файл Excel
df.to_excel('output.xlsx', index=False)
