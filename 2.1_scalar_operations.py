from PIL import Image # Python Image Library
from matplotlib import pyplot as plt
import numpy as np

from time import time # Для того, чтобы засекать время на выполнение операций

imgRGB = np.array(Image.open('cat.jpg'))
print('Размеры изображения:', imgRGB.shape)

plt.imshow(imgRGB)
plt.show()

def toneImgByPixel(image, rgbTone):
    """
    Эта функция тонирует RGB изображение, умножая компоненту каждого пикселя 
    на соответствующий коэффициент из rgbTone
    """
    print('Идёт тонирование изображения...')
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Читаем компоненты цвета пикселя исходного изображения
            r = image[y][x][0]
            g = image[y][x][1]
            b = image[y][x][2]
            # Умножаем их на коэффициенты тонирования
            rTonedFloat = r * rgbTone[0]
            gTonedFloat = g * rgbTone[1]
            bTonedFloat = b * rgbTone[2]
            # Поскольку коэффициенты имеют тип float,
            # получившиеся значения тоже оказываются float.
            # Конвертируем их в 8-битный integer (0-255)
            rToned256 = np.clip(rTonedFloat, 0, 255).astype(np.uint8)
            gToned256 = np.clip(gTonedFloat, 0, 255).astype(np.uint8)
            bToned256 = np.clip(bTonedFloat, 0, 255).astype(np.uint8)
            # Записываем компоненты обратно
            image[y][x][0] = rToned256
            image[y][x][1] = gToned256
            image[y][x][2] = bToned256
    return image

# Задаём вектор тонирующих коэффициентов
toneVector = np.array([1.1, 0.6, 0.7])

# Запускаем процесс тонирования
start = time()
tonedImage = toneImgByPixel(imgRGB, toneVector)
print('Время выполнения:', time()-start, 'c')

plt.imshow(tonedImage)
plt.show()