from PIL import Image # Python Image Library
from matplotlib import pyplot as plt
import numpy as np

from time import time # Для того, чтобы засекать время на выполнение операций

imgRGB = np.array(Image.open('cat.jpg'))
print('Размеры изображения:', imgRGB.shape)

plt.imshow(imgRGB)
plt.show()

def toneImg(image, rgbTone):
    """
    Эта функция тонирует RGB изображение, умножая весь массив
    на вектор коэффициентов rgbTone
    """
    return np.clip(image * rgbTone, 0, 255).astype(np.uint8)

# Задаём вектор тонирующих коэффициентов
toneVector = np.array([1.1, 0.6, 0.7])

# Запускаем процесс тонирования
start = time()
tonedImage = toneImg(imgRGB, toneVector)
print('Время выполнения:', time()-start, 'c')

plt.imshow(tonedImage)
plt.show()