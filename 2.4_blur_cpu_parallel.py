from PIL import Image # Python Image Library
from matplotlib import pyplot as plt
import numpy as np

from time import time # Для того, чтобы засекать время на выполнение операций

from multiprocessing.dummy import Pool
from multiprocessing import cpu_count
from itertools import repeat

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

def makeKernel(blurVector):
    """
    Эта функция создаёт ядро фильтра размытия по направлению
    на основе заданного вектора по XY
    """
    # Находим модули компонент вектора, чтобы по ним
    # найти размеры матрицы ядра
    blurVectorAbs  = np.abs(blurVector)
    # Определяем размеры четверти ядра по модулям компонент 
    kernelX = int(np.ceil(blurVectorAbs[0]))
    kernelY = int(np.ceil(blurVectorAbs[1]))
    # Создаём соответствующую нулевую матрицу
    kernelQuad = np.zeros((kernelY, kernelX), dtype=np.float32)
    # Заполняем диагональ единицами, проходя в цикле по наиболее
    # длинной стороне, чтобы не пропустить ни один элемент
    kernelForm = 'wide' if kernelY < kernelX else 'tall'
    if kernelForm == 'wide':
        for i in range(kernelX):
            j = int(i / kernelX * kernelY)
            kernelQuad[j,i] = 1.0
    elif kernelForm == 'tall':
        for j in range(kernelY):
            i = int(j / kernelY * kernelX)
            kernelQuad[j,i] = 1.0
    # Следующие две строки визуализируют построенную четверть ядра
    #plt.imshow(kernelQuad, cmap='Greys_r')
    #plt.show()
    # Создаём нулевую матрицу ядра
    kernel = np.zeros((2*kernelY+1, 2*kernelX+1), dtype=np.float32)
    # Записываем левую верхнюю и правую нижнюю четверти,
    # а также центральный пиксель
    kernel[0:kernelY,  0:kernelX]  = kernelQuad
    kernel[kernelY,    kernelX]    = 1.0
    kernel[kernelY+1:, kernelX+1:] = kernelQuad
    # Находим знаки компонент вектора, чтобы по ним
    # правильно отразить матрицу ядра
    blurVectorSign = np.sign(blurVector)
    # Отражаем матрицу так, чтобы её ненулевая диагональ совпала
    # по направлению с исходным вектором
    if blurVectorSign[0] == 1:    # нулевая компонента вектора - Х
        kernel = np.flip(kernel, axis=1)  # первая ось матрицы - Х
    # Поскольку строки матрицы идут сверху вниз, а Y – снизу вверх,
    # отражаем, когда Y компонента имеет знак +
    if blurVectorSign[1] == -1:    # первая компонента вектора - Y
        kernel = np.flip(kernel, axis=0) # нулевая ось матрицы - Y
    # Следующие две строки визуализируют построенное ядро
    #plt.imshow(kernel, cmap='Greys_r')
    #plt.show()
    return kernelX, kernelY, kernel

def motionBlurCoreCPU(idx, image, kernelX, kernelY, kernel):
    """
    Эта функция фильтрует один пиксель изображения image, имеющий одномерный индекс idx,
    делая усреднение по всем пикселям, которые соответствуют ненулевым элементам ядра kernel
    """
    # Восстанавливаем двумерный индекс пикселя
    i = idx %  image.shape[1]
    j = idx // image.shape[1]
    # Вычисляем границы фрагмента изображения вокруг пикселя i,j
    l = np.clip(i-kernelX,   0, image.shape[1]-1)
    r = np.clip(i+kernelX+1, 0, image.shape[1])
    t = np.clip(j-kernelY,   0, image.shape[0]-1)
    b = np.clip(j+kernelY+1, 0, image.shape[0])
    # Вычисляем границы соответствующего фрагмента ядра
    kl = max(kernelX-i, 0)
    kr = min(r-i+kernelX, 2*kernelX+1)
    kt = max(kernelY-j, 0)
    kb = min(b-j+kernelY, 2*kernelY+1)
    # Вырезаем фрагмент изображения по найденным границам
    imgFrag = image[t:b,l:r]
    # Вырезаем фрагмент ядра по найденным границам
    kerFrag = kernel[kt:kb,kl:kr]
    # Векторно считаем количество ненулевых элементов этого фрагмента ядра
    kerNorm = np.count_nonzero(kerFrag)
    # Векторно усредняем пиксели, соответствующие ненулевым элементам
    rBlurred = np.sum(imgFrag[...,0], where=(kerFrag > 0)) / kerNorm
    gBlurred = np.sum(imgFrag[...,1], where=(kerFrag > 0)) / kerNorm
    bBlurred = np.sum(imgFrag[...,2], where=(kerFrag > 0)) / kerNorm
    pixelBlurred = np.array([rBlurred, gBlurred, bBlurred]).astype(np.uint8)    
    return pixelBlurred

# Задаём вектор тонирующих коэффициентов
toneVector = np.array([1.1, 0.6, 0.7])

# Задаём вектор размытия по XY
blurVector = np.array([-10.5, 25.1])

# Запускаем процесс тонирования
tonedImage = toneImg(imgRGB, toneVector)

# Создаём ядро по вектору размытия
kernelX, kernelY, kernel = makeKernel(blurVector)

# Запускаем процесс размытия, с распараллеливанием на CPU
pool = Pool(processes=cpu_count()) # cpu_count() – число логических ядер
pixelCount = tonedImage.shape[0] * tonedImage.shape[1]
# Все аргументы нужно повторить столько раз, на сколько распараллеливаем
args = zip(range(pixelCount), repeat(tonedImage), repeat(kernelX), repeat(kernelY), repeat(kernel))
# Запускаем параллельные вычисления
start = time()
print('Идёт размытие изображения...')
blurredImageList = pool.starmap(motionBlurCoreCPU, args)
blurredImage = np.array(blurredImageList).reshape(tonedImage.shape)
print('Время выполнения:', time()-start, 'c')

plt.imshow(blurredImage)
plt.show()