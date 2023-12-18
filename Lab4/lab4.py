# Вступ до комп’ютерного зору
# Лабораторна робота No4
# КПI iм. Iгоря Сiкорського
# Листопад 2023

# 1 Видалення шуму
# 1.1 Мета
# Метою цiєї лабораторної роботи є отримання навичок роботи з методами
# видалення шуму з зображень.
# 1.2 Завдання
# Виконати наступнi кроки:
# 1. Обрати декiлька (3+) зображень, без вираженого шуму.
# Рис. 1: Приклад зображення.
# 2. Для кожного зображення додати наступнi типи шуму:
# • Salt-Pepper (Iмпульсний шум)
# • Gaussian noise (Нормальний шум)
# 3. Для кожного зображення застосувати наступнi методи видалення шу-
# му:
# • Box average (згладжування коробкою :D)
# • Медiана
# • Зважена медiана
# 4. Порiвняти результати, та зробити висновки.
# 5. (Опцiонально, +5 балiв) Виконати фiльтрацiю у HSV представленi, та
# порiвняти результат з фiльтрацiєю в RGB.
# 1.3 Вимоги до виконання

# • Дозволяється використання стороннiх модулiв лише для генерацiї шу-
# му (Також додам .py файл готовими методами генерацiї шуму).

# • Використовуйте вiртуальнi середовища (venv).
# • Необхiднi для вашого коду модулi зберiгайте у файл “requirements.txt”,
# та додавайте його на github разом с кодом.
# • Код, оригiнальнi фото та отриманi результати залити на свiй github

# • У classroom додати тiльки код та результати (для звiтностi), та поси-
# лання на github.

from PIL import Image
import numpy as np
import random
import os

def sp_noise_gray(image, prob=0.03):
    '''
    Add salt and pepper noise to a gray image [0,255]
    
    image: Numpy 2D array
    prob: Probability of the noise
    returns: Numpy 2D array
    '''    
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                image[i,j] = 0
            elif rdn > thres:
                image[i,j] = 255
    return image

def sp_noise_color(image, prob=0.03, white=[255,255,255], black=[0,0,0]):
    '''
    Add salt and pepper noise to a color image
    
    image: Numpy 2D array
    prob: Probability of the noise
    returns: Numpy 2D array
    '''    
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                image[i,j,:] = black
            elif rdn > thres:
                image[i,j,:] = white
    return image

def norm_noise_gray(image, mean=0, var=0.1, a=0.5):
    '''
    Add gaussian noise to gray image 
    
    image: Numpy 2D array
    mean: scalar
    vat: scalar
    returns: Numpy 2D array
    '''    
    sigma = var**0.5
    
    row,col= image.shape[:2]
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = a*image + (1-a)*gauss

    noisy = noisy-np.min(noisy)
    noisy = 255*(noisy/np.max(noisy))
    
    return noisy.astype(np.uint8)

def norm_noise_color(image, mean=0, var=0.1, a=0.5):
    '''
    Add gaussian noise to color image 
    
    image: Numpy 2D array
    mean: scalar - mean
    var: scalar - variance
    a: scalar [0-1] - alpha blend
    returns: Numpy 2D array
    '''    
    sigma = var**0.5
    
    row,col,ch= image.shape[:3]
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = a*image + (1-a)*gauss

    noisy = noisy-np.min(noisy)
    noisy = 255*(noisy/np.max(noisy))
    
    return noisy.astype(np.uint8)

try:
    os.listdir('photos\\')
    if not os.path.exists('results\\'):
        os.mkdir('results\\')
except FileNotFoundError:
    os.mkdir('photos\\')
    if not os.path.exists('results\\'):
        os.mkdir('results\\')
        
for item in os.listdir('photos\\'):
    image = np.array(Image.open('photos\\'+item))#, 0)
    # print(image.shape)
    if len(image.shape) == 3:
        sp_noise_img = sp_noise_color(image,0.07)
        norm_noise_img = norm_noise_color(image, mean=0, var=10, a=0.1)
    else:
        sp_noise_img = sp_noise_gray(image,0.07)
        norm_noise_img = norm_noise_gray(image, mean=0, var=10, a=0.1)
    sp_img = Image.fromarray(sp_noise_img)
    norm_img = Image.fromarray(norm_noise_img.astype(np.uint8))
    # sp_img.show()
    # norm_img.show()
    sp_img.save('results\\res_sp_'+item)
    norm_img.save('results\\res_norm_'+item)