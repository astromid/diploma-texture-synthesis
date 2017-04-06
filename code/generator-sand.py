#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 23:16:12 2017

@author: ysbudakyan
'Sand' dataset generator
"""

from PIL import Image, ImageDraw
from numpy.random import randint, poisson, choice
from tqdm import tqdm
from os import mkdir


# dataset images shape
W = 256
H = 256
# antialiasing coeff
AA = 3
# linear trend l = lambda = Kx + b
l_start = 0.2
l_end = 5
K = (l_end - l_start) / (W * AA)
# "used x" %
B = 0.3
# radius
r = 3
# filename & path
file_name = "/sample"
ext = ".jpg"
file_path = "../data/sand/trend3"

# making directory structure
try:
    mkdir(file_path)
    mkdir(file_path + '/train')
    mkdir(file_path + '/train/panorama')
    mkdir(file_path + '/train/side1')
    mkdir(file_path + '/train/side2')
    mkdir(file_path + '/validation')
    mkdir(file_path + '/validation/panorama')
    mkdir(file_path + '/validation/side1')
    mkdir(file_path + '/validation/side2')
except FileExistsError:
    print('Directories already exist')


def generateSample(w, h, l, B, num, AA, val=False, side=1):
    # новое ч/б изображение, размером в AA**2 раз больше для суперсэмплинга
    W = w * AA
    H = h * AA
    R = r * AA
    image = Image.new('1', (W, H), 'white')
    draw = ImageDraw.Draw(image)
    # карта заполнения, чтобы предотвратить взаимопроникновение "песчинок"
    pixel_map = image.load()
    # цикл по x-ам
    for k in range(int(B * W)):
        x = randint(1, W)
        # число событий пуассоновского потока на данной вертикали
        Nx = poisson(l(x))
        # находим недоступные на данный момент y
        banned_y = set()
        for y in range(H):
            ban_cond = 0
            for j in range(R + 1):
                # условия недоступности y
                if (x - j > 0):
                    c_left = pixel_map[x - j, y]
                else:
                    c_left = True
                if (x + j < W - 1):
                    c_right = pixel_map[x + j, y]
                else:
                    c_right = True
                if (y - j > 0):
                    c_top = pixel_map[x, y - j]
                else:
                    c_top = True
                if (y + j < H - 1):
                    c_bot = pixel_map[x, y + j]
                else:
                    c_bot = True
                free_cond = ban_cond or c_left or c_right or c_top or c_bot
            if(not(free_cond)):
                banned_y.add(y)
        # заполняем вертикаль
        free_y = set(range(W)) - banned_y
        for j in range(Nx):
            # генерируем y из доступных значений
            y = choice(list(free_y))
            # little 'shake'
            shake = [x for x in range(-int(R / 2), int(R / 2 + 1))]
            shake = choice(shake)
            x_sh = x + shake
            draw.ellipse((x_sh - R, y - R, x_sh + R, y + R), fill='black')
            free_y -= set(range(y - R, y + R + 1))
    # суперсэмплинг с антиалиасингом
    image = image.resize((w, h))
    full_file_path = file_path
    if (val is False):
        full_file_path += "/train"
    else:
        full_file_path += "/validation"
    if (side == 1):
        full_file_path += "/side1"
    elif (side == 2):
        full_file_path += "/side2"
    else:
        full_file_path += "/panorama"
    full_file_path += (file_name + str(num) + ext)
    image.save(full_file_path)

# number of samples to generate
N = 25
# train/validation = 80/20
N_train = int(0.8 * N)
N_val = N - N_train
for i in tqdm(range(N_train), desc='Train'):
    generateSample(W, H, lambda x: l_start, B, i, AA, val=False, side=1)
    generateSample(W, H, lambda x: l_end, B, i, AA, val=False, side=2)
    generateSample(W, H, lambda x: l_start + K*x, B, i, AA, val=False, side=3)

for i in tqdm(range(N_val), desc='Validation'):
    generateSample(W, H, lambda x: l_start, B, i, AA, val=True, side=1)
    generateSample(W, H, lambda x: l_end, B, i, AA, val=True, side=2)
    generateSample(W, H, lambda x: l_start + K * x, B, i, AA, val=True, side=3)
