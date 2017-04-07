#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 23:16:12 2017

@author: ysbudakyan
'Sand' dataset generator
"""

import argparse
import numpy as np
from PIL import Image, ImageDraw
from numpy.random import choice, rand
from math import log
from tqdm import tqdm
from os import mkdir


def generate_sample(w, h, l, num, AA, val=False, side=1):
    # новое ч/б изображение, размером в AA**2 раз больше для суперсэмплинга
    W = w * AA
    H = h * AA
    R = r * AA
    # generate x and counts
    X = np.zeros(W)
    l_u = l(W)
    x = 0
    while(int(x) < W):
        u1 = rand()
        x -= log(u1) / l_u
        u2 = rand()
        if (u2 <= l(x) / l_u):
            try:
                X[int(x)] += 1
            except IndexError:
                break
    X = X.astype('int')
    image = Image.new('1', (W, H), 'white')
    draw = ImageDraw.Draw(image)
    # карта заполнения, чтобы предотвратить взаимопроникновение "песчинок"
    pixel_map = image.load()
    # цикл по x-ам
    for x in tqdm(range(W), desc='Side ' + str(side), leave=False):
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
        for j in range(X[x]):
            # генерируем y из доступных значений
            if (len(free_y) != 0):
                y = choice(list(free_y))
                draw.ellipse((x - R, y - R, x + R, y + R), fill='black')
                free_y -= set(range(y - R, y + R + 1))
            else:
                break
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

# parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('N', type=int)
parser.add_argument('trend_num', type=int)
parser.add_argument('-W', type=int, default=256)
parser.add_argument('-H', type=int, default=256)
parser.add_argument('-AA', type=int, default=3)
parser.add_argument('-l_start', type=float, default=0.1)
parser.add_argument('-l_end', type=float, default=2)
parser.add_argument('-r', type=int, default=3)
parser.add_argument('-ratio', type=float, defalut=0.95)
args = parser.parse_args()

# dataset images shape
W = args.W
H = args.H
# antialiasing coeff
AA = args.AA
# linear trend l = lambda = Kx + b
l_start = args.l_start
l_end = args.l_end
K = (l_end - l_start) / (W * AA)
# radius
r = args.r
# filename & path
file_name = "/sample"
ext = ".jpg"
trend_num = args.trend_num
file_path = "../data/sand/trend" + str(trend_num)

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


# number of samples to generate
N = args.N
# train/validation = 80/20
N_train = int(args.ratio * N)
N_val = N - N_train

for i in tqdm(range(N_train), desc='Train dataset'):
    generate_sample(W, H, lambda x: l_start, i, AA, val=False, side=1)
    generate_sample(W, H, lambda x: l_end, i, AA, val=False, side=2)
    generate_sample(W, H, lambda x: l_start + K * x, i, AA, val=False, side=3)

for i in tqdm(range(N_val), desc='Validation dataset'):
    generate_sample(W, H, lambda x: l_start, i, AA, val=True, side=1)
    generate_sample(W, H, lambda x: l_end, i, AA, val=True, side=2)
    generate_sample(W, H, lambda x: l_start + K * x, i, AA, val=True, side=3)
