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
from joblib import Parallel, delayed


def generate_sample(w, h, l, num, AA, val=False, side=1):
    np.random.seed()
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
    # x's loop
    # for x in tqdm(range(W), desc='Side ' + str(side), leave=False):
    for x in range(W):
        # находим недоступные на данный момент y
        banned_y = set()
        for y in range(H):
            ban_cond = 0
            for j in range(R + 1):
                # banned y's conditions
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
            # generate y from avalible values
            if (len(free_y) != 0):
                y = choice(list(free_y))
                draw.ellipse((x - R, y - R, x + R, y + R), fill='black')
                free_y -= set(range(y - R, y + R + 1))
            else:
                break
    # supersampling with antialiasing
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
parser.add_argument('-l_0', type=float, nargs=6,
                    default=None)
parser.add_argument('-l_1', type=float, nargs=6,
                    default=None)
parser.add_argument('-r', type=int, default=3)
parser.add_argument('-ratio', type=float, default=0.95)
parser.add_argument('-shift', type=int, default=0)
args = parser.parse_args()

# dataset images shape
W = args.W
H = args.H
# antialiasing coeff
AA = args.AA
# linear trend l = lambda = Kx + b
l_0 = args.l_0
l_1 = args.l_1
# print(l_0)
# print(l_1)
# radius
r = args.r
# filename & path
file_name = "/sample"
ext = ".png"
trend_num = args.trend_num
file_path = "../data/sand/trend" + str(trend_num)
shift = args.shift

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
N_train = int(args.ratio * N)
N_val = N - N_train

for i in tqdm(range(N_train), desc='Train dataset'):
    if l_0 is None:
        l_start = 30 * rand()
    else:
        l_start = choice(l_0)
    if l_1 is None:
        l_end = 30 * rand()
    else:
        l_end = choice(l_1)
    K = (l_end - l_start) / (W * AA)

    def l0(x):
        return l_start

    def l1(x):
        return l_end

    def l_trend(x):
        return l_start + K * x

    ag = zip((l0, l1, l_trend), (1, 2, 3))
    Parallel(n_jobs=-1)(delayed(generate_sample)(W, H, l, i+shift, AA, False,
                        side) for l, side in ag)

for i in tqdm(range(N_val), desc='Validation dataset'):
    if l_0 is None:
        l_start = 30 * rand()
    else:
        l_start = choice(l_0)
    if l_1 is None:
        l_end = 30 * rand()
    else:
        l_end = choice(l_1)
    K = (l_end - l_start) / (W * AA)

    def l0(x):
        return l_start

    def l1(x):
        return l_end

    def l_trend(x):
        return l_start + K * x

    ag = zip((l0, l1, l_trend), (1, 2, 3))
    Parallel(n_jobs=-1)(delayed(generate_sample)(W, H, l, i+shift, AA, True,
                        side) for l, side in ag)
