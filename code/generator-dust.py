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
from numpy.random import choice, uniform
from math import log
from tqdm import tqdm
from os import mkdir
from joblib import Parallel, delayed


def generate_sample(w, h, l, num, val=False, side=1):
    np.random.seed()
    # новое ч/б изображение, размером в AA**2 раз больше для суперсэмплинга
    # generate x and counts
    X = np.zeros(w)
    l_u = l(w)
    x = 0
    while(int(x) < w):
        u1 = uniform()
        x -= log(u1) / l_u
        u2 = uniform()
        if (u2 <= l(x) / l_u):
            try:
                X[int(x)] += 1
            except IndexError:
                break
    X = X.astype('int')
    image = Image.new('1', (w, h), 'white')
    draw = ImageDraw.Draw(image)
    for x in range(w):
        for j in range(X[x]):
            y = choice(range(w))
            draw.point((x, y), fill='black')
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
parser.add_argument('-l_0', type=float, nargs=6,
                    default=None)
parser.add_argument('-l_1', type=float, nargs=6,
                    default=None)
parser.add_argument('-ratio', type=float, default=0.95)
parser.add_argument('-shift', type=int, default=0)
args = parser.parse_args()

# dataset images shape
W = args.W
H = args.H
# linear trend l = lambda = Kx + b
l_0 = args.l_0
l_1 = args.l_1
# print(l_0)
# print(l_1)
# filename & path
file_name = "/sample"
ext = ".png"
trend_num = args.trend_num
file_path = "../data/dust/trend" + str(trend_num)
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
        l_start = 35 * uniform()
    else:
        l_start = choice(l_0)
    if l_1 is None:
        l_end = 35 * uniform()
    else:
        l_end = choice(l_1)
    K = (l_end - l_start) / W

    def l0(x):
        return l_start

    def l1(x):
        return l_end

    def l_trend(x):
        return l_start + K * x

    ag = zip((l0, l1, l_trend), (1, 2, 3))
    Parallel(n_jobs=-1)(delayed(generate_sample)(W, H, l, i+shift, False,
                        side) for l, side in ag)

for i in tqdm(range(N_val), desc='Validation dataset'):
    if l_0 is None:
        l_start = 35 * uniform()
    else:
        l_start = choice(l_0)
    if l_1 is None:
        l_end = 35 * uniform()
    else:
        l_end = choice(l_1)
    K = (l_end - l_start) / W

    def l0(x):
        return l_start

    def l1(x):
        return l_end

    def l_trend(x):
        return l_start + K * x

    ag = zip((l0, l1, l_trend), (1, 2, 3))
    Parallel(n_jobs=-1)(delayed(generate_sample)(W, H, l, i+shift, True,
                        side) for l, side in ag)
