#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 23:21:26 2017

@author: ysbudakyan
"""


import numpy as np
import multiprocessing as mp
from PIL import Image, ImageDraw
from numpy.random import choice, rand
from math import log
from tqdm import tqdm
from time import sleep
from joblib import Parallel, delayed
#from generator-sand.py import generate_sample

r = 3
def generate_sample(w, h, l, num, AA, q, val=False, side=1, save=True):
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
    #for x in tqdm(range(W), desc='Side ' + str(side), leave=False):
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
        q.put(1)
    # supersampling with antialiasing
    image = image.resize((w, h))
    if (save is True):
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
    else:
        return image

for i in tqdm(range(3), desc='Main loop'):
    def l1(x):
        return 1

    def l2(x):
        return 2

    def l3(x):
        return 0.0001 * x + 1

    def pbar(q):
        bar = tqdm(total=256*3*3)
        for _ in iter(q.get, None):
            bar.update()
        
    manager = mp.Manager()
    q = manager.Queue()
    progressbar = mp.Process(target=pbar, args=(q,))
    progressbar.start()
    params = (1, 2, 3)
    ag = zip(params, (l1, l2, l3))
    res = Parallel(n_jobs=-1)(delayed(generate_sample)(256, 256, l, 1, 3, q,
                              False, side, False) for side, l in ag)
    q.put(None)
    progressbar.join()
