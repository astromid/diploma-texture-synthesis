#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 22:45:13 2017

@author: ysbudakyan
Various utility functions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image, ImageDraw
from numpy.random import choice, rand
from math import log
from os import listdir, mkdir
from keras.utils import plot_model
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from joblib import Parallel, delayed


def generate_sample(w, h, l, num, AA, r):
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
    return image


# load dataset, returns keras.ImageDataGenerators over train and validation
def load_dataset(dataset_path, trend_num, mode=3, W=256, H=256):
    # mode: 1 - 1 image mode (only panorama generators)
    #       3 - 3 image mode (side1, side2, panorama)
    # list of filenames to load
    dataset_path_with_trend = dataset_path + '/trend' + str(trend_num)
    train_path = dataset_path_with_trend + '/train'
    validation_path = dataset_path_with_trend + '/validation'
    train_list = listdir(train_path + '/panorama')
    val_list = listdir(validation_path + '/panorama')
    N_train = len(train_list)
    N_val = len(val_list)

    # 1 image mode
    if (mode == 1):
        # load and normalization train images
        panorama_train = np.empty((N_train, W, H, 1))
        for i, file in enumerate(tqdm(train_list, desc='Train dataset')):
            image = Image.open(train_path + '/panorama/' + file)
            panorama_train[i] = (np.array(image).reshape(W, H, 1)) - 127.5
            panorama_train[i] /= 127.5

        # load and normalization validation images
        panorama_val = np.empty((N_val, W, H, 1))
        for i, file in enumerate(tqdm(val_list, desc='Validation dataset')):
            image = Image.open(validation_path + '/panorama/' + file)
            panorama_val[i] = (np.array(image).reshape(W, H, 1)) - 127.5
            panorama_val[i] /= 127.5

        return (panorama_train, panorama_val)

    # 3 image mode
    if (mode == 3):
        side1_train = np.empty((N_train, W, H, 1))
        side2_train = np.empty((N_train, W, H, 1))
        panorama_train = np.empty((N_train, W, H, 1))

        for i, file in enumerate(tqdm(train_list, desc='Train dataset')):
            image = Image.open(train_path + '/side1/' + file)
            side1_train[i] = (np.array(image).reshape(W, H, 1)) - 127.5
            side1_train[i] /= 127.5

            image = Image.open(train_path + '/side2/' + file)
            side2_train[i] = (np.array(image).reshape(W, H, 1)) - 127.5
            side2_train[i] /= 127.5

            image = Image.open(train_path + '/panorama/' + file)
            panorama_train[i] = (np.array(image).reshape(W, H, 1)) - 127.5
            panorama_train[i] /= 127.5

        side1_val = np.empty((N_val, W, H, 1))
        side2_val = np.empty((N_val, W, H, 1))
        panorama_val = np.empty((N_val, W, H, 1))

        for i, file in enumerate(tqdm(val_list, desc='Validation dataset')):
            image = Image.open(validation_path + '/side1/' + file)
            side1_val[i] = (np.array(image).reshape(W, H, 1)) - 127.5
            side1_val[i] /= 127.5

            image = Image.open(validation_path + '/side2/' + file)
            side2_val[i] = (np.array(image).reshape(W, H, 1)) - 127.5
            side2_val[i] /= 127.5

            image = Image.open(validation_path + '/panorama/' + file)
            panorama_val[i] = (np.array(image).reshape(W, H, 1)) - 127.5
            panorama_val[i] /= 127.5

        return (side1_train, side2_train, panorama_train,
                side1_val, side2_val, panorama_val, N_train, N_val)


def one_image_generators(panorama_train, panorama_val, batch_size=50):
    pan_train_gen = ImageDataGenerator(
            vertical_flip=True).flow(panorama_train,
                                     batch_size=batch_size, shuffle=False)
    pan_val_gen = ImageDataGenerator(
            vertical_flip=True).flow(panorama_val,
                                     batch_size=batch_size, shuffle=False)
    return (pan_train_gen, pan_val_gen)


def three_image_generators(side1_train, side2_train, panorama_train,
                           side1_val, side2_val, panorama_val, batch_size=50):
    side1_train_gen = ImageDataGenerator(
            vertical_flip=True).flow(side1_train,
                                     batch_size=batch_size, shuffle=False)
    side2_train_gen = ImageDataGenerator(
            vertical_flip=True).flow(side2_train,
                                     batch_size=batch_size, shuffle=False)
    pan_train_gen = ImageDataGenerator(
            vertical_flip=True).flow(panorama_train,
                                     batch_size=batch_size, shuffle=False)

    side1_val_gen = ImageDataGenerator(
            vertical_flip=True).flow(side1_val,
                                     batch_size=batch_size, shuffle=False)
    side2_val_gen = ImageDataGenerator(
            vertical_flip=True).flow(side2_val,
                                     batch_size=batch_size, shuffle=False)
    pan_val_gen = ImageDataGenerator(
            vertical_flip=True).flow(panorama_val,
                                     batch_size=batch_size, shuffle=False)

    # генераторы, возвращающие тройки изображений
    train_gen = zip(side1_train_gen, side2_train_gen, pan_train_gen)
    val_gen = zip(side1_val_gen, side2_val_gen, pan_val_gen)
    return (train_gen, val_gen)


def save_p2p_models(models_path, trend_num, nn_name, f_gen, d, losses):
    path = models_path + '/trend' + str(trend_num)
    try:
        mkdir(path + '/' + nn_name)
    except FileExistsError:
        print('Dir already exist')
    f_gen.save_weights(path + '/' + nn_name + '/f_gen.weights')
    f_gen.save(path + '/' + nn_name + '/f_gen.h5')
    d.save_weights(path + '/' + nn_name + '/d.weights')
    d.save(path + '/' + nn_name + '/d.h5')
    np.save(path + '/' + nn_name + '/losses.npy', losses)
    print('Models saved successfully')


# def load_p2p_models(models_path, trend_num, nn_name, f_gen, d):
def load_p2p_models(models_path, trend_num, nn_name):
    path = models_path + '/trend' + str(trend_num)
    # f_gen.load_weights(path + '/' + nn_name + '/f_gen.weights')
    # d.load_weights(path + '/' + nn_name + '/d.weights')
    f_gen = load_model(path + '/' + nn_name + '/f_gen.h5')
    d = load_model(path + '/' + nn_name + '/d.h5')
    losses = np.load(path + '/' + nn_name + '/losses.npy').item()
    return (f_gen, d, losses)


def plot_p2p_models(models_path, trend_num, nn_name, f_gen, d, p2p):
    path = models_path + '/trend' + str(trend_num)
    try:
        mkdir(path + '/' + nn_name)
    except FileExistsError:
        print('Dir already exist')
    plot_model(f_gen, to_file=path + '/' + nn_name + '/f_gen.png',
               show_shapes=True, show_layer_names=True)
    plot_model(d, to_file=path + '/' + nn_name + '/d.png',
               show_shapes=True, show_layer_names=True)
    plot_model(p2p, to_file=path + '/' + nn_name + '/p2p.png',
               show_shapes=True, show_layer_names=True)


def plot_p2p_losses(models_path, trend_num, nn_name, losses):
    path = models_path + '/trend' + str(trend_num)
    try:
        mkdir(path + '/' + nn_name)
    except FileExistsError:
        print('Dir already exist')
    plt.figure(figsize=(10, 5))
    plt.plot(losses['d'], label='d')
    plt.plot(losses['p2p'], label='p2p')
    plt.legend()
    plt.savefig(path + '/' + nn_name + '/loss_train.png')

    plt.figure(figsize=(10, 5))
    plt.plot(losses['d_val'], label='d_val')
    plt.plot(losses['p2p_val'], label='p2p_val')
    plt.legend()
    plt.savefig(path + '/' + nn_name + '/loss_val.png')


def nn_verification(models_path, trend_num, nn_name, f_gen, n, W, H, l0, l1,
                    l_trend, AA, r):
    path = models_path + '/trend' + str(trend_num) + '/' + nn_name

    try:
        mkdir(path + '/verification')
        mkdir(path + '/verification' + '/side1')
        mkdir(path + '/verification' + '/side2')
        mkdir(path + '/verification' + '/panorama')
        mkdir(path + '/verification' + '/nn_output')
    except FileExistsError:
        print('Dir already exist')
    for i in tqdm(range(n), desc='Verification'):
        file_name = 'sample' + str(i) + '.jpg'
        ag = (l0, l1, l_trend)
        res = Parallel(n_jobs=-1)(delayed(generate_sample)(W, H, l, i, AA,
                                  r) for l in ag)
        side1 = res[0]
        side2 = res[1]
        pan = res[2]
        side1.save(path + '/verification/side1/' + file_name)
        side2.save(path + '/verification/side2/' + file_name)
        pan.save(path + '/verification/panorama/' + file_name)
        side1 = (np.array(side1).reshape(1, W, H, 1) - 127.5) / 127.5
        side2 = (np.array(side2).reshape(1, W, H, 1) - 127.5) / 127.5
        input_data = np.concatenate((side1, side2), axis=3)
        gen = f_gen.predict(input_data)
        nn_img = (127.5 * gen.reshape(W, H) + 127.5).astype('uint8')
        nn_img = Image.fromarray(nn_img, mode='L')
        nn_img.save(path + '/verification/nn_output/' + file_name)
    print('NN output saved successfully.')


def create_tb_callback(models_path, trend_num, nn_name):
    path = models_path + '/trend' + str(trend_num) + '/' + nn_name + '/tb_logs'
    tbCallback = TensorBoard(log_dir=path, histogram_freq=1, write_images=True)
    return tbCallback


# trend MSE metrcis
def tr_mse(sample, tr_mse_0, window):
    W = sample.width
    H = sample.height
    pixel_map = sample.load()
    steps = W - window + 1
    tr = np.zeros(steps)
    for shift in range(steps):
        window_sum = 0
        for i in range(window):
            for j in range(H):
                window_sum += abs(pixel_map[shift+i, j] - 255) / 255
        tr[shift] = window_sum / (window * H)
    err = (tr - tr_mse_0) ** 2
    return (err.mean(), err, tr)


def tr_mse_nn_output(verification_path, r):
    side1_list = listdir(verification_path + '/side1')
    side2_list = listdir(verification_path + '/side2')
    pan_list = listdir(verification_path + '/panorama')
    nn_output_list = listdir(verification_path + '/nn_output')
    N = len(nn_output_list)
    # just for parameter definition
    img = Image.open(verification_path + '/nn_output/' + nn_output_list[0])
    W = img.width
    window = 2 * r
    steps = W - window + 1
    tr_side1 = np.zeros(steps)
    tr_side2 = np.zeros(steps)
    tr_pan = np.zeros(steps)
    val_mse = np.zeros(steps)
    for file in tqdm(side1_list, desc='Side 1'):
            image = Image.open(verification_path + '/side1/' + file)
            _, _, tr = tr_mse(image, np.zeros(steps), window)
            tr_side1 += tr

    for file in tqdm(side2_list, desc='Side 2'):
            image = Image.open(verification_path + '/side2/' + file)
            _, _, tr = tr_mse(image, np.zeros(steps), window)
            tr_side2 += tr

    for file in tqdm(pan_list, desc='Panorama'):
            image = Image.open(verification_path + '/panorama/' + file)
            _, err, tr = tr_mse(image, np.zeros(steps), window)
            tr_pan += tr
            val_mse += err

    val_mse /= N
    tr_side1 /= N
    tr_side2 /= N
    tr_pan /= N
    nn_mse = 0
    nn_err = np.zeros(steps)
    nn_tr = np.zeros(steps)
    for file in tqdm(nn_output_list, desc='NN output'):
            image = Image.open(verification_path + '/nn_output/' + file)
            mse, err, tr = tr_mse(image, val_mse, window)
            nn_mse += mse
            nn_err += err
            nn_tr += tr
    try:
        mkdir(verification_path + '/metrics')
    except FileExistsError:
        print('Dir already exist')
    nn_mse /= N
    nn_err /= N
    nn_tr /= N
    np.save(verification_path + '/metrics/tr_side1.npy', tr_side1)
    np.save(verification_path + '/metrics/tr_side2.npy', tr_side2)
    np.save(verification_path + '/metrics/tr_pan.npy', tr_pan)
    np.save(verification_path + '/metrics/nn_mse.npy', nn_mse)
    np.save(verification_path + '/metrics/nn_err.npy', nn_err)
    np.save(verification_path + '/metrics/nn_tr.npy', nn_tr)
    print('Metrics saved successfully.')
    return (nn_mse, nn_err, nn_tr, tr_side1, tr_side2, tr_pan)
