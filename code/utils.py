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
from PIL import Image
from os import listdir, mkdir
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard


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
            vertical_flip=True).flow(panorama_train, batch_size=batch_size)
    pan_val_gen = ImageDataGenerator(
            vertical_flip=True).flow(panorama_val, batch_size=batch_size)
    return (pan_train_gen, pan_val_gen)


def three_image_generators(side1_train, side2_train, panorama_train,
                           side1_val, side2_val, panorama_val, batch_size=50):
    side1_train_gen = ImageDataGenerator(
            vertical_flip=False).flow(side1_train, batch_size=batch_size)
    side2_train_gen = ImageDataGenerator(
            vertical_flip=False).flow(side2_train, batch_size=batch_size)
    pan_train_gen = ImageDataGenerator(
            vertical_flip=False).flow(panorama_train, batch_size=batch_size)

    side1_val_gen = ImageDataGenerator(
            vertical_flip=False).flow(side1_val, batch_size=batch_size)
    side2_val_gen = ImageDataGenerator(
            vertical_flip=False).flow(side2_val, batch_size=batch_size)
    pan_val_gen = ImageDataGenerator(
            vertical_flip=False).flow(panorama_val, batch_size=batch_size)

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
    d.save_weights(path + '/' + nn_name + '/d.weights')
    np.save(path + '/' + nn_name + '/losses.npy', losses)
    print('Models saved successfully')


def load_p2p_models(models_path, trend_num, nn_name, f_gen, d):
    path = models_path + '/trend' + str(trend_num)
    f_gen.load_weights(path + '/' + nn_name + '/f_gen.weights')
    d.load_weights(path + '/' + nn_name + '/d.weights')
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


def gen_nn_output(models_path, trend_num, nn_name, f_gen, dataGen, n, W, H):
    path = models_path + '/trend' + str(trend_num)
    try:
        mkdir(path + '/' + nn_name + '/nn_output')
    except FileExistsError:
        print('Dir already exist')
    side1, side2, pan = next(dataGen)
    if(n > pan.shape[0]):
        n = pan.shape[0]
    input_data = np.concatenate((side1[:n], side2[:n]), axis=3)
    gen = f_gen.predict(input_data)
    side1_images = []
    side2_images = []
    pan_images = []
    samples = []
    for i in range(n):
        curr_s1 = (127.5 * side1[i].reshape(W, H) + 127.5).astype('uint8')
        curr_s2 = (127.5 * side2[i].reshape(W, H) + 127.5).astype('uint8')
        curr_pan = (127.5 * pan[i].reshape(W, H) + 127.5).astype('uint8')
        curr_smpl = (127.5 * gen[i].reshape(W, H) + 127.5).astype('uint8')
        side1_images.append(Image.fromarray(curr_s1, mode='L'))
        side2_images.append(Image.fromarray(curr_s2, mode='L'))
        pan_images.append(Image.fromarray(curr_pan, mode='L'))
        samples.append(Image.fromarray(curr_smpl, mode='L'))
        fileName = 'sample' + str(i) + '.jpg'
        samples[i].save(path + '/' + nn_name + '/nn_output/' + fileName)
    print('NN output saved successfully.')
    return (side1_images, side2_images, pan_images, samples)


def create_tb_callback(models_path, trend_num, nn_name):
    path = models_path + '/trend' + str(trend_num) + '/' + nn_name + '/tb_logs'
    tbCallback = TensorBoard(log_dir=path, histogram_freq=1, write_images=True)
    return tbCallback


# trend MSE metrcis
def tr_mse(sample, tr_mse_0, r):
    # window width
    window = 2 * r
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


# fit trend MSE metrics
def tr_mse_fit(dataset_path, trend_num, r):
    dataset_path_with_trend = dataset_path + '/trend' + str(trend_num)
    train_path = dataset_path_with_trend + '/train'
    validation_path = dataset_path_with_trend + '/validation'
    train_list = listdir(train_path + '/panorama')
    val_list = listdir(validation_path + '/panorama')
    N_train = len(train_list)
    N_val = len(val_list)
    # just for parameter definition
    img = Image.open(train_path + '/panorama/' + train_list[0])
    W = img.width
    window = 2 * r
    steps = W - window + 1
    train_mse = np.zeros(steps)
    val_mse = np.zeros(steps)
    for i, file in enumerate(tqdm(train_list, desc='Train dataset')):
            image = Image.open(train_path + '/panorama/' + file)
            mse, err, _ = tr_mse(image, np.zeros(steps), r)
            train_mse += err
    for i, file in enumerate(tqdm(val_list, desc='Validation dataset')):
            image = Image.open(validation_path + '/panorama/' + file)
            mse, err, _ = tr_mse(image, np.zeros(steps), r)
            val_mse += err
    train_mse /= N_train
    val_mse /= N_val
    try:
        mkdir(dataset_path_with_trend + '/metrics')
    except FileExistsError:
        print('Dir already exist')
    np.save(dataset_path_with_trend + '/metrics/tmse.npy', train_mse)
    np.save(dataset_path_with_trend + '/metrics/vmse.npy', val_mse)
    print('Metrics saved successfully.')
    return (train_mse.mean(), val_mse.mean(), train_mse, val_mse)
