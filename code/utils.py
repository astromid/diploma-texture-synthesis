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
from keras.models import load_model
from keras.utils import plot_model


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
            panorama_train[i] = (np.array(image).T.reshape(W, H, 1)) - 127.5
            panorama_train[i] /= 127.5

        # load and normalization validation images
        panorama_val = np.empty((N_val, W, H, 1))
        for i, file in enumerate(tqdm(val_list, desc='Validation dataset')):
            image = Image.open(validation_path + '/panorama/' + file)
            panorama_val[i] = (np.array(image).T.reshape(W, H, 1)) - 127.5
            panorama_val[i] /= 127.5

        return (panorama_train, panorama_val)

    # 3 image mode
    if (mode == 3):
        side1_train = np.empty((N_train, W, H, 1))
        side2_train = np.empty((N_train, W, H, 1))
        panorama_train = np.empty((N_train, W, H, 1))

        for i, file in enumerate(tqdm(train_list, desc='Train dataset')):
            image = Image.open(train_path + '/side1/' + file)
            side1_train[i] = (np.array(image).T.reshape(W, H, 1)) - 127.5
            side1_train[i] /= 127.5

            image = Image.open(train_path + '/side2/' + file)
            side2_train[i] = (np.array(image).T.reshape(W, H, 1)) - 127.5
            side2_train[i] /= 127.5

            image = Image.open(train_path + '/panorama/' + file)
            panorama_train[i] = (np.array(image).T.reshape(W, H, 1)) - 127.5
            panorama_train[i] /= 127.5

        side1_val = np.empty((N_val, W, H, 1))
        side2_val = np.empty((N_val, W, H, 1))
        panorama_val = np.empty((N_val, W, H, 1))

        for i, file in enumerate(tqdm(val_list, desc='Validation dataset')):
            image = Image.open(validation_path + '/side1/' + file)
            side1_val[i] = (np.array(image).T.reshape(W, H, 1)) - 127.5
            side1_val[i] /= 127.5

            image = Image.open(validation_path + '/side2/' + file)
            side2_val[i] = (np.array(image).T.reshape(W, H, 1)) - 127.5
            side2_val[i] /= 127.5

            image = Image.open(validation_path + '/panorama/' + file)
            panorama_val[i] = (np.array(image).T.reshape(W, H, 1)) - 127.5
            panorama_val[i] /= 127.5

        return (side1_train, side2_train, panorama_train,
                side1_val, side2_val, panorama_val)


def save_p2p_models(models_path, trend_num, nn_name, f_gen, d, losses):
    path = models_path + '/trend' + str(trend_num)
    try:
        mkdir(path + '/' + nn_name)
    except FileExistsError:
        print('Dir already exist')
    f_gen.save(path + '/' + nn_name + '/f_gen.h5')
    d.save(path + '/' + nn_name + '/d.h5')
    np.save(path + '/' + nn_name + '/losses.npy', losses)
    print('Models saved successfully')


def load_p2p_models(models_path, trend_num, nn_name):
    path = models_path + '/trend' + str(trend_num)
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
    plt.plot(losses['dVal'], label='dVal')
    plt.plot(losses['p2pVal'], label='p2pVal')
    plt.legend()
    plt.savefig(path + '/' + nn_name + '/loss_val.png')


def generate_samples(models_path, trend_num, nn_name, f_gen, dataGen, n, W, H):
    path = models_path + '/trend' + str(trend_num)
    try:
        mkdir(path + '/' + nn_name + '/nn_output')
    except FileExistsError:
        print('Dir already exist')
    side1, side2, pan = next(dataGen)
    input_data = np.concatenate((side1[:n], side2[:n]), axis=3)
    gen = f_gen.predict(input_data)
    side1_images = []
    side2_images = []
    pan_images = []
    samples = []
    for i in range(n):
        curr_s1 = (127.5 * (side1[i].reshape(W, H).T) + 127.5).astype('uint8')
        curr_s2 = (127.5 * (side2[i].reshape(W, H).T) + 127.5).astype('uint8')
        curr_pan = (127.5 * (pan[i].reshape(W, H).T) + 127.5).astype('uint8')
        curr_smpl = (127.5 * (gen[i].reshape(W, H).T) + 127.5).astype('uint8')
        side1_images.append(Image.fromarray(curr_s1, mode='L'))
        side2_images.append(Image.fromarray(curr_s2, mode='L'))
        pan_images.append(Image.fromarray(curr_pan, mode='L'))
        samples.append(Image.fromarray(curr_smpl, mode='L'))
        fileName = 'sample' + str(i) + '.jpg'
        samples[i].save(path + '/' + nn_name + '/nn_output/' + fileName)
    print('NN output saved successfully.')
    return (side1_images, side2_images, pan_images, samples)


def tr_mse(sample, tr_mse_0):
    return 0
