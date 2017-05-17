#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 16:42:18 2017

@author: ysbudakyan
Keras models definitions
"""

from keras import objectives
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Activation, Dropout


# U-Net Generator
def g_unet(nf, a_ch=1, b_ch=1, out_ch=1, alpha=0.2,
           model_name='unet'):
    ''' params:
    input shape = (256, 256, a_ch + b_ch)
    output = (256, 256, out_ch)
    nf - number of filters on input layer
    alpha - LeakyReLU parameter
    '''
    i = Input(shape=(256, 256, a_ch + b_ch))
    # (256, 256, a_ch + b_ch)

    conv1 = Conv2D(nf, (3, 3), padding='same', strides=(2, 2))(i)
    conv1 = BatchNormalization(axis=3)(conv1)
    x = LeakyReLU(alpha)(conv1)
    # (128, 128, nf)

    conv2 = Conv2D(nf*2, (3, 3), padding='same', strides=(2, 2))(x)
    conv2 = BatchNormalization(axis=3)(conv2)
    x = LeakyReLU(alpha)(conv2)
    # (64, 64, nf*2)

    conv3 = Conv2D(nf*4, (3, 3), padding='same', strides=(2, 2))(x)
    conv3 = BatchNormalization(axis=3)(conv3)
    x = LeakyReLU(alpha)(conv3)
    # (32, 32, nf*4)

    conv4 = Conv2D(nf*8, (3, 3), padding='same', strides=(2, 2))(x)
    conv4 = BatchNormalization(axis=3)(conv4)
    x = LeakyReLU(alpha)(conv4)
    # (16, 16, nf*8)

    conv5 = Conv2D(nf*8, (3, 3), padding='same', strides=(2, 2))(x)
    conv5 = BatchNormalization(axis=3)(conv5)
    x = LeakyReLU(alpha)(conv5)
    # (8, 8, nf*8)

    conv6 = Conv2D(nf*8, (3, 3), padding='same', strides=(2, 2))(x)
    conv6 = BatchNormalization(axis=3)(conv6)
    x = LeakyReLU(alpha)(conv6)
    # (4, 4, nf*8)

    conv7 = Conv2D(nf*8, (3, 3), padding='same', strides=(2, 2))(x)
    conv7 = BatchNormalization(axis=3)(conv7)
    x = LeakyReLU(alpha)(conv7)
    # (2, 2, nf*8)

    conv8 = Conv2D(nf*8, (2, 2), padding='valid', strides=(1, 1))(x)
    conv8 = BatchNormalization(axis=3)(conv8)
    x = LeakyReLU(alpha)(conv8)
    # (1, 1, nf*8)

    dconv1 = Conv2DTranspose(nf*8, (2, 2), padding='valid', strides=(1, 1))(x)
    dconv1 = BatchNormalization(axis=3)(dconv1)
    dconv1 = Dropout(0.5)(dconv1)
    x = concatenate([dconv1, conv7], axis=3)
    # x = dconv1
    x = LeakyReLU(alpha)(x)
    # (2, 2, nf*(8 + 8))

    dconv2 = Conv2DTranspose(nf*8, (3, 3), padding='same', strides=(2, 2))(x)
    dconv2 = BatchNormalization(axis=3)(dconv2)
    dconv2 = Dropout(0.5)(dconv2)
    x = concatenate([dconv2, conv6], axis=3)
    # x = dconv2
    x = LeakyReLU(alpha)(x)
    # (4, 4, nf*(8 + 8))

    dconv3 = Conv2DTranspose(nf*8, (3, 3), padding='same', strides=(2, 2))(x)
    dconv3 = BatchNormalization(axis=3)(dconv3)
    # dconv3 = Dropout(0.5)(dconv3)
    x = concatenate([dconv3, conv5], axis=3)
    # x = dconv3
    x = LeakyReLU(alpha)(x)
    # (8, 8, nf*(8 + 8))

    dconv4 = Conv2DTranspose(nf*8, (3, 3), padding='same', strides=(2, 2))(x)
    dconv4 = BatchNormalization(axis=3)(dconv4)
    # dconv4 = Dropout(0.5)(dconv4)
    x = concatenate([dconv4, conv4], axis=3)
    # x = dconv4
    x = LeakyReLU(alpha)(x)
    # (16, 16, nf*(8 + 8))

    dconv5 = Conv2DTranspose(nf*4, (3, 3), padding='same', strides=(2, 2))(x)
    dconv5 = BatchNormalization(axis=3)(dconv5)
    # dconv5 = Dropout(0.5)(dconv5)
    x = concatenate([dconv5, conv3], axis=3)
    # x = dconv5
    x = LeakyReLU(alpha)(x)
    # (32, 32, nf*(4 + 4))

    dconv6 = Conv2DTranspose(nf*2, (3, 3), padding='same', strides=(2, 2))(x)
    dconv6 = BatchNormalization(axis=3)(dconv6)
    # dconv6 = Dropout(0.5)(dconv6)
    x = concatenate([dconv6, conv2], axis=3)
    # x = dconv6
    x = LeakyReLU(alpha)(x)
    # (64, 64, nf*(2 + 2))

    dconv7 = Conv2DTranspose(nf, (3, 3), padding='same', strides=(2, 2))(x)
    dconv7 = BatchNormalization(axis=3)(dconv7)
    x = concatenate([dconv7, conv1], axis=3)
    # x = dconv7
    x = LeakyReLU(alpha)(x)
    # (128, 128, nf*(1 + 1))

    dconv8 = Conv2DTranspose(out_ch, (3, 3), padding='same', strides=(2, 2))(x)
    # (256, 256, out_ch)

    out = Activation('tanh')(dconv8)
    unet = Model(i, out, name=model_name)

    return unet


# Discriminator
def discriminator(nf, a_ch=1, b_ch=1, c_ch=1, opt=Adam(lr=1e-4, beta_1=0.2),
                  alpha=0.2, model_name='d'):
    ''' params:
    a_ch - first image channels
    b_ch - second
    c_ch - third
    nf - number of filters on input layer
    alpha - LeakyReLU parameter
    '''
    i = Input(shape=(256, 256, a_ch + b_ch + c_ch))
    # (256, 256, a_ch + b_ch + c_ch)

    conv1 = Conv2D(nf, (3, 3), padding='same', strides=(2, 2))(i)
    x = LeakyReLU(alpha)(conv1)
    # (128, 128, nf)

    conv2 = Conv2D(nf*2, (3, 3), padding='same', strides=(2, 2))(x)
    x = LeakyReLU(alpha)(conv2)
    # (64, 64, nf*2)

    conv3 = Conv2D(nf*4, (3, 3), padding='same', strides=(2, 2))(x)
    x = LeakyReLU(alpha)(conv3)
    # (32, 32, nf*4)

    conv4 = Conv2D(nf*8, (3, 3), padding='same', strides=(2, 2))(x)
    x = LeakyReLU(alpha)(conv4)
    # (16, 16, nf*8)

    conv5 = Conv2D(1, (3, 3), padding='same', strides=(2, 2))(x)
    out = Activation('tanh')(conv5)
    # (8, 8, 1)

    d = Model(i, out, name=model_name)

    def d_loss(y_true, y_pred):
        L = objectives.binary_crossentropy(K.batch_flatten(y_true),
                                           K.batch_flatten(y_pred))
        return L

    d.compile(optimizer=opt, loss=d_loss)
    return d


def pix2pix(atob, d, a_ch=1, b_ch=1, eta=100, opt=Adam(lr=1e-4, beta_1=0.2),
            model_name='pix2pix'):
    '''
    atob - full generator
    d - discriminator
    '''
    a1 = Input(shape=(256, 256, a_ch))
    a2 = Input(shape=(256, 256, a_ch))
    b = Input(shape=(256, 256, b_ch))

    # generate image on a1 и a2 with generator
    bp = atob(concatenate([a1, a2], axis=3))

    # discriminator input is triplet of images
    d_in = concatenate([a1, a2, bp], axis=3)
    pix2pix = Model([a1, a2, b], d(d_in), name=model_name)

    def p2p_loss(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)

        # adversarial loss
        L_adv = objectives.binary_crossentropy(y_true_flat, y_pred_flat)

        # atob loss
        b_flat = K.batch_flatten(b)
        bp_flat = K.batch_flatten(bp)
        L_atob = K.mean(K.abs(b_flat - bp_flat))

        return L_adv + eta * L_atob

    # train generator - freeze discriminator
    pix2pix.get_layer('d').trainable = False

    pix2pix.compile(optimizer=opt, loss=p2p_loss)
    return pix2pix
