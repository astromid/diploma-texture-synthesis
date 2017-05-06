#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 17:30:47 2017

@author: ysbudakyan
Functions to train NN
"""

import numpy as np
from tqdm import tnrange
from IPython.display import clear_output
from keras.callbacks import ReduceLROnPlateau


# generator of labeled data for discriminator training
def d_generator(data_gen, atob, dout_size):
    while True:
        # fake triplet
        a1_fake, a2_fake, _ = next(data_gen)
        a_fake = np.concatenate((a1_fake, a2_fake), axis=3)
        b_fake = atob.predict(a_fake)
        # real triplet
        a1_real, a2_real, b_real = next(data_gen)
        # unify into united batch
        a_real = np.concatenate((a1_real, a2_real), axis=3)
        batch_a = np.concatenate((a_fake, a_real), axis=0)
        batch_b = np.concatenate((b_fake, b_real), axis=0)
        batch_x = np.concatenate((batch_a, batch_b), axis=3)
        # labels: fake - 1, real - 0
        batch_y = np.ones((batch_x.shape[0], 1) + dout_size)
        batch_y[a_fake.shape[0]:] = 0
        yield batch_x, batch_y


def train_discriminator(d, data_gen, steps_per_epoch=40, cb=[]):
    return d.fit_generator(data_gen, steps_per_epoch=steps_per_epoch*2,
                           epochs=1, verbose=1, callbacks=cb)


# generator for pix2pix net
def p2p_generator(data_gen, dout_size):
    for a1, a2, b in data_gen:
        # labels: fake - 1, real - 0
        y = np.zeros((a1.shape[0], 1) + dout_size)
        yield [a1, a2, b], y


def train_p2p(p2p, data_gen, steps_per_epoch=40, cb=[]):
    return p2p.fit_generator(data_gen, steps_per_epoch=steps_per_epoch,
                             epochs=1, verbose=1, callbacks=cb)


def metrics(d_gen, p2p_gen, d, p2p, losses, val_steps):
    d_loss = d.evaluate_generator(d_gen, val_steps)
    p2p_loss = p2p.evaluate_generator(p2p_gen, val_steps)
    losses['d_val'].append(d_loss)
    losses['p2p_val'].append(p2p_loss)
    return d_loss, p2p_loss


def train_iteration(d, p2p, d_gen, p2p_gen, losses, steps_per_epoch,
                    cb_d, cb_p2p):
    # discriminator
    d_hist = train_discriminator(d, d_gen, steps_per_epoch, cb_d)
    losses['d'].extend(d_hist.history['loss'])
    # generator
    p2p_hist = train_p2p(p2p, p2p_gen, steps_per_epoch, cb_p2p)
    losses['p2p'].extend(p2p_hist.history['loss'])


def train(atob, d, p2p, train_gen, val_gen, epochs, train_samples, val_samples,
          batch_size):
    # create necessary generators
    dout_size = d.output_shape[1:3]
    d_gen_train = d_generator(train_gen, atob, dout_size)
    # for tensorflow
    next(d_gen_train)
    d_gen_val = d_generator(val_gen, atob, dout_size)
    p2p_gen_train = p2p_generator(train_gen, dout_size)
    p2p_gen_val = p2p_generator(val_gen, dout_size)
    losses = {'p2p': [], 'd': [], 'p2p_val': [], 'd_val': []}
    steps_per_epoch = np.ceil(train_samples / batch_size)
    val_steps = np.ceil(val_samples / batch_size)
    # create callbacks
    reduce_lr_d = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                    patience=5, min_lr=1e-7)
    reduce_lr_p2p = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                      patience=5, min_lr=1e-7)
    # train loop
    for e in tnrange(epochs, desc='Epoches'):
        clear_output()
        train_iteration(d, p2p, d_gen_train, p2p_gen_train, losses,
                        steps_per_epoch, [reduce_lr_d], [reduce_lr_p2p])
        # evaluate metrics
        metrics(d_gen_val, p2p_gen_val, d, p2p, losses, val_steps)
    return losses
