# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:50:34 2019

© MIT 6.S191: Introduction to Deep Learning
introtodeeplearning.com

"""

from IPython import display
import matplotlib.pyplot as plt
import time
from string import Formatter
import tensorflow as tf
import numpy as np


class PeriodicPlotter:
    def __init__(self, sec, xlabel='', ylabel='', scale=None):

        self.xlabel = xlabel
        self.ylabel = ylabel
        self.sec = sec
        self.scale = scale

        self.tic = time.time()

    def plot(self, data):
        if time.time() - self.tic > self.sec:
            plt.cla()

            if self.scale is None:
                plt.plot(data)
            elif self.scale == 'semilogx':
                plt.semilogx(data)
            elif self.scale == 'semilogy':
                plt.semilogy(data)
            elif self.scale == 'loglog':
                plt.loglog(data)
            else:
                raise ValueError("unrecognized parameter scale {}".format(self.scale))

            plt.xlabel(self.xlabel);
            plt.ylabel(self.ylabel)
            display.clear_output(wait=True)
            display.display(plt.gcf())

            self.tic = time.time()


def display_model(model):
    tf.keras.utils.plot_model(model,
                              to_file='tmp.png',
                              show_shapes=True)
    return display.Image('tmp.png')


def plot_sample(x, y, vae):
    plt.figure(figsize=(2, 1))
    plt.subplot(1, 2, 1)

    idx = np.where(y.numpy() == 1)[0][0]
    plt.imshow(x[idx])
    plt.grid(False)

    plt.subplot(1, 2, 2)
    plt.imshow(vae(x)[idx])
    plt.grid(False)

    plt.show()


class LossHistory:
    def __init__(self, smoothing_factor=0.0):
        self.alpha = smoothing_factor
        self.loss = []

    def append(self, value):
        self.loss.append(self.alpha * self.loss[-1] + (1 - self.alpha) * value if len(self.loss) > 0 else value)

    def get(self):
        return self.loss
