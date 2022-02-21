#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: i2cy(i2cy@outlook.com)
# Project: main.py
# Filename: utils
# Created on: 2021/11/21

import random
import colorsys
from .mca import MCA
from .energy_axis import linear_regression
import numpy as np


class ColorManager(object):

    def __init__(self):

        self.base = []
        self.gen_color(16)

        self.index = 0

    def __getitem__(self, item):
        return self.gen_color(item)

    def __len__(self):
        return len(self.base)

    def get_color(self, index=None):
        ret = None
        if index is None:
            if self.index >= len(self):
                self.index = 0
            index = self.index
            self.index += 1
        ret = colorsys.hls_to_rgb(*self.base[index])
        ret = [int(255)*ele for ele in ret]
        return ret

    def gen_color(self, num):
        hls_colors = []
        i = 0
        step = 360.0 / num
        while i < 360:
            h = i
            s = 90 + random.random() * 10
            l = 50 + random.random() * 10
            t_hlsc = [h / 360.0, l / 100.0, s / 100.0]
            hls_colors.append(t_hlsc)
            i += step

        random.shuffle(hls_colors)
        self.base = hls_colors


def get_R_square(original_plot, current_plot):
    """
    linear_regression with R^2

    :param original_plot:
    :param current_plot:
    :return:
    """
    assert isinstance(original_plot, MCA)
    assert isinstance(current_plot, MCA)
    original_data = original_plot.as_numpy()
    original_data.sort()
    current_data = current_plot.as_numpy()
    current_data.sort()
    b, a = linear_regression(original_data, current_data)

    rr = np.sum(((original_data * a + b) - np.mean(current_data))**2) \
         / np.sum((current_data - np.mean(current_data))**2)

    return rr, a, b
