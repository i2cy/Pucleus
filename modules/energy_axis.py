#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: i2cy(i2cy@outlook.com)
# Project: main.py
# Filename: energy_axis
# Created on: 2021/11/23

import numpy as np


def linear_regression(x, y):
    """

    :param x: list
    :param y: list
    :return: b, a, y=ax+b
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=np.float64)
    if not isinstance(y, np.ndarray):
        y = np.array(y, dtype=np.float64)

    N = len(x)
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum(x ** 2)
    sumxy = sum(x * y)

    A = np.mat([[N, sumx], [sumx, sumx2]])
    b = np.array([sumy, sumxy])

    return np.linalg.solve(A, b)
