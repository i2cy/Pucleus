#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: i2cy(i2cy@outlook.com)
# Project: main.py
# Filename: smooth
# Created on: 2021/11/23

import numpy as np
from .mca import MCA


class Smoother(MCA):

    def __init__(self, data):
        assert isinstance(data, MCA)
        super(Smoother, self).__init__(data)

    def __call__(self, *key, **kwargs):
        return self.smooth(*key, **kwargs)

    def smooth(self, *key, **kwargs):
        return None


class Mean(Smoother):

    def __init__(self, data):
        super(Mean, self).__init__(data)

    def smooth(self, window_size=3):
        ret = self.data.copy()
        window = window_size // 2
        for i, e in enumerate(self.data[window: -window - 1]):
            ret[window + i] = np.mean(self.data[i: window_size + i])

        self.data = ret

        return self.copy()


class BaryCenter(Smoother):

    def __init__(self, data):
        super(BaryCenter, self).__init__(data)

    def smooth(self, window_size=3):
        ret = self.data.copy()
        window = window_size // 2
        for i, e in enumerate(self.data[window: -window - 1]):
            t1 = self.data[i: window_size + i]
            t1 = list(t1)
            t0 = []
            while len(t1) > 1:
                for ti, te in enumerate(t1[:-1]):
                    t0.append((te + t1[ti + 1]) / 2.0)
                t1 = t0
                t0 = []
            ret[window + i] = t1[0]

        self.data = ret

        return self.copy()


class PolynomialLeastSquareMethod(Smoother):

    def __init__(self, data):
        super(PolynomialLeastSquareMethod, self).__init__(data)

        self.__ks = [[0, 0, 0, 0, 0, 0, 0],
                     [35, 17, 12, -3, 0, 0, 0],
                     [21, 7, 6, 3, -2, 0, 0],
                     [231, 59, 54, 39, 14, -21, 0],
                     [429, 89, 84, 69, 44, 9, -36]
                     ]

    def smooth(self, m, h):
        """
        :param m: int, 2m+1为扫描窗口宽度
        :param h: int, 每个等距点之间的间距
        :return: mca
        """
        ret = self.data.copy()
        ks = self.__ks[m - 1]

        for i, e in enumerate(self.data[m * h: -m * h]):
            cal_t = ks[1] * self.data[m * h + i]

            for i2 in range(m):
                cal_t += self.data[(m - i2 - 1) * h + i] * ks[i2 + 2]

                cal_t += self.data[(m + i2 + 1) * h + i] * ks[i2 + 2]

            cal_t /= ks[0]

            if cal_t < 0:
                cal_t = 0

            ret[m * h + i] = cal_t

        self.data = ret

        return self.copy()
