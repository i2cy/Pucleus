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
        for i, e in enumerate(self.data[window: -window-1]):
            ret[window + i] = np.mean(self.data[i: window_size+i])

        self.data = ret

        return self.copy()


class BaryCenter(Smoother):

    def __init__(self, data):
        super(BaryCenter, self).__init__(data)

    def smooth(self, window_size=3):
        ret = self.data.copy()
        window = window_size // 2
        for i, e in enumerate(self.data[window: -window - 1]):
            t1 = self.data[i: window_size+i]
            t1 = list(t1)
            t0 = []
            while len(t1) > 1:
                for ti, te in enumerate(t1[:-1]):
                    t0.append((te + t1[ti+1]) / 2.0)
                t1 = t0
                t0 = []
            ret[window + i] = t1[0]

        self.data = ret

        return self.copy()
