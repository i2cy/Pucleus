#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: i2cy(i2cy@outlook.com)
# Project: main.py
# Filename: find_peek
# Created on: 2021/11/23

from .mca import MCA, Pulses
import numpy as np


class PeekFinder(object):

    def __init__(self, mca=None, range=(0, 1024)):
        """
        Peek finder

        :param mca: MCA object
        """
        assert isinstance(mca, MCA)
        self.mca = mca
        self.peeks = []
        self.range = range

        self.__index = 0

    def __len__(self):
        return len(self.peeks)

    def __getitem__(self, item):
        """
        return the index of peeks

        :param item:
        :return:
        """
        return self.peeks[item]

    def __iter__(self):
        self.__index = 0
        return self

    def __next__(self):
        if self.__index >= len(self.peeks):
            raise StopIteration
        else:
            ret = self.peeks[self.__index]
            self.__index += 1
            return ret

    def reset(self):
        self.__index = 0
        self.peeks = []

    def search(self, range=None):
        if range is not None:
            self.range = range


class Peek(object):

    def __init__(self, position, edges, mca_data):

        self.__mca_data = mca_data

        self.position = position
        self.edges = edges

        self.mean = position
        self.left_edge = self.edges[0]
        self.right_edge = self.edges[1]


class SimpleCompare(PeekFinder):

    def __init__(self, k, m):
        """
        简单比较法

        :param k: float, 找峰阈值，一般在1~1.5之间
        :param m: int, 寻峰宽度因子
        """

        self.k = k
        self.m = m

        self.left_edge = -1
        self.right_edge = -1

    def __find_peek(self, index, channel_data):
        peek = None
        if index >= len(channel_data - self.m):
            return peek

        for i, cnt in enumerate(channel_data[index+self.m:-self.m]):
            peek_value = cnt - (self.k * cnt**-0.5)
            if channel_data[index + i + self.m] < peek_value \
                    > channel_data[index + i - self.m]:
                peek = index + i

        if peek is None:
            return peek

        maxi = max(channel_data[peek-self.m, peek+self.m])
        peek = channel_data


        return peek

    def search(self, range=None):
        super(SimpleCompare, self).search(range)

        index = 0

        channel_data = self.mca[self.range[0], self.range[1]]

        while index < len(channel_data) - 1:
            peek = self.__find_peek(index, channel_data)






