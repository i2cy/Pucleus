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


class Peek(object):

    def __init__(self, position, edges, mca_data):

        self.__mca_data = mca_data

        self.position = position
        self.edges = edges

        self.mean = position
        self.left_edge = self.edges[0]
        self.right_edge = self.edges[1]
