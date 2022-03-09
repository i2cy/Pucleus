#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: i2cy(i2cy@outlook.com)
# Project: main.py
# Filename: find_peek
# Created on: 2021/11/23

from .mca import MCA, Pulses
import numpy as np
import matplotlib.pyplot as plt


class PeekFinder(object):

    def __init__(self, mca=None, ranges=(0, 1024)):
        """
        Peek finder

        :param mca: MCA object
        """
        assert isinstance(mca, MCA)
        self.mca = mca
        self.peeks = []
        self.range = [int(ele) for ele in ranges]
        self.flag_searched = False

        self.__index = 0

    def __len__(self):
        if not self.flag_searched:
            self.search()
        return len(self.peeks)

    def __getitem__(self, item):
        """
        return the index of peeks

        :param item:
        :return:
        """
        if not self.flag_searched:
            self.search()
        return self.peeks[item]

    def __iter__(self):
        if not self.flag_searched:
            self.search()
        self.__index = 0
        return self

    def __next__(self):
        if not self.flag_searched:
            self.search()
        if self.__index >= len(self.peeks):
            raise StopIteration
        else:
            ret = self.peeks[self.__index]
            self.__index += 1
            return ret

    def reset(self):
        self.__index = 0
        self.peeks = []
        self.flag_searched = False

    def search(self, range=None):
        if range is not None:
            self.range = range

    def is_searched(self):
        return self.flag_searched


class Peek(object):

    def __init__(self, position, edges, mca_data):
        self.__mca_data = mca_data

        self.position = position
        self.edges = edges

        self.mean = position
        self.left_edge = self.edges[0]
        self.right_edge = self.edges[1]

    def plotshow(self):
        ind = np.zeros(len(self.__mca_data))
        ind[self.left_edge] = self.__mca_data[self.left_edge] + 100
        ind[self.right_edge] = self.__mca_data[self.right_edge] + 100
        top = np.zeros(len(self.__mca_data))
        top[self.position] = self.__mca_data[self.position] + 100

        plt.plot(self.__mca_data)
        plt.plot(ind, color="orange")
        plt.plot(top, color="red")

        plt.show()

    def area(self):
        ret = self.__mca_data[int(self.left_edge): int(self.right_edge)].sum()
        return ret

    def pure_area(self):
        ret = self.area()
        noise = self.__mca_data[int(self.left_edge)] + self.__mca_data[int(self.right_edge)]
        noise *= self.right_edge - self.left_edge
        noise /= 2
        ret -= noise
        return ret

    def get_feature_array(self):
        ret = np.zeros(len(self.__mca_data), dtype=np.float64)
        for i, ele in enumerate(self.__mca_data[int(self.left_edge):int(self.right_edge)]):
            ret[i + int(self.left_edge)] = ele
        return ret

    def get_reversed_feature_array(self):
        t = self.get_feature_array()
        ret = self.__mca_data.copy()
        ret -= t
        return ret

    def get_clip_feature_array(self, offset=1):
        y = self.__mca_data[int(self.left_edge):int(self.right_edge)]
        x = np.linspace(int(self.left_edge) + offset, int(self.right_edge) + offset,
                        len(y),
                        dtype=np.int32)
        return x, y

    def peek_location(self):
        ret = self.position
        a = self.__mca_data[self.position + 1] - self.__mca_data[self.position - 1]
        a /= 2 * self.__mca_data[self.position] - self.__mca_data[self.position + 1] - \
             self.__mca_data[self.position - 1]
        a /= 2
        ret += a
        return ret

    def peek_point(self):
        x = self.peek_location()
        y = self.__mca_data[int(x)]

        return x, y


class Derivative(PeekFinder):

    def __init__(self, mca, scan_range, level, dots):
        """
        导数法

        :param mca: MCA, MCA对象
        :param scan_range: (int index_satrt, int index_end)
        :param level: int, 阶数
        :param dots: int, 点数，一般取5， 7， 9， 11
        """
        super(Derivative, self).__init__(mca, scan_range)

        # Kb A0 A1 A2 A3 A4 A5
        # 注意一阶、三阶导数法参数呈中心对称

        self.__level_table = [[
            [], [],
            [12, 0, 8, -1, 0, 0, 0],
            [252, 0, 58, 67, -22, 0, 0],
            [1188, 0, 126, 193, 142, -86, 0],
            [5148, 0, 296, 503, 532, 294, -300]
        ], [
            [], [],
            [7, -2, -1, 2, 0, 0, 0],
            [42, -4, -3, 0, 5, 0, 0],
            [462, -20, -17, -8, 7, 28, 0],
            [429, -10, -9, -6, -1, 6, 15]
        ], [
            [], [],
            [2, 0, -2, 1, 0, 0, 0],
            [6, 0, -1, -1, 1, 0, 0]
        ]]

        self.m = dots // 2
        self.level = level

    def __get_derivatives(self):
        m = self.m
        level = self.level
        ret = np.zeros(len(self.mca), dtype=np.float64)
        for i, ele in enumerate(self.mca[m:-m]):
            ret[m + i] = ele * self.__level_table[level - 1][m][1]
            for i2 in range(m):
                ret[m + i] += self.mca[m + i + i2 + 1] * self.__level_table[level - 1][m][i2 + 2]
                if level % 2:
                    ret[m + i] += self.mca[m + i - i2 - 1] * self.__level_table[level - 1][m][i2 + 2] * -1
                else:
                    ret[m + i] += self.mca[m + i - i2 - 1] * self.__level_table[level - 1][m][i2 + 2]
            ret[m + 1] /= self.__level_table[level - 1][m][0]

        return ret

    def __validate(self, peek_location, edges, deri):
        deri_range = deri[edges[0]:edges[1]]
        n = max(deri_range) - min(deri_range)
        hight_base = self.mca[edges[0]] + self.mca[edges[1]]
        hight_base /= 2

        half_hight = self.mca[int(peek_location)] - hight_base
        half_hight /= 2
        half_hight += hight_base

        left_half = edges[0]
        right_half = edges[1]

        for i, ele in enumerate(self.mca[peek_location:edges[0] - 1:-1]):
            if ele <= half_hight:
                left_half = peek_location - i

        for i, ele in enumerate(self.mca[peek_location:edges[1]]):
            if ele <= half_hight:
                right_half = peek_location + i

        fwhm = right_half - left_half
        ret = 0.8 * fwhm <= n <= 3 * fwhm

        return ret

    def __find_edges(self, deri, peek_location):
        left_edge = self.range[0]
        right_edge = self.range[1]
        for i, ele in enumerate(deri[peek_location:self.m - 1:-1]):
            if ele >= 0 and deri[peek_location - i - 1] <= 0:
                k = ele - deri[peek_location - i - 1]
                if k:
                    # y = kx + b <=> b = y - kx <=> x = (y - b) / k
                    b = ele - k * (peek_location - i)
                    left_edge = (-b) / k
                else:
                    left_edge = peek_location - i - 0.5
        for i, ele in enumerate(deri[peek_location:self.m + 1]):
            if ele <= 0 and deri[peek_location + i + 1] >= 0:
                k = deri[peek_location + i + 1] - ele
                if k:
                    # y = kx + b <=> b = y - kx <=> x = (y - b) / k
                    b = ele - k * (peek_location + i)
                    right_edge = (-b) / k
                else:
                    right_edge = peek_location + i + 0.5
        return left_edge, right_edge

    def __find_peeks(self, deri):
        for i, ele in enumerate(deri[self.range[0]:self.range[1] - 1]):
            if ele >= 0 and deri[self.range[0] + i + 1] <= 0:
                k = deri[self.range[0] + i + 1] - ele
                if k:
                    b = ele - k * (self.range[0] + i)
                    peek = (-b) / k
                else:
                    peek = self.range[0] + i + 0.5

                edges = self.__find_edges(deri, peek)
                if self.__validate(peek, edges, deri):
                    self.peeks.append(Peek(peek, edges, self.mca))

    def search(self, ranges=None):
        super(Derivative, self).search(ranges)

        deri = self.__get_derivatives()
        self.__find_peeks(deri)

        self.flag_searched = True


class SimpleCompare(PeekFinder):

    def __init__(self, mca, scan_range, k, m):
        """
        简单比较法

        :param mca: MCA, MCA 谱线对象
        :param scan_range: (int index_satrt, int index_end)
        :param k: float, 找峰阈值，一般在1~1.5之间
        :param m: int, 寻峰宽度因子
        """

        super(SimpleCompare, self).__init__(mca, scan_range)
        self.k = k
        self.m = m

    def __find_peek(self, index, channel_data):
        peek = None
        if index >= len(channel_data - self.m):
            return peek

        for i, cnt in enumerate(channel_data[index + self.m:-self.m]):
            if not cnt:
                continue
            peek_value = cnt - (self.k * cnt ** -0.5)

            # print("[DEBUG] index: {}  peek: {}  peek_value: {}  last: {}  next: {}".format(
            #     i, cnt, peek_value, channel_data[index + i],
            #     channel_data[index + i + 2*self.m]))
            # print("[DEBUG] last_index: {}  current_index: {}  next_index: {}".format(
            #     index + i, index + i + self.m, index + i + 2*self.m))

            if channel_data[index + i + 2 * self.m] < peek_value and \
                    peek_value > channel_data[index + i]:
                peek = index + i + self.m
                break

        if peek is None:
            return peek

        maxi = max(channel_data[peek - self.m:peek + self.m])
        assert isinstance(channel_data, np.ndarray)
        peek = channel_data.tolist().index(maxi, peek - self.m, peek + self.m)

        return peek

    def __find_left(self, peek_index, channel_data):
        left_edge = channel_data[0]
        for i, cnt in enumerate(channel_data[peek_index:self.m - 1:-1]):
            if not cnt:
                left_edge = peek_index - i
                break
            cnt_v = cnt + (self.k * cnt ** -0.5)
            # print(cnt, cnt_v, channel_data[peek_index - i - self.m])
            if cnt_v <= channel_data[peek_index - i - self.m]:
                left_edge = peek_index - i
                break

        return left_edge

    def __find_right(self, peek_index, channel_data):
        right_edge = channel_data[-1]
        for i, cnt in enumerate(channel_data[peek_index:-self.m]):
            if not cnt:
                right_edge = peek_index - i
                break
            cnt_v = cnt + (self.k * cnt ** -0.5)
            if cnt_v <= channel_data[peek_index + i + self.m]:
                right_edge = peek_index + i
                break

        return right_edge

    def search(self, scan_range=None):
        super(SimpleCompare, self).search(scan_range)

        index = 0

        channel_data = self.mca[self.range[0]:self.range[1]]

        while index < len(channel_data) - 1:
            peek = self.__find_peek(index, channel_data)
            if peek is None:
                break
            right_edge = self.__find_right(peek, channel_data) + self.range[0]
            left_edge = self.__find_left(peek, channel_data) + self.range[0]
            index = peek

            edges = [left_edge, right_edge]

            self.peeks.append(Peek(peek + self.range[0], edges, self.mca))

        self.flag_searched = True
