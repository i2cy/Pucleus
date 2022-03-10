#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: i2cy(i2cy@outlook.com)
# Project: main.py
# Filename: libraries
# Created on: 2022/3/10

import xlrd
import csv
from .utils import IsFloat


class Nucleo(object):

    def __init__(self, data):
        self.name = data[2]
        self.cluster_prob = data[1]
        self.energy = data[0]


class Library(object):

    def __init__(self, filename):

        self.filename = filename.split("/")[-1].split("\\")[-1]
        file_type = filename.split(".")[-1]
        if file_type == "xls":
            self.workbook = xlrd.open_workbook(filename).sheet_by_index(0)
            self.workbook = [
                [
                    ele[0].value,
                    ele[1].value,
                    ele[2].value
                ] for ele in self.workbook
            ]
        else:
            f = open(filename, encoding="utf-8")
            self.workbook = csv.reader(f)
            self.workbook = [ele for ele in self.workbook]
            f.close()

        data_index = 0
        for i, ele in enumerate(self.workbook):
            cec = 0
            for e in ele:
                if IsFloat(str(e)):
                    cec += 1
            if cec:
                data_index = i
                break

        energy_index = 0
        prob_index = 1
        name_index = 2

        if data_index:
            for i, ele in enumerate(self.workbook[0]):
                if "能量" in ele or "energy" in ele.lower() or "feat" in ele.lower():
                    energy_index = i
                elif "分支" in ele or "比" in ele or "率" in ele or "rate" in ele.lower() or "cluster" in ele.lower():
                    prob_index = i
                else:
                    name_index = i

        else:
            data_raw = []

            for i in range(3):
                data_raw.append([str(ele[i]).replace("(", "") for ele in self.workbook])
                data_raw[i] = [ele[i].replace(")", "") for ele in data_raw[i]]
                if "%" in str(data_raw[i][0]):
                    prob_index = i
                elif IsFloat(str(data_raw[i][0])):
                    raw = [float(ele) for ele in data_raw[i]]
                    raw_min = min(raw)
                    raw_max = max(raw)
                    raw_mean = sum(raw) / len(raw)
                    if raw_max <= 1 and raw_min > 0:
                        prob_index = i
                    elif raw_mean > 1 and raw_min > 0:
                        energy_index = i
                else:
                    name_index = i

        if energy_index + prob_index + name_index != 3:
            raise Exception("无法加载核素库，缺少表头，无法通过数据统计值自动分辨每列元素。"
                            "库文件应当至少包含3列，分别是‘特征能量’、‘分支比’、‘核素名称’")

        data_raw = [
            [
                float(ele[energy_index]),
                float(str(ele[prob_index]).
                      replace("%", "").
                      replace("（", "").
                      replace("(", "").
                      replace("）", "").
                      replace(")", "")),
                str(ele[name_index])
            ] for ele in self.workbook[data_index:]
        ]

        self.__data = data_raw
        self.__data.sort(key=lambda x: (x[0], x[1], x[2].lower()))

        self.KE = (self.__data[-1][0] - self.__data[0][0]) / len(self.__data)

        for i, ele in enumerate(self.__data[:-1]):
            gap = self.__data[i + 1][0] - ele[0]
            if gap < self.KE and gap:
                self.KE = gap

        prob = [ele[1] for ele in self.__data]
        if max(prob) > 1:
            self.__data = [
                [
                    ele[0],
                    ele[1] / 100,
                    ele[2]
                ] for ele in self.__data
            ]

    def __str__(self):
        return self.summary()[0]

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, item):
        return self.__data[item]

    def __iter__(self):
        return self.__data.__iter__()

    def summary(self):
        basic = "{}  总数: {}  KE: {:.2f} KeV".format(self.filename, len(self.__data), self.KE)
        details = ""
        for i in self.__data:
            details += "{}: {:.2f} KeV({:.2f}%)\n".format(i[2], i[0], i[1] * 100)
        details = details[:-1]
        return basic, details

    def set_KE(self, KE):
        if KE > 10:
            KE = 10
        self.KE = KE

    def match(self, energy):
        res = []
        found = False
        for i, ele in enumerate(self.__data):
            gap = abs(energy - ele[0])
            if gap <= self.KE:
                res.append(Nucleo(ele))
                found = True
            else:
                if found:
                    break
        return res
