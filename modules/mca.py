#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: i2cy(i2cy@outlook.com)
# Project: main.py
# Filename: mca_reader
# Created on: 2021/11/20

import numpy as np
import time
import hashlib


class MCA(object):

    def __init__(self, data=None):

        data_raw_int = []
        data_t = []
        self.channels = 1024
        self.total_time = 0
        self.energyX_a = 0
        self.energyX_b = 0
        self.timestamp = 0

        if isinstance(data, str):

            self.from_file(data)

        elif isinstance(data, list) or isinstance(data, np.ndarray) or isinstance(data, tuple):
            data_t = data
            self.data = np.array(data_t, dtype=np.float64)
            self.channels = len(self.data)

        elif isinstance(data, MCA):
            self.channels = data.channels
            self.total_time = data.total_time
            self.energyX_a = data.energyX_a
            self.energyX_b = data.energyX_b
            self.timestamp = data.timestamp
            self.data = data.data.copy()

        else:
            data_t = np.zeros(1024, dtype=np.float64)
            self.data = np.array(data_t, dtype=np.float64)
            self.channels = len(self.data)

    def __call__(self, *args, **kwargs):
        return self.data

    def __add__(self, other):
        if isinstance(other, MCA):
            ret = self.data + other.data
        else:
            ret = self.data + other
        return MCA(ret)

    def __sub__(self, other):
        if isinstance(other, MCA):
            ret = self.data - other.data
        else:
            ret = self.data - other
        return MCA(ret)

    def __mul__(self, other):
        if isinstance(other, MCA):
            ret = self.data * other.data
        else:
            ret = self.data * other
        return MCA(ret)

    def __truediv__(self, other):
        if isinstance(other, MCA):
            ret = self.data / other.data
        else:
            ret = self.data / other
        return MCA(ret)

    def __floordiv__(self, other):
        if isinstance(other, MCA):
            ret = self.data // other.data
        else:
            ret = self.data // other
        return MCA(ret)

    def __mod__(self, other):
        if isinstance(other, MCA):
            ret = self.data % other.data
        else:
            ret = self.data % other
        return MCA(ret)

    def __pow__(self, power, modulo=None):
        ret = self.data.__pow__(power, modulo)
        return MCA(ret)

    def __iadd__(self, other):
        if isinstance(other, MCA):
            self.data += other.data
        else:
            self.data += other
        return self

    def __isub__(self, other):
        if isinstance(other, MCA):
            self.data -= other.data
        else:
            self.data -= other
        return self

    def __imul__(self, other):
        if isinstance(other, MCA):
            self.data *= other.data
        else:
            self.data *= other
        return self

    def __itruediv__(self, other):
        if isinstance(other, MCA):
            self.data /= other.data
        else:
            self.data /= other
        return self

    def __ifloordiv__(self, other):
        if isinstance(other, MCA):
            self.data //= other.data
        else:
            self.data //= other
        return self

    def __imod__(self, other):
        if isinstance(other, MCA):
            self.data %= other.data
        else:
            self.data %= other
        return self

    def __ipow__(self, other, **kwargs):
        self.data.__ipow__(other, **kwargs)

    def __str__(self):
        return self.data.__str__()

    def __getitem__(self, item):
        return self.data[item]

    def __iter__(self):
        return self.data.__iter__()

    def __setitem__(self, key, value):
        self.data[key] = value

    def __len__(self):
        return len(self.data)

    def as_numpy(self):
        return self.data

    def copy(self):
        return MCA(self)

    def sum(self):
        return np.sum(self.data)

    def max(self):
        return np.max(self.data)

    def min(self):
        return np.min(self.data)

    def mean(self):
        return np.mean(self.data)

    def std(self):
        return np.std(self.data)

    def from_file(self, filename):
        if filename[-3:] in ("mca", "txt"):
            f = open(filename, 'r')
            data_raw = []
            data = f.readlines()

            for i, ele in enumerate(data):
                while len(ele) > 1 and ele[-1] in ("\r", "\n"):
                    ele = ele[:-1]
                if ele:
                    data_raw.append(int(ele))

            data_t = data_raw[:-1]

            for i in range(5):
                data_t[i] = 0

            self.data = np.array(data_t, dtype=np.float64)
            self.channels = len(self.data)

        elif filename[-3:] == "chn":
            f = open(filename, 'rb')
            data_raw = f.read()
            args = data_raw[:32]
            data_raw = data_raw[32:]
            data_t = []
            for i, e in enumerate(data_raw[::4]):
                data_t.append(int.from_bytes(data_raw[4 * i:4 * (i + 1)], "little", signed=False))

            self.data = np.array(data_t, dtype=np.float64)
            self.channels = len(self.data)

        elif filename[-3:] == "tps":
            pulse = Pulses(filename)
            self.from_pulses(pulse)

        elif filename[-3:] == "tch":
            with open(filename, "rb") as f:
                data = f.read()
                f.close()
            head = data[0:3]  # 3B 文件头标识CHN
            if head == b"CHN":
                head = head.decode()
            else:
                raise TypeError("错误的文件类型")

            ts = data[3:9]  # 6B 时间戳（单位ms）
            self.timestamp = int().from_bytes(ts, "little", signed=False)

            channels = data[9:11]  # 2B channel分辨率
            self.channels = int().from_bytes(channels, "little", signed=False)

            total_time = data[11:14]  # 3B 测量时间（单位s）
            self.total_time = int().from_bytes(total_time, "little", signed=False)

            enX_a = data[14:18]  # 4B 能量刻度参数A
            self.energyX_a = int().from_bytes(enX_a, "little", signed=True) / 1000000

            enX_b = data[18:22]  # 4B 能量刻度参数B
            self.energyX_b = int().from_bytes(enX_b, "little", signed=True) / 1000000

            sha = data[22:54]  # 32B 全文除此字段的sha256校验和
            hasher = hashlib.sha256()
            hasher.update(data[:22])
            hasher.update(data[54:])
            if sha != hasher.digest():
                raise Exception("文件损坏")

            data = data[54:]  # 4N*B 脉冲数据
            self.data = np.frombuffer(data, dtype=np.uint32)

    def to_accumulation(self):
        sum = []
        sum_raw = 0
        for i in self.data:
            sum_raw += i
            sum.append(sum_raw)
        ret = np.array(sum, dtype=np.float64)
        return MCA(ret)

    def to_probDensity(self):
        sum = self.to_accumulation()
        ret = sum / self.sum()
        return ret

    def to_density(self):
        return self / self.sum()

    def to_file(self, filename):
        if "." not in filename:
            filename += ".tch"

        data = b""

        if filename[-3:] == "tch":  # 自定义TCH能谱文件格式
            head = "CHN"
            head = head.encode()
            data += head

            ts = self.timestamp.to_bytes(6, "little", signed=False)
            data += ts

            channels = self.channels.to_bytes(2, "little", signed=False)
            data += channels

            total_time = self.total_time
            total_time = total_time.to_bytes(3, "little", signed=False)
            data += total_time

            enX_a = self.energyX_a.to_bytes(4, "little", signed=False)
            data += enX_a

            enX_b = self.energyX_b.to_bytes(4, "little", signed=False)
            data += enX_b

            data_raw = self.data.astype(np.uint32).tobytes()

            sha = hashlib.sha256()
            sha.update(data)
            sha.update(data_raw)
            sha = sha.digest()
            data += sha

            data += data_raw

        with open(filename, "wb") as f:
            ret = f.write(data)
            f.close()

        return ret

    def to_pulses(self, total_pulses=None):
        if total_pulses is None:
            total_pulses = int(self.sum())
        np.random.seed(int(time.time()))
        accu = self.to_accumulation()
        rand = np.random.randint(0, accu.max(), size=(total_pulses,))
        res = np.zeros(total_pulses, dtype=np.int32)
        for ele in accu:
            t = rand >= ele
            res += t

        res = Pulses(res)
        res.total_time = self.total_time
        res.energyX_a = self.energyX_a
        res.energyX_b = self.energyX_b
        res.timestamp = int(time.time() * 1000)
        res.channels = self.channels

        return res

    def to_timed_pulses(self, csp_rate=2000, total_time=200):
        pulses = self.to_pulses(int(csp_rate * total_time))
        pulses = pulses.data

        time_x = np.linspace(0, -np.log(10**-9) / csp_rate, 8193, dtype=np.float64)
        time_x0 = time_x[0: 8192].copy()
        time_x1 = time_x[1: 8193].copy()

        time_p = np.exp(-csp_rate * time_x0) - np.exp(-csp_rate * time_x1)

        time_p_sum = np.zeros(len(time_p), dtype=np.float64)

        inter = 0
        for i, ele in enumerate(time_p):
            inter += ele
            time_p_sum[i] = inter

        rand = np.random.random(len(pulses))

        time_res_index = np.zeros(len(pulses), dtype=np.uint32)

        for i in time_p_sum:
            t = rand >= i
            time_res_index += t

        #print(len(time_p), len(time_p_sum), np.max(time_res_index), np.min(time_res_index))
        time_x0 *= 10**6

        time_res = [int(time_x0[ele]) for ele in time_res_index]

        #print(pulses.ndim, pulses.ndim)
        res = np.dstack((pulses, time_res)).astype(np.uint32)
        res = np.squeeze(res)

        return Pulses(res)

    def from_pulses(self, pulses, max_channel=1024):
        if isinstance(pulses, list) or isinstance(pulses, tuple) or isinstance(pulses, np.ndarray):
            res = np.zeros(max_channel, dtype=np.int32)
            for ele in pulses:
                res[ele] += 1

            self.data = res.astype(np.float64)

        elif isinstance(pulses, Pulses):
            self.channels = pulses.channels
            self.total_time = pulses.total_time
            self.energyX_a = pulses.energyX_a
            self.energyX_b = pulses.energyX_b
            self.timestamp = pulses.timestamp
            res = np.zeros(self.channels, dtype=np.int32)

            if pulses.data.ndim == 1:
                for ele in pulses:
                    res[ele] += 1
            elif pulses.data.ndim == 2:
                for ele in pulses.data[:, 0]:
                    res[ele] += 1

            self.data = res.astype(np.float64)

        elif isinstance(pulses, str):
            filename = pulses
            pulses = Pulses(filename)
            self.from_pulses(pulses)

        return self


class Pulses(object):

    def __init__(self, data=None):
        self.data = data
        self.channels = 1024
        self.total_time = 0
        self.energyX_a = 0
        self.energyX_b = 0
        self.timestamp = 0

        if isinstance(data, str):
            filename = data
            self.from_file(filename)

    def __iter__(self, **kwargs):
        return self.data.__iter__(**kwargs)

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def get_abs_time(self):
        if self.data.ndim == 1:
            return
        time_i = 0
        time_abs = np.zeros(len(self.data), dtype=np.float64)
        for i, ele in enumerate(self.data[:, 1]):
            time_abs[i] = time_i
            time_i += ele / 10**6

        return time_abs

    def from_file(self, filename):
        if filename[-3:] == "tps":  # 自定义的核脉冲数据文件
            with open(filename, "rb") as f:
                data = f.read()
                f.close()
            head = data[0:3]  # 3B 文件头标识CHP
            if head in (b"CHP", b"CHT"):
                head = head.decode()
            else:
                raise TypeError("错误的文件类型")

            ts = data[3:9]  # 6B 时间戳（单位ms）
            self.timestamp = int().from_bytes(ts, "little", signed=False)

            channels = data[9:11]  # 2B channel分辨率
            self.channels = int().from_bytes(channels, "little", signed=False)

            total_time = data[11:14]  # 3B 测量时间（单位s）
            self.total_time = int().from_bytes(total_time, "little", signed=False)

            enX_a = data[14:18]  # 4B 能量刻度参数A
            self.energyX_a = int().from_bytes(enX_a, "little", signed=True) / 1000000

            enX_b = data[18:22]  # 4B 能量刻度参数B
            self.energyX_b = int().from_bytes(enX_b, "little", signed=True) / 1000000

            sha = data[22:54]  # 32B 全文除此字段的sha256校验和
            hasher = hashlib.sha256()
            hasher.update(data[:22])
            hasher.update(data[54:])
            if sha != hasher.digest():
                raise Exception("文件损坏")

            data = data[54:]  # 4N*B 脉冲数据
            self.data = np.frombuffer(data, dtype=np.uint32)
            length = len(self.data)
            if head == "CHT":
                self.data = np.reshape(self.data, (length // 2, 2))

            return self

    def to_mca(self):
        return MCA().from_pulses(self)

    def to_file(self, filename):
        data = b""

        if self.data.ndim == 1:
            head = "CHP"
        elif self.data.ndim == 2:
            head = "CHT"
        else:
            head = "UNO"
        head = head.encode()
        data += head

        ts = self.timestamp.to_bytes(6, "little", signed=False)
        data += ts

        channels = self.channels.to_bytes(2, "little", signed=False)
        data += channels

        total_time = self.total_time
        if head == b"CHT":
            total_time = self.data[-1, 1] / 1000
            if total_time % 1:
                total_time = total_time // 1 + 1
        total_time = int(total_time)
        total_time = total_time.to_bytes(3, "little", signed=False)
        data += total_time

        enX_a = self.energyX_a.to_bytes(4, "little", signed=False)
        data += enX_a

        enX_b = self.energyX_b.to_bytes(4, "little", signed=False)
        data += enX_b

        data_raw = self.data.astype(np.uint32).tobytes()

        sha = hashlib.sha256()
        sha.update(data)
        sha.update(data_raw)
        sha = sha.digest()
        data += sha

        data += data_raw

        if "." not in filename or filename.split(".")[-1] != "tps":
            filename += ".tps"

        with open(filename, "wb") as f:
            ret = f.write(data)
            f.close()

        return ret
