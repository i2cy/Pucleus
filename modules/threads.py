#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: i2cy(i2cy@outlook.com)
# Project: main.py
# Filename: q_threads
# Created on: 2022/2/22

import numpy as np
from PyQt5.QtCore import QThread
from PyQt5.Qt import pyqtSignal
from .mca import MCA, Pulses
import time


class TestThread(QThread):

    def __init__(self):
        super(TestThread, self).__init__()

    def run(self):
        for i in range(100):
            print(i)
            time.sleep(0.5)


class PulseGenThread(QThread):
    open_later = pyqtSignal(MCA, str)

    def __init__(self, parent, filename=None, csp_rate=None,
                 measure_time=None, timed=None,
                 open_after=False):
        super(PulseGenThread, self).__init__()
        self.parent = parent
        self.logger = self.parent.logger
        self.filename = filename
        self.csp_rate = csp_rate
        self.total_time = measure_time
        self.timed = timed
        self.open_after = open_after
        self.curve = None
        self.file_unpack_dict = self.parent.file_unpack_dict

    def set_values(self, filename, csp_rate,
                   measure_time, timed,
                   open_after, curve
                   ):
        self.filename = filename
        self.csp_rate = csp_rate
        self.total_time = measure_time
        self.timed = timed
        self.open_after = open_after
        self.curve = curve

    def run(self):
        self.logger.INFO("[核脉冲模块] 正在生成和脉冲数据，请稍后")

        self.parent.flag_pulse_generating = True

        mca = self.curve
        mca = mca[self.file_unpack_dict["current_mca"]]
        assert isinstance(mca, MCA)
        if self.timed:
            pulse = mca.to_timed_pulses(csp_rate=self.csp_rate, total_time=self.total_time)
        else:
            pulse = mca.to_pulses(self.csp_rate * self.total_time)

        pulse.total_time = self.total_time
        if self.parent.flag_energyX_available:
            pulse.energyX_a = int(self.parent.K_energy_a * 1000000)
            pulse.energyX_b = int(self.parent.K_energy_b * 1000000)
        pulse.to_file(self.filename)

        if self.open_after:
            self.open_later.emit(MCA(self.filename), self.filename)

        self.parent.flag_pulse_generating = False

        self.logger.INFO("[核脉冲模块] 核脉冲数据已生成至文件\"{}\"".format(self.filename))


class UpdatePulseInfoThread(QThread):
    draw_pulse = pyqtSignal(tuple)
    draw_pulse_time = pyqtSignal(tuple)

    def __init__(self, parent):
        super(UpdatePulseInfoThread, self).__init__()
        self.parent = parent
        self.file_unpack_dict = self.parent.file_unpack_dict
        self.pulse_data = None
        self.pulse_time = None
        self.logger = self.parent.logger
        self.curve = None

    def static_convert_pulses(self, pulses):
        x = pulses.get_abs_time()
        y = pulses[:, 0]

        return x, y

    def static_convert_time_to_posibility(self, pulses, divs=1000):
        """

        :param pulses:
        :param divs:
        :return: x(毫秒), y概率
        """
        assert isinstance(pulses, Pulses)
        times = pulses.data[:, 1]
        assert isinstance(times, np.ndarray)
        time_p = np.zeros(divs, dtype=np.float64)
        t_max = times.max()
        dt = t_max / divs
        x = np.linspace(0, t_max / 1000, divs)

        for i in range(divs):
            condition_1 = dt * i < times
            condition_2 = times <= dt * (i + 1)
            time_p[i] += (condition_1 * condition_2).sum()

        time_p /= time_p.sum()
        return x, time_p

    def set_curve(self, curve):
        self.curve = curve

    def run(self):
        # draw pulse tab
        if self.curve is None:
            return
        pulse = self.curve[self.file_unpack_dict["pulse"]]
        if not isinstance(pulse, Pulses):
            # print("pulse is ", pulse)
            return

        if pulse.data.ndim != 2:
            # print("ndim is ", pulse.data.ndim)
            return

        self.logger.INFO("[核脉冲模块] 正在绘制核脉冲预览")

        # print(avg_time)
        self.pulse_data = self.static_convert_pulses(pulse)
        self.pulse_time = self.static_convert_time_to_posibility(pulse)

        self.draw_pulse.emit(self.pulse_data)
        self.draw_pulse_time.emit(self.pulse_time)
