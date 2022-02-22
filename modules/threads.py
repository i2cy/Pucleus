#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: i2cy(i2cy@outlook.com)
# Project: main.py
# Filename: q_threads
# Created on: 2022/2/22

from PyQt5.QtCore import QThread
from PyQt5.Qt import pyqtSignal
from .mca import MCA, Pulses
import pyqtgraph as pg
import time


class TestThread(QThread):

    def __init__(self):
        super(TestThread, self).__init__()

    def run(self):
        for i in range(100):
            print(i)
            time.sleep(0.5)


class PulseGenThread(QThread):
    open_later = pyqtSignal()

    def __init__(self, parent, filename=None, csp_rate=None,
                 measure_time=None, timed=None,
                 open_after=False):
        super(PulseGenThread, self).__init__()
        self.parent = parent
        self.filename = filename
        self.csp_rate = csp_rate
        self.total_time = measure_time
        self.timed = timed
        self.open_after = open_after

    def set_values(self, filename, csp_rate,
                   measure_time, timed,
                   open_after
                   ):
        self.filename = filename
        self.csp_rate = csp_rate
        self.total_time = measure_time
        self.timed = timed
        self.open_after = open_after
        self.flag_pulse_generating = flag_pulse_generating
        self.logger = logger

    def run(self):
        if not self.parent.flag_file_opened:
            return

        if self.filename is None:
            return

        self.parent.flag_pulse_generating = True
        # print(filename)

        self.parent.logger.INFO("[pulse] 正在生成和脉冲数据，请稍后")

        mca = self.parent.static_get_current_curve()
        mca = mca[self.parent.file_unpack_dict["current_mca"]]
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
        print(pulse.total_time)

        if self.open_after:
            self.parent.static_add_file(MCA(self.filename), self.filename)

        self.parent.flag_pulse_generating = False

        self.parent.logger.INFO("[pulse] 核脉冲数据已生成至文件\"{}\"".format(self.filename))


class UpdatePulseInfoThread(QThread):
    draw = pyqtSignal(tuple)

    def __init__(self, parent):
        super(UpdatePulseInfoThread, self).__init__()
        self.parent = parent
        self.file_unpack_dict = self.parent.file_unpack_dict
        self.data = None
        self.logger = self.parent.logger
        self.curve = None

    def static_convert_pulses(self, pulses):
        x = pulses.get_abs_time()
        y = pulses[:, 0]

        return x, y

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

        self.logger.INFO("[pulse] 正在绘制核脉冲预览")

        # print(avg_time)
        self.data = self.static_convert_pulses(pulse)

        self.draw.emit(self.data)
