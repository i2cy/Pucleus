#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: i2cy(i2cy@outlook.com)
# Project: main.py
# Filename: utils
# Created on: 2021/11/21

import random
import colorsys
import re
from .mca import MCA
from .energy_axis import linear_regression
import numpy as np
import pyqtgraph as pg
from i2cylib import Logger
from pyqtgraph import TargetItem
from PyQt5.Qt import pyqtSignal
from PyQt5 import QtCore


class ModTargetItem(TargetItem):

    clicked = pyqtSignal(int)

    def __init__(self,
                 pos=None,
                 size=10,
                 radii=None,
                 symbol="crosshair",
                 pen=None,
                 hoverPen=None,
                 brush=None,
                 hoverBrush=None,
                 movable=True,
                 label=None,
                 labelOpts=None,
                 item_ID=0
                 ):
        """
        Parameters
        ----------
        pos : list, tuple, QPointF, QPoint, Optional
            Initial position of the symbol.  Default is (0, 0)
        size : int
            Size of the symbol in pixels.  Default is 10.
        radii : tuple of int
                    Deprecated.  Gives size of crosshair in screen pixels.
        pen : QPen, tuple, list or str
            Pen to use when drawing line. Can be any arguments that are valid
            for :func:`~pyqtgraph.mkPen`. Default pen is transparent yellow.
        brush : QBrush, tuple, list, or str
            Defines the brush that fill the symbol. Can be any arguments that
            is valid for :func:`~pyqtgraph.mkBrush`. Default is transparent
            blue.
        movable : bool
            If True, the symbol can be dragged to a new position by the user.
        hoverPen : QPen, tuple, list, or str
            Pen to use when drawing symbol when hovering over it. Can be any
            arguments that are valid for :func:`~pyqtgraph.mkPen`. Default pen
            is red.
        hoverBrush : QBrush, tuple, list or str
            Brush to use to fill the symbol when hovering over it. Can be any
            arguments that is valid for :func:`~pyqtgraph.mkBrush`. Default is
            transparent blue.
        symbol : QPainterPath or str
            QPainterPath to use for drawing the target, should be centered at
            ``(0, 0)`` with ``max(width, height) == 1.0``.  Alternatively a string
            which can be any symbol accepted by
            :func:`~pyqtgraph.ScatterPlotItem.setSymbol`
        label : bool, str or callable, optional
            Text to be displayed in a label attached to the symbol, or None to
            show no label (default is None). May optionally include formatting
            strings to display the symbol value, or a callable that accepts x
            and y as inputs.  If True, the label is ``x = {: >.3n}\ny = {: >.3n}``
            False or None will result in no text being displayed
        labelOpts : dict
            A dict of keyword arguments to use when constructing the text
            label. See :class:`TargetLabel` and :class:`~pyqtgraph.TextItem`
        """

        super(ModTargetItem, self).__init__(pos=pos,
                                            size=size,
                                            radii=radii,
                                            symbol=symbol,
                                            pen=pen,
                                            hoverPen=hoverPen,
                                            brush=brush,
                                            hoverBrush=hoverBrush,
                                            movable=movable,
                                            label=label,
                                            labelOpts=labelOpts)

        self.itemID = item_ID

    def mouseClickEvent(self, ev):
        super(ModTargetItem, self).mouseClickEvent(ev)
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            self.clicked.emit(self.itemID)


class ModLogger:

    def __init__(self, level="DEBUG"):
        self.logger = Logger(level=level, echo=False)
        self.log_buffer = []
        self.qlabel_object = None

        self.ui_object = None

    def bind_QLabel(self, ui_object):
        self.ui_object = ui_object
        self.qlabel_object = ui_object.label_logger

    def export_logs(self, filename):
        """
        save logs to file

        :param filename: str, target log path
        :return: bool, action status
        """
        status = False
        try:
            with open(filename, "w") as f:
                for i in self.log_buffer:
                    f.write(i)
                f.close()
            status = True
        except Exception:
            pass

        return status

    def DEBUG(self, msg):
        ret = self.logger.DEBUG(msg)
        if self.qlabel_object is not None:
            self.qlabel_object.setText(msg)
        self.log_buffer.append(ret)
        return ret

    def INFO(self, msg):
        ret = self.logger.INFO(msg)
        if self.qlabel_object is not None:
            self.qlabel_object.setText(msg)
        self.log_buffer.append(ret)
        return ret

    def WARNING(self, msg):
        ret = self.logger.WARNING(msg)
        if self.qlabel_object is not None:
            self.qlabel_object.setText(msg)
        self.log_buffer.append(ret)
        return ret

    def ERROR(self, msg):
        ret = self.logger.ERROR(msg)
        if self.qlabel_object is not None:
            self.qlabel_object.setText(msg)
        self.log_buffer.append(ret)
        return ret

    def CRITICAL(self, msg):
        ret = self.logger.CRITICAL(msg)
        if self.qlabel_object is not None:
            self.qlabel_object.setText(msg)
        self.log_buffer.append(ret)
        return ret


class ColorManager(object):

    def __init__(self):

        self.base = []
        self.gen_color(16)

        self.index = 0

    def __getitem__(self, item):
        return self.gen_color(item)

    def __len__(self):
        return len(self.base)

    def get_color(self, index=None):
        ret = None
        if index is None:
            if self.index >= len(self):
                self.index = 0
            index = self.index
            self.index += 1
        ret = colorsys.hls_to_rgb(*self.base[index])
        ret = [int(255) * ele for ele in ret]
        return ret

    def gen_color(self, num):
        hls_colors = []
        i = 0
        step = 360.0 / num
        while i < 360:
            h = i
            s = 90 + random.random() * 10
            l = 50 + random.random() * 10
            t_hlsc = [h / 360.0, l / 100.0, s / 100.0]
            hls_colors.append(t_hlsc)
            i += step

        random.shuffle(hls_colors)
        self.base = hls_colors


class Mod_PlotWidget(pg.PlotWidget):

    def __init__(self, *key, **kwargs):
        super(Mod_PlotWidget, self).__init__(*key, **kwargs)

        self.post_mousePressEvent_handler = self.void_func
        self.post_mouseReleaseEvent_handler = self.void_func

        self.post_keyPressEvent = self.void_func
        self.post_keyReleaseEvent = self.void_func

    def void_func(self, *key, **kwargs):
        pass

    def bind_mousePressEvent(self, func):
        self.post_mousePressEvent_handler = func

    def bind_mouseReleaseEvent(self, func):
        self.post_mouseReleaseEvent_handler = func

    def bind_keyPressEvent(self, func):
        self.post_keyPressEvent = func

    def bind_keyReleaseEvent(self, func):
        self.post_keyReleaseEvent = func

    def mousePressEvent(self, ev):
        super(Mod_PlotWidget, self).mousePressEvent(ev)
        # print("pressed,", ev.pos())
        self.post_mousePressEvent_handler(ev)

    def mouseReleaseEvent(self, ev):
        super(Mod_PlotWidget, self).mouseReleaseEvent(ev)
        # print("released", ev.pos())
        self.post_mouseReleaseEvent_handler(ev)

    def keyPressEvent(self, ev):
        super(Mod_PlotWidget, self).keyPressEvent(ev)
        # print(ev)
        self.post_keyPressEvent(ev)

    def keyReleaseEvent(self, ev):
        super(Mod_PlotWidget, self).keyReleaseEvent(ev)
        # print(ev)
        self.post_keyReleaseEvent(ev)


def get_R_square(original_plot, current_plot):
    """
    linear_regression with R^2

    :param original_plot:
    :param current_plot:
    :return:
    """
    assert isinstance(original_plot, MCA)
    assert isinstance(current_plot, MCA)
    original_data = original_plot.as_numpy()
    original_data.sort()
    current_data = current_plot.as_numpy()
    current_data.sort()
    b, a = linear_regression(original_data, current_data)

    rr = np.sum(((original_data * a + b) - np.mean(current_data)) ** 2) \
         / np.sum((current_data - np.mean(current_data)) ** 2)

    return rr, a, b


# 判断一个字符串是否是小数
def IsFloat(s):
    pattern = '^-?\d+\.?\d*$'  # 匹配数字: 从头开始匹配 -0或1次 数字1或多次 .0或1次 数字0或多次 匹配到字符串末尾
    match = re.match(pattern, s)
    return match != None
