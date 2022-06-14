###############################################
# MPU6050 9-DoF Example Printout

from mpu9250_i2c import *

time.sleep(1) # delay necessary to allow mpu9250 to settle

print('recording data')
# while 1:
#     try:
#         ax,ay,az,wx,wy,wz = mpu6050_conv() # read and convert mpu6050 data
#         # mx,my,mz = AK8963_conv() # read and convert AK8963 magnetometer data
#     except:
#         continue
    
#     print('{}'.format('-'*30))
#     print('accel [g]: x = {0:2.2f}, y = {1:2.2f}, z {2:2.2f}= '.format(ax,ay,az))
#     print('gyro [dps]:  x = {0:2.2f}, y = {1:2.2f}, z = {2:2.2f}'.format(wx,wy,wz))
#     # print('mag [uT]:   x = {0:2.2f}, y = {1:2.2f}, z = {2:2.2f}'.format(mx,my,mz))
#     # print('{}'.format('-'*30))
    
#     time.sleep(0.01)


#!/usr/bin/python
"""
Update a simple plot as rapidly as possible to measure speed.
"""

import argparse
from collections import deque
from time import perf_counter

import numpy as np

import pyqtgraph as pg
import pyqtgraph.functions as fn
import pyqtgraph.parametertree as ptree
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

nsamples = 50
ax_list =[0]*nsamples
ay_list =[0]*nsamples
az_list =[0]*nsamples
gx_list =[0]*nsamples
gy_list =[0]*nsamples
gz_list =[0]*nsamples
readrate = 100

class MonkeyCurveItem(pg.PlotCurveItem):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.monkey_mode = ''

    def setMethod(self, param, value):
        self.monkey_mode = value

    def paint(self, painter, opt, widget):
        if self.monkey_mode not in ['drawPolyline']:
            return super().paint(painter, opt, widget)

        painter.setRenderHint(painter.RenderHint.Antialiasing, self.opts['antialias'])
        painter.setPen(pg.mkPen(self.opts['pen']))

        if self.monkey_mode == 'drawPolyline':
            painter.drawPolyline(fn.arrayToQPolygonF(self.xData, self.yData))

app = pg.mkQApp("Plot Speed Test")

default_pen = pg.mkPen()

pw = pg.PlotWidget()
pw.setRange(QtCore.QRectF(0, -5, nsamples, 10))
splitter = QtWidgets.QSplitter()
splitter.addWidget(pw)
splitter.show()

pw.setWindowTitle('pyqtgraph example: PlotSpeedTest')
pw.setLabel('bottom', 'Index', units='B')
curve = MonkeyCurveItem(pen=default_pen, brush='b')
pw.addItem(curve)

rollingAverageSize = 1000
elapsed = deque(maxlen=rollingAverageSize)

def resetTimings(*args):
    elapsed.clear()

# def makeData(*args):
#     global data, connect_array, ptr 
#     # sigopts = params.child('sigopts')
#     if sigopts['noise']:
#         data += np.random.normal(size=data.shape)
#     connect_array = np.ones(data.shape[-1], dtype=bool)
#     ptr = 0
#     pw.setRange(QtCore.QRectF(0, -10, nsamples, 20))

def onUseOpenGLChanged(param, enable):
    pw.useOpenGL(enable)

def onEnableExperimentalChanged(param, enable):
    pg.setConfigOption('enableExperimental', enable)

def onPenChanged(param, pen):
    curve.setPen(pen)

def onFillChanged(param, enable):
    curve.setFillLevel(0.0 if enable else None)

# params.child('sigopts').sigTreeStateChanged.connect(makeData)
# params.child('useOpenGL').sigValueChanged.connect(onUseOpenGLChanged)
# params.child('enableExperimental').sigValueChanged.connect(onEnableExperimentalChanged)
# params.child('pen').sigValueChanged.connect(onPenChanged)
# params.child('fill').sigValueChanged.connect(onFillChanged)
# params.child('plotMethod').sigValueChanged.connect(curve.setMethod)
# params.sigTreeStateChanged.connect(resetTimings)

# makeData()

fpsLastUpdate = perf_counter()
def update():
    global curve, data, ptr, elapsed, fpsLastUpdate, readrate
    global ax_list,ay_list,az_list,gx_list,gy_list,gz_list
    t_start = perf_counter()
    ax,ay,az,gx,gy,gz = mpu6050_conv() # read and convert mpu6050 data
    ax_list.append(ax)
    ay_list.append(ay)
    az_list.append(az)
    gx_list.append(gx)
    gy_list.append(gy)
    gz_list.append(gz)
    ax_list = ax_list[1:]
    ay_list = ay_list[1:]
    az_list = az_list[1:]
    gx_list = gx_list[1:]
    gy_list = gy_list[1:]
    gz_list = gz_list[1:]
    # Measure
    curve.setData(ax_list)
    app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
    t_end = perf_counter()
    time.sleep((1/readrate)-(t_end-t_start)) # desire - currentfps
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

if __name__ == '__main__':
    pg.exec()
