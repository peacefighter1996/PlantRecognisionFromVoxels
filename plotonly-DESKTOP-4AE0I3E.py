# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 20:05:59 2018

@author: Ian-A
"""
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui


def translation_matrix(direction):
    M = np.identity(4)
    M[:3, 3] = direction[:3]
    return M


def translation_from_matrix(matrix):
    return np.array(matrix, copy=False)[:3, 3].copy()


def box(size=(1, 1, 1)):
    import numpy as np
    vertices = np.zeros((8, 3))
    faces = np.zeros((12, 3), dtype=np.uint)
    xdim = size[0]
    ydim = size[1]
    zdim = size[2]
    vertices[0, :] = np.array([0, ydim, 0])
    vertices[1, :] = np.array([xdim, ydim, 0])
    vertices[2, :] = np.array([xdim, 0, 0])
    vertices[3, :] = np.array([0, 0, 0])
    vertices[4, :] = np.array([0, ydim, zdim])
    vertices[5, :] = np.array([xdim, ydim, zdim])
    vertices[6, :] = np.array([xdim, 0, zdim])
    vertices[7, :] = np.array([0, 0, zdim])

    faces = np.array([
        # bottom (clockwise, while looking from top)
        [2, 1, 0],
        [3, 2, 0],
        # sides (counter-clock-wise)
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
        # top (counter-clockwise)
        [4, 5, 6],
        [4, 6, 7]
    ],
        dtype=np.uint)

    return vertices, faces


class Window(QtGui.QWidget):
    def __init__(self, scatterplot, locs, frame, griddata):
        QtGui.QWidget.__init__(self)

        self.button1 = QtGui.QPushButton('error', self)
        self.button1.clicked.connect(self.error)
        self.button2 = QtGui.QPushButton('stem', self)
        self.button2.clicked.connect(self.stem)
        self.button3 = QtGui.QPushButton('leaf', self)
        self.button3.clicked.connect(self.leaf)
        self.button4 = QtGui.QPushButton('save', self)
        self.button4.clicked.connect(self.save)
        self.button5 = QtGui.QPushButton('skip', self)
        self.button5.clicked.connect(self.skip)
        self.glwindow = gl.GLViewWidget()
        self.glwindow2 = gl.GLViewWidget()
        self.glwindow.opts['distance'] = 40
        self.glwindow.setWindowTitle('GLScatterPlotItem')

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.addWidget(self.button1)
        self.layout.addWidget(self.button2)
        self.layout.addWidget(self.button3)
        self.layout.addWidget(self.button5)
        self.layout.addWidget(self.glwindow)
        self.layout.addWidget(self.glwindow2)
        self.layout.addWidget(self.button4)

        self.frame = frame
        self.i = 1
        self.locs = locs
        self.glwindow.addItem(scatterplot)
        self.glwindow.addItem(self.frame)
        T = translation_matrix(locs[0])
        loc = QtGui.QMatrix4x4(T.flatten())
        self.frame.setTransform(loc)
        self.glwindow.updateGL()
        self.griddata = griddata

        
        self.items = []
        self.secplot()

        self.errorlist = []
        self.leaflist = []
        self.stemlist = []

    def update(self, typeof):
        # print("here")
        data = self.griddata[self.i].split(',')
        # print(data)
        data2 = data[0].split(':')
        data[0] += '\n'
        # print("here2")
        for x in range(0, len(data2)):
            data2[x] = data2[x].split('.')
        # print("and here 2")
        if typeof is 0:
            print(f"error")
            self.errorlist.append(data[0])
        elif typeof is 1:
            print(f"leaf")
            self.leaflist.append(data[0])
        elif typeof is 2:
            print(f"stem")
            self.stemlist.append(data[0])
        self.i += 1
        T = translation_matrix(self.locs[self.i])
        loc = QtGui.QMatrix4x4(T.flatten())
        self.frame.setTransform(loc)
        self.glwindow.updateGL()
        self.secplot()
        self.glwindow2.show()

    def secplot(self):
        scale = 16  # equals to hexadecimal
        num_of_bits = 13
        self.frame1 = gl.GLAxisItem(antialias=True, glOptions='opaque')
        self.glwindow2.items = []
        self.glwindow2.addItem(self.frame1)
        data = self.griddata[self.i].split(',')

        data = data[0].split(':')
        for x in range(0, len(data)):
            data[x] = data[x].split('.')
            for y in range(0, len(data[x])):
                temp1 = bin(int(data[x][y], scale))[2:].zfill(num_of_bits)
                temp = []
                for k in range(0, 13):
                    temp.append(bool(int(temp1[k])))
                data[x][y] = temp

        joint_color = (1., 1., .4, 1)   # yellow

        data = np.array(data)
        self.items = []
        for x in range(0, 13):
            for y in range(0, 13):
                for z in range(0, 13):
                    if data[x][y][z]:
                        vertices, faces = box()
                        box1 = gl.GLMeshItem(vertexes=vertices,
                                             faces=faces,
                                             drawEdges=False,
                                             drawFaces=True,
                                             color=joint_color)
                        box1.setParentItem(self.frame1)
                        box1.translate(x-7.5, y-7.5, z-7.5)

                        self.items.append(box1)
                        self.glwindow2.addItem(self.items[-1])
        self.glwindow2.updateGL()

    def error(self):
        self.update(0)

    def leaf(self):
        self.update(1)

    def stem(self):
        self.update(2)

    def skip(self):
        self.update(-1)

    def save(self):
        errordata = open("error.txt", "w")
        for e in self.errorlist:
            errordata.write(e)
        errordata.close()
        errordata = open("leaf.txt", "w")
        for e in self.leaflist:
            errordata.write(e)
        errordata.close()
        errordata = open("stem.txt", "w")
        for e in self.stemlist:
            errordata.write(e)
        errordata.close()


voxelsize = 0.05

# file selection
filename = "scandata201810291313"
fileload = filename + ".txt"
filesave = filename + ".csv"
filesavecleaned = filename + "_cleansmall.csv"

# scan type
lidardata = open(filesavecleaned, "r").readlines()

saveinfile = open("grids.txt", "r").readlines()
locs = []
for i in range(0, len(saveinfile)):
    data = saveinfile[i].split(',')
    data = data[1].split(':')

    locs.append([float(data[0])/10,
                 float(data[1])/10,
                 float(data[2])/10])

nplidardata = []
for i in range(0, len(lidardata)):
    data = lidardata[i].split(',')
    nplidardata.append([float(data[0])/10,
                        float(data[1])/10,
                        float(data[2])/10])
nplidardata = np.array(nplidardata)


color = (1.0, 1.0, 1.0, 1.0)
sp1 = gl.GLScatterPlotItem(pos=nplidardata,
                           size=voxelsize,
                           color=color,
                           pxMode=False)
c = (1.0, 0.0, 0.0, 1.0)
boxsize = 13*2*voxelsize
i = 1
frame = gl.GLAxisItem(antialias=True, glOptions='opaque')

# visualisation
try:
    window.close()
    # app = QtGui.QApplication([])
    window = Window(sp1, locs, frame, saveinfile)
except NameError:
    app = QtGui.QApplication([])
    window = Window(sp1, locs, frame, saveinfile)
window.show()
