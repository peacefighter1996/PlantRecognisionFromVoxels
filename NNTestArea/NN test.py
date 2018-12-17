# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 16:02:47 2018

@author: Ian-A

torch project
"""
# from __future__ import print_function


from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg

import torch
import torch.nn as nn #import torch neural network library
import torch.nn.functional as F #import functional neural network module
import numpy as np
import time
import json
import sys
import torch.optim as optim
import random

class Window(QtGui.QWidget):
    def __init__(self, scatterplots):
        QtGui.QWidget.__init__(self)

        self.glwindow = gl.GLViewWidget()
        self.glwindow.opts['distance'] = 40
        self.glwindow.setWindowTitle('GLScatterPlotItem')

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.addWidget(self.glwindow)
        

        for x in range(0, len(scatterplots)):
            self.glwindow.addItem(scatterplots[x])

        self.glwindow.updateGL()

""""

"""

def clear_data(coordinates, minimum_ammount=1):
    """
    Voxel based data clearing for pointclouds
        :param coordinates: pointcloud that is going to be voxelized
        :param plotsize: size of the coordinate system a list of 6 items.\n
            x minimum: plotsize[0]\n
            x maximum: plotsize[1]\n
            y minimum: plotsize[2]\n
            y maximum: plotsize[3]\n
            z minimum: plotsize[4]\n
            z maximum: plotsize[5]
        :param minimum_ammount=1: minimum amount of points to consider a voxel
            point.
    Output:
        list of the voxel locations
    .. note::

        this product is still in development
    """
    plotsize = []
    plotsize.append(coordinates[0][0])
    plotsize.append(coordinates[0][0])
    plotsize.append(coordinates[0][1])
    plotsize.append(coordinates[0][1])
    plotsize.append(coordinates[0][2])
    plotsize.append(coordinates[0][2])
    for n in range(1, len(coordinates)):
        for x in range(0, 3):
            if coordinates[n][x] < plotsize[x*2]:
                plotsize[x*2] = coordinates[n][x]
            elif coordinates[n][x] > plotsize[x*2+1]:
                plotsize[x*2+1] = coordinates[n][x]
    print(plotsize)
    areashape = (int((plotsize[1] - plotsize[0]))+1,
                 int((plotsize[3] - plotsize[2]))+1,
                 int((plotsize[5] - plotsize[4]))+1)
    mode = 0

    if (minimum_ammount == 1):
        voxels = np.full(areashape,
                         False,
                         dtype=int)
        mode = 0
    elif (minimum_ammount >= 2):
        voxels = np.full(areashape,
                         0,
                         dtype=int)
        mode = 1
    else:
        raise ValueError('mimimum_ammount cant be lower then 1')

    newcoordinates = []

    for coor in coordinates:
        x = int((coor[0] - plotsize[0]))
        y = int((coor[1] - plotsize[2]))
        z = int((coor[2] - plotsize[4]))
        
        if(mode == 0):
            if (voxels[x][y][z]==0):
                voxels[x][y][z] = 1
                newcoordinates.append([x, y, z])

        if(mode == 1):
            if (voxels[x][y][z] >= 0):
                
                voxels[x][y][z] += 1
                if (voxels[x][y][z] >= minimum_ammount):
                    newcoordinates.append([x, y, z])
                    voxels[x][y][z] = -1


    return voxels

class DeepNeuralNetwork(nn.Module):
    def __init__(self,u):
        super(DeepNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(2197, int(u*2.5))
        self.fc2 = nn.Linear(int(u*2.5), u*2)
        self.fc3 = nn.Linear(u*2, u)
        self.fc4 = nn.Linear(u, 3)
        self.ReLu = nn.ReLU()
        self.Sigmoid = nn.Sigmoid() 

    def forward(self, x): #feed forward
        #self.layer1 = x.view(-1, 2197) #make it flat in one dimension from 0 - 784
        self.layer1 = x
        # print(layer1)
        self.layer2 = self.ReLu(self.fc1(self.layer1))
        layer3 = self.Sigmoid(self.fc2(self.layer2))
        layer4 = self.Sigmoid(self.fc3(layer3))
        layer5 = self.Sigmoid(self.fc4(layer4))
        return layer5
    # F.log_softmax(layer4)
    def __repr__(self):
        return json.dumps(self.__dict__)

# create your optimizer


dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
types = ["stemT.txt", "leafT.txt", "errorT.txt"]
convel = [[], [], []]
scale = 16  # equals to hexadecimal
num_of_bits = 13
for t in range(0, len(types)):
    saveinfile = open(types[t], "r").readlines()
    del saveinfile[-1]
    locs = []
    for i in range(0, len(saveinfile)):
        data = saveinfile[i].split(':')
        for x in range(0, len(data)):
            data[x] = data[x].split('.')
            for y in range(0, len(data[x])):
                temp1 = bin(int(data[x][y], scale))[2:].zfill(num_of_bits)
                temp = []
                for k in range(0, 13):
                    temp.append(float(int(temp1[k])))
                data[x][y] = temp

        joint_color = (1., 1., .4, 1)   # yellow
        data = np.array(data).flatten()
        convel[t].append(data)

y_out = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

x_train = []
y_train = []

x_test = []
y_test = []
for y in range(0,3):
    samples = random.sample(range(1, len(convel[y])), 150)
    for x in samples:
        x_train.append(convel[y][x])
        y_train.append(y_out[y])
for y in range(0,3):
    for x in range(0,len(convel[y])):
        x_test.append(convel[y][x])
        y_test.append(y_out[y])

x = torch.tensor(np.array(x_train), dtype=torch.float, requires_grad=False, device= device)
y = torch.tensor(np.array(y_train), dtype=torch.float, requires_grad=False, device= device)
xt = torch.tensor(np.array(x_test), dtype=torch.float, requires_grad=False, device= device)
yt = torch.tensor(np.array(y_test), dtype=torch.float, requires_grad=False, device= device)
# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.

print("start")
train = True
if train:
    model = DeepNeuralNetwork(100).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    # The nn package also contains definitions of popular loss functions; in this
    # case we will use Mean Squared Error (MSE) as our loss function.
    # loss_fn = torch.nn.KLDivLoss(size_average=False)
    progress = [[],[]]
    criterion = nn.MSELoss()
    n = 20000
    time1 = time.time()
    for t in range(n):
        optimizer.zero_grad()   # zero the gradient buffers
        output = model(xt)
        loss = criterion(output, yt)
        #print(model.fc1.bias.grad)
        if (t%100==0):
            print("{:3.1f}%".format(t/n*100), flush=True, end=" ")
            progress[0].append(loss.item())
            output2 = model(xt)
            loss2 = criterion(output2, yt)
            progress[1].append(loss2.item())
    
        loss.backward()
        #print(f"after:{model.fc1.bias.grad}")
        optimizer.step() 
    print(time.time()-time1)

progressplot = False
if progressplot:
    try:
        mw.close()
        mw = QtGui.QMainWindow()
    except NameError:
        app = QtGui.QApplication([])
        mw = QtGui.QMainWindow()
    progress = np.array(progress)
    mw.resize(800,800)
    view = pg.GraphicsLayoutWidget()  ## GraphicsView with GraphicsLayout inserted by default
    mw.setCentralWidget(view)
    mw.show()
    mw.setWindowTitle('pyqtgraph example: ScatterPlot')
    
    w1 = view.addPlot()
    s1 = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
    pos = []
    n=len(progress[0])
    for k in range(0,n):
        pos.append(np.array([k,progress[0][k]]))
    pos1 = np.array(pos)
    pos = []
    for k in range(0,n):
        pos.append(np.array([k,progress[1][k]]))
    pos2 = np.array(pos)
    spots = [{'pos': pos1.T[:,i], 'data': 1} for i in range(n)]
    spots2 = [{'pos': pos2.T[:,i], 'data': 1} for i in range(n)]
    s2 = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 255, 120))
    s1.addPoints(spots)
    s2.addPoints(spots2)
    w1.addItem(s1)
    w1.addItem(s2)
    mw.show()

# program 2
if progressplot == False:
    voxelsize = 0.5
    
    # file selection
    filename = "scandata201810291313"
    fileload = filename + ".txt"
    filesave = filename + ".csv"
    filesavecleaned = filename + "_clean.csv"
    
    # scan type
    lidardata = open(filesavecleaned, "r").readlines()
    
    nplidardata = []
    for i in range(0, len(lidardata)):
        data = lidardata[i].split(',')
        nplidardata.append([float(data[0]),
                            float(data[1]),
                            float(data[2])])
    nplidardata = np.array(nplidardata)
    
    voxels = clear_data(nplidardata)
    
    areashape = voxels.shape
    
    gridsize = 13
    delta = int((gridsize+1)/2)
    areas = []
    
    pointlist = [[], [], [], []]
    testdataloc= []
    testdata = []
    for x in range(delta, areashape[0] - delta):
        for y in range(delta, areashape[1] - delta):
            for z in range(delta, areashape[2] - delta):
                if (voxels[x][y][z] == 1 or voxels[x][y][z] == -1):
                    # print("yeah")
                    testdata.append(np.array(voxels[x-delta:x+delta-1,
                                                 y-delta:y+delta-1,
                                                 z-delta:z+delta-1]).flatten())
                    testdataloc.append([x, y, z])
                    # print([x, y, z])
    data = torch.tensor(np.array(testdata),
                        dtype=torch.float,
                        requires_grad=False,
                        device=device)
    output1 = model(data)
    result = output1.cpu().detach().numpy()
    # print(result)
    for x in range(0, len(result)):
        if result[x][0] > 0.75:
            pointlist[0].append(testdataloc[x])
        elif result[x][1] > 0.75:
            pointlist[1].append(testdataloc[x])
        elif result[x][2] > 0.75:
            pointlist[2].append(testdataloc[x])
        else:
            pointlist[3].append(testdataloc[x])
                    # print(area)
                    
    x = 0
    sp=[]
    color= [(0.0, 1.0, 1.0, 1.0),
            (0.0, 1.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 1.0),
            (1.0, 1.0, 1.0, 1.0)]
    for x in range (0,len(pointlist)):
        temppoints = np.array(pointlist[x])
        sp.append(gl.GLScatterPlotItem(pos=temppoints,
                                       size=voxelsize,
                                       color=color[x],
                                       pxMode=False))
    
    try:
        window.close()
        # app = QtGui.QApplication([])
        window = Window(sp)
    except NameError:
        app = QtGui.QApplication([])
        window = Window(sp)
    window.show()

# Start Qt event loop unless running in interactive mode.
"""
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
"""
