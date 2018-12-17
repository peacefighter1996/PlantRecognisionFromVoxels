# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 16:02:47 2018

@author: Ian-A

torch project
"""
from __future__ import print_function


from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg

import torch
import torch.nn as nn #import torch neural network library
import torch.nn.functional as F #import functional neural network module
import numpy as np
import time
import json

import torch.optim as optim




try:
    mw.close()
    mw = QtGui.QMainWindow()
except NameError:
    app = QtGui.QApplication([])
    mw = QtGui.QMainWindow()
    


class DeepNeuralNetwork(nn.Module):
    def __init__(self,u):
        super(DeepNeuralNetwork, self).__init__() #load super class for training data
        self.fc1 = nn.Linear(1, u) #defining fully connected with input 2 and output 3
        self.fc2 = nn.Linear(u, u) #defining fully connected with input 3 and output 3
        self.fc3 = nn.Linear(u, u)
        self.fc4 = nn.Linear(u, 1) #defining fully connected with input 3 and output 1
        self.ReLu = nn.ReLU() #defining Rectified Linear Unit as activation function
        self.Sigmoid = nn.Softsign() #defining Rectified Linear Unit as activation function
        self.Tanhshrink = nn.Tanh()
        self.Softplus = nn.ELU()
    def forward(self, x): #feed forward
        layer1 = x.view(-1, 1) #make it flat in one dimension from 0 - 784
        # print(layer1)
        layer2 = self.ReLu(self.fc1(layer1)) #layer2 = layer1 -> fc1 -> relu
        layer3 = self.Sigmoid(self.fc2(layer2)) #layer3 = layer2 -> fc2 -> Sigmoid
        layer4 = self.Tanhshrink(self.fc3(layer3)) #layer3 = layer2 -> fc2 -> Sigmoid
        layer5 = self.Tanhshrink(self.fc4(layer4)) #layer3 = layer2 -> fc2 -> Sigmoid
        return layer5 #softmax activation to layer4
    # F.log_softmax(layer4)
    def __repr__(self):
        return json.dumps(self.__dict__)

# create your optimizer


dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 100000, 100, 100, 10

# Create random Tensors to hold inputs and outputs
def f(x):
    """Nonlinear function to be learnt with a multi-layer Perceptron."""
    return np.sin(5*x)/(5*x)
# amount of test/validation data
x_train = 5*np.random.random_sample((N,1))-5
print(x_train)
# training inputs/features
x_test  = 2*np.random.random_sample((N,1))-1
# testing inputs/features
v_train = 0.01*np.random.randn(N ,1)
# noise on training set
v_test  = 0*np.random.randn(N ,1)
# no noise on testing set
y_train = f(x_train) + v_train
# training outputs
y_test  = f(x_test)  + v_test
# testing outputs



x = torch.tensor(x_train, dtype=torch.float, requires_grad=False, device= device)
y = torch.tensor(y_train, dtype=torch.float, requires_grad=False, device= device)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
model = DeepNeuralNetwork(100).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1)
# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
# loss_fn = torch.nn.KLDivLoss(size_average=False)

criterion = nn.MSELoss()
n = 1000
time1 = time.time()
for t in range(n):
    optimizer.zero_grad()   # zero the gradient buffers
    output = model(x)
    loss = criterion(output, y)
    #print(model.fc1.bias.grad)
    if (t%100==0):
        print(t/n)
    loss.backward()
    #print(f"after:{model.fc1.bias.grad}")
    optimizer.step() 
    
    
time1 = time.time()-time1   
print(time1)
counter = 0
for param in model.parameters():
    #print(counter)
    counter += 1
    # param -= learning_rate * param.grad
    #print(param)
x = torch.tensor(x_test, dtype=torch.float, device= device)
y_pred = model(x)

x_np = x.cpu().detach().numpy()
y_pred = y_pred.cpu().detach().numpy()

#print(y_pred.T)
#print(x_test.T,y_test.T)

n=len(x_test.T[0])

mw.resize(800,800)
view = pg.GraphicsLayoutWidget()  ## GraphicsView with GraphicsLayout inserted by default
mw.setCentralWidget(view)
mw.show()
mw.setWindowTitle('pyqtgraph example: ScatterPlot')

w1 = view.addPlot()
s1 = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
pos = []
for k in range(0,n):
    pos.append(np.array([x_test.T[0][k],y_test.T[0][k]]))
pos1 = np.array(pos)
pos2 = []
for k in range(0,n):
    pos2.append(np.array([x_np.T[0][k],y_pred.T[0][k]]))
pos2 = np.array(pos2)
spots2 = [{'pos': pos2.T[:,i], 'data': 2} for i in range(n)]
spots = [{'pos': pos1.T[:,i], 'data': 1} for i in range(n)]

s1.addPoints(spots)
w1.addItem(s1)

s2 = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 120))
s2.addPoints(spots2)
w1.addItem(s2)

mw.show()
## Start Qt event loop unless running in interactive mode.
"""
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
"""
