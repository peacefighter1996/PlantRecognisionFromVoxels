# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 11:43:17 2018

@author: Bram / Ian Arbouw
"""
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
import matplotlib

matplotlib.lines.Line2D
try:
    w.close()
    w=gl.GLViewWidget()
except NameError:
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
size = []


def clear_data(coordinates, plotsize, voxelsize, minimum_ammount=1):
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
        :param voxelsize: size of the voxel in mm
        :param minimum_ammount=1: minimum amount of points to consider a voxel
            point.
    Output:
        list of the voxel locations
    .. note::

        this product is still in development
    """
    areashape = (int((plotsize[1] - plotsize[0]) / voxelsize)+1,
                 int((plotsize[3] - plotsize[2]) / voxelsize)+1,
                 int((plotsize[5] - plotsize[4]) / voxelsize)+1)
    mode = 0

    if (minimum_ammount == 1):
        voxels = np.full(areashape,
                         False,
                         dtype=bool)
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
        x = int((coor[0] - plotsize[0]) / voxelsize)
        y = int((coor[1] - plotsize[2]) / voxelsize)
        z = int((coor[2] - plotsize[4]) / voxelsize)
        
        if(mode == 0):
            if (voxels[x][y][z]==False):
                voxels[x][y][z] = True
                newcoordinates.append([x, y, z])

        if(mode == 1):
            if (voxels[x][y][z] >= 0):
                
                voxels[x][y][z] += 1
                if (voxels[x][y][z] >= minimum_ammount):
                    newcoordinates.append([x, y, z])
                    voxels[x][y][z] = -1
    gridsize = 13
    delta = int((gridsize+1)/2)
    areas = []
    for x in range(delta, areashape[0] - delta):
        for y in range(delta, areashape[1] - delta):
            for z in range(delta, areashape[2] - delta):
                if (voxels[x][y][z] == 1 or voxels[x][y][z] == -1):
                    # print("yeah")
                    areas.append([voxels[x-delta:x+delta-1,
                                         y-delta:y+delta-1,
                                         z-delta:z+delta-1],
                                  [[x, y, z]]
                                  ]
                                 )
                    # print(area)
    x = 0
    """
    while x < len(areas):
        y = len(areas)-1
        while y > x:
            if np.array_equal(areas[x][0], areas[y][0]):
                for z in range(0, len(areas[y][1])):
                    areas[x][1].append(areas[y][1][z])
                del areas[y]↓
            y -= 1
        x += 1
        print(x/len(areas))
    """
    return newcoordinates, areas


def angle_to_coordinates(theta_distance_line, x_size,
                         x_min=0.0, x_max=0.0,
                         x_min_check=True, x_max_check=True,
                         y_min=0.0, y_max=0.0,
                         y_min_check=True, y_max_check=True,
                         z_min=0.0, z_max=0.0,
                         z_min_check=True, z_max_check=True):
    coordinates = []
    """
    zmax = 5.0
    zmin = -100.0
    ymax = -25.0
    ymin = -175.0
    xmax = 0.0
    xmin = 0.0
    """
    zmax = 5.0
    zmin = -100.0
    ymax = -90.0
    ymin = -140.0
    xmax = 0.0
    xmin = -30.0

    for i in range(0, theta_distance_line.shape[0]):
        z = (theta_distance_line[i][1] *
             np.cos(np.radians(theta_distance_line[i][0])) / 10)
        y = (theta_distance_line[i][1] *
             np.sin(np.radians(theta_distance_line[i][0])) / 10)
        x = theta_distance_line[i][2] * x_size
        
        mode = 2
        if mode == 1:
            if (y < ymax and z < zmax and y > ymin and z > zmin):
                if (x > xmax):
                    xmax = x
                if (x < xmin):
                    xmin = x
    
                coordinates.append([x, y, z])
        else:
            if (y < ymax and z < zmax and y > ymin and z > zmin and x < xmax and x > xmin):    
                coordinates.append([x, y, z])

    Size = [xmin, xmax,
            ymin, ymax,
            zmin, zmax]
    return coordinates, Size


filename = ["scandata201810291313"
            ]
for i in range(0, len(filename)):
    fileload = str(filename[i] + ".txt")
    filedistancesave = str(filename[i] + "_distance.csv")
    filesave = str(filename[i] + ".csv")
    filesavecleaned = str(filename[i] + "_cleansmall.csv")
    lidardata = open(fileload, "r").readlines()

    s = (len(lidardata), 3)
    theta_distance_line = np.zeros(s)
    regel = 0.0
    rotationcount = 1;
    memrotation = 0.0;
    for i in range(0, len(lidardata)):
        words = lidardata[i].split()
        # print (words)
        if (len(words) != 0):
            try:
                if (words[0] == 'z:'):
                    regel = float(words[1])
                else:
                    newline = (float(words[1]), float(words[3]), -regel)
                    theta_distance_line[i] = newline
                    if (float(words[1])<memrotation):
                        rotationcount+=1
                    memrotation = float(words[1])
                        
            except IndexError:
                print("index error")
        # print("{:.4f}%".format(100*(float(i)/float(len(lidardata)))/3))
    print(rotationcount)
    # print(theta_distance_line)
    X_size = 48.0/float(regel)
    lidarsavecoor = open(filedistancesave, "w")
    for coor in theta_distance_line:
        lidarsavecoor.write("{},{},{}\n".format(
                coor[0],
                coor[1],
                coor[2]
                ))

    lidarsavecoor.close()
    coordinates, size = angle_to_coordinates(theta_distance_line, X_size)
    lidarsavecoor = open(filesave, "w")
    for coor in coordinates:
        lidarsavecoor.write("{},{},{}\n".format(
                coor[0],
                coor[1],
                coor[2]
                ))

    lidarsavecoor.close()
    voxel_size = X_size * 8
    print (voxel_size)
    point_ammound_per_voxel = 1
    cleanedupcoordinates,voxelgrid = clear_data(coordinates,
                                      size, 
                                      voxel_size,
                                      point_ammound_per_voxel)

    lidarsavecoor = open(filesavecleaned, "w")

    for coor in cleanedupcoordinates:
        lidarsavecoor.write("{},{},{}\n".format(
                coor[0],
                coor[1],
                coor[2]
                ))
    lidarsavecoor.close()
