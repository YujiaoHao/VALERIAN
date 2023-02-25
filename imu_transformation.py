# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 14:54:56 2022
generate 8 types of data transformation for IMU data
@author: haoyu
"""
import numpy as np
from scipy.interpolate import CubicSpline      # for warping
from transforms3d.axangles import axangle2mat  # for rotation

# =============================================================================
# 8 tasks: noised, scaled, rotated, negated, horizontally flipped, permuted, time-warped, channel-shuffled
# =============================================================================
np.random.seed(101)

#1. noised (jitter)
sigma = 0.05
def Jitter(X, sigma=0.05):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X+myNoise
def DA_Jitter(X, indices):
    res = []
    m = X.shape[0]
    for i in range(m):
        if i in indices:
            res.append(Jitter(X[i]).reshape(1,200,6))
        else:
            res.append(X[i].reshape(1,200,6))
    return np.concatenate(res)

#2. scaled
sigma = 0.1
def Scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,X.shape[1])) # shape=(1,3)
    myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
    return X*myNoise
def DA_Scaling(X, indices):
    res = []
    m = X.shape[0]
    for i in range(m):
        if i in indices:
            res.append(Scaling(X[i]).reshape(1,200,6))
        else:
            res.append(X[i].reshape(1,200,6))
    return np.concatenate(res)


#3. rotated
def Rotation(X):
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(X , axangle2mat(axis,angle))
def Rotation_(X):
    x1 = Rotation(X[:,:3])
    x2 = Rotation(X[:,3:])
    return np.hstack((x1, x2))
def DA_Rotation(X, indices):
    res = []
    m = X.shape[0]
    for i in range(m):
        if i in indices:
            res.append(Rotation_(X[i]).reshape(1,200,6))
        else:
            res.append(X[i].reshape(1,200,6))
    return np.concatenate(res)

#4. negated
def Negated(X):
    return -1 * X
def DA_Negated(X, indices):
    res = []
    m = X.shape[0]
    for i in range(m):
        if i in indices:
            res.append(Negated(X[i]).reshape(1,200,6))
        else:
            res.append(X[i].reshape(1,200,6))
    return np.concatenate(res)

#5. horizontally flipped
def Horizontal_flip(X):
    return np.flip(X,axis = 0)
def DA_Horizontal_flip(X, indices):
    res = []
    m = X.shape[0]
    for i in range(m):
        if i in indices:
            res.append(Horizontal_flip(X[i]).reshape(1,200,6))
        else:
            res.append(X[i].reshape(1,200,6))
    return np.concatenate(res)

#6. permuted
nPerm = 4
minSegLength = 100
def Permutation(X, nPerm=4, minSegLength=10):
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile == True:
        segs = np.zeros(nPerm+1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0]-minSegLength, nPerm-1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:]-segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii]+1],:]
        X_new[pp:pp+len(x_temp),:] = x_temp
        pp += len(x_temp)
    return(X_new)
def DA_Permutation(X, indices):
    res = []
    m = X.shape[0]
    for i in range(m):
        if i in indices:
            res.append(Permutation(X[i]).reshape(1,200,6))
        else:
            res.append(X[i].reshape(1,200,6))
    return np.concatenate(res)


#7. time-warped
sigma = 0.2
knot = 4
def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:,0], yy[:,0])
    cs_y = CubicSpline(xx[:,1], yy[:,1])
    cs_z = CubicSpline(xx[:,2], yy[:,2])
    return np.array([cs_x(x_range),cs_y(x_range),cs_z(x_range)]).transpose()
def DistortTimesteps(X, sigma=0.2):
    tt = GenerateRandomCurves(X, sigma) # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [(X.shape[0]-1)/tt_cum[-1,0],(X.shape[0]-1)/tt_cum[-1,1],(X.shape[0]-1)/tt_cum[-1,2]]
    tt_cum[:,0] = tt_cum[:,0]*t_scale[0]
    tt_cum[:,1] = tt_cum[:,1]*t_scale[1]
    tt_cum[:,2] = tt_cum[:,2]*t_scale[2]
    return tt_cum
def DA_DistortTimesteps(X):
    x1 = DistortTimesteps(X[:,:3])
    x2 = DistortTimesteps(X[:,3:])
    return np.hstack((x1, x2))
def DA_TimeWarp(X, indices):
    res = []
    m = X.shape[0]
    for i in range(m):
        if i in indices:
            res.append(DA_DistortTimesteps(X[i]).reshape(1,200,6))
        else:
            res.append(X[i].reshape(1,200,6))
    return np.concatenate(res)
    

#8. channel-shuffled
def Channel_shuffle(X):
    np.random.shuffle(X.T)
    return X
def DA_channel_shuffle(X, indices):
    res = []
    m = X.shape[0]
    for i in range(m):
        if i in indices:
            res.append(Channel_shuffle(X[i]).reshape(1,200,6))
        else:
            res.append(X[i].reshape(1,200,6))
    return np.concatenate(res)
