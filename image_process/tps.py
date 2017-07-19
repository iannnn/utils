# -*- coding: utf-8 -*-
#import ipdb
import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist, squareform

global T

def makeT(cp):
# cp: [K x 2] control points
# T: [(K+3) x (K+3)]
K = cp.shape[0]
T = np.zeros((K+3, K+3))
T[:K, 0] = 1
T[:K, 1:3] = cp
T[K, 3:] = 1
T[K+1:, 3:] = cp.T
'''
[[ 1. -1. -1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
[ 1. 0. -1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
[ 1. 1. -1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
[ 1. -1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
[ 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
[ 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
[ 1. -1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
[ 1. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
[ 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
[ 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
[ 0. 0. 0. -1. 0. 1. -1. 0. 1. -1. 0. 1.]
[ 0. 0. 0. -1. -1. -1. 0. 0. 0. 1. 1. 1.]]
'''
#print pdist(cp, metric='euclidean')
R = squareform(pdist(cp, metric='euclidean'))
R = R * R
R[R == 0] = 1 # a trick to make R ln(R) 0
R = R * np.log(R)
np.fill_diagonal(R, 0)
T[:K, 3:] = R
return T

def liftPts(p, cp):
# p: [N x 2], input points
# cp: [K x 2], control points
# pLift: [N x (3+K)], lifted input points
N, K = p.shape[0], cp.shape[0]
pLift = np.zeros((N, K+3))
pLift[:,0] = 1
pLift[:,1:3] = p
R = cdist(p, cp, 'euclidean')
R = R * R
R[R == 0] = 1
R = R * np.log(R)
pLift[:,3:] = R
return pLift

def warpCoordinates(cps, cpt, gps):
# construct T
global T
T = makeT(cps)

# solve cx, cy (coefficients for x and y)
xt, yt = np.split(cpt.T, 2)
xtAug = np.concatenate([xt.flatten(), np.zeros(3)])
ytAug = np.concatenate([yt.flatten(), np.zeros(3)])
cx = nl.solve(T, xtAug) # [K+3] #solve inverse matrix
cy = nl.solve(T, ytAug) # T * cy = yrAug

# transform
pgLift = liftPts(gps, cps) # [N x (K+3)]
xgt = np.dot(pgLift, cx.T)
ygt = np.dot(pgLift, cy.T)

# # display
# xs, ys = np.split(cps.T, 2)
# xgs, ygs = np.split(gps.T, 2)
# plt.xlim(-1, 12)
# plt.ylim(-2, 4)
# plt.subplot(2, 1, 1)
# plt.title('Input Image')
# #plt.xlim(-1, 12)
# #plt.ylim(-2, 4)
# plt.grid()
# plt.scatter(xs, ys, marker='+', c='g', s=40)
# plt.scatter(xgs, ygs, marker='.', c='r', s=5)
# plt.subplot(2, 1, 2)
# plt.title('Warped Image')
# #plt.xlim(-1, 12)
# #plt.ylim(-2, 4)
# plt.grid()
# plt.scatter(xt, yt, marker='+', c='y', s=40)
# plt.scatter(xgt, ygt, marker='.', c='b', s=5)
# plt.show()

return np.vstack([xgt,ygt]).T


def test():
# source control points
x, y = np.linspace(-4, 5, 10), np.linspace(0, 1, 2)
x, y = np.meshgrid(x, y)
xs = x.flatten()
ys = y.flatten()
cps = np.vstack([xs, ys]).T

# target control points
xt = xs + np.random.uniform(-0.3, 0.3, size=xs.size)
yt = ys + np.random.uniform(-0.3, 0.3, size=ys.size)
cpt = np.vstack([xt, yt]).T

# dense grid
N = 100
M = 32
x = np.linspace(-6, 7, N)
y = np.linspace(-3, 4, M)
x, y = np.meshgrid(x, y)
xgs, ygs = x.flatten(), y.flatten()
gps = np.vstack([xgs, ygs]).T

gpt = warpImage(cps, cpt, gps)
