​
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tps

# compute cross point of two line, given two points on each line
def solveCrossPoint(x0,y0,x1,y1,x2,y2,x3,y3):
if y0 != y1 and x0 != x1 and y2 != y3 and x2 != x3:
a = y1-y0
b = x1*y0-x0*y1
c = x1-x0
d = y3-y2
e = x3*y2-x2*y3
f = x3-x2
y = float(a*e-b*d)/(a*f-c*d)
x = float(y*c-b)/a
return x,y
if (y0 == y1 and y2 == y3) or (x0 == x1 and x2 == x3):
return -999, -999
if y0 == y1 and x2 == x3:
return x2, y0
if y0 == y1 and x2 != x3:
return (y0-y2)*(x3-x2)/(y3-y2) + x2, y0
if y0 != y1 and x2 == x3:
return x2, (y1-y0)*(x2-x0)/(x1-x0) + y0
if x0 == x1 and y2 == y3:
return x0, y2
if x0 == x1 and y2 != y3:
return x0, (y3-y2)*(x0-x2)/(x3-x2) + y2
if x0 != x1 and y2 == y3:
return (y2-y0)*(x1-x0)/(y1-y0) + y0, y2

# resize the coordinates of the points on the second line
# pts2: points to be resized
# pts1: baseline points
# x0_1, y0_1: the circle center of the baseline points pts1
def resizeLine(pts2, pts1, x0_1, y0_1):
x1_1, y1_1 = pts1[0]
x2_1, y2_1 = pts1[-1]
x1_2, y1_2 = pts2[0]
x2_2, y2_2 = pts2[-1]
cx1, cy1 = solveCrossPoint(x0_1,y0_1,x1_1,y1_1,x1_2,y1_2,x2_2,y2_2)
cx2, cy2 = solveCrossPoint(x0_1,y0_1,x2_1,y2_1,x1_2,y1_2,x2_2,y2_2)
l_old = np.sqrt((x1_2 - x2_2)**2 + (y1_2 - y2_2)**2)
l_new = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
xc = (x1_2 + x2_2)/2
yc = (y1_2 + y2_2)/2
pts2_new = []
for pt in pts2:
x_ = (pt[0]-xc)*l_new/l_old + xc
y_ = (pt[1]-yc)*l_new/l_old + yc
pts2_new.append([x_,y_])
return np.array(pts2_new)

# compute corresponding circle center and radius
# side: 0 -- ^, 1 -- v
def solveCircle(x1, y1, x2, y2, theta, side=0):
xc = (x1 + x2)/2
yc = (y1 + y2)/2
r = np.sqrt((x1-x2)**2 + (y1-y2)**2 )/(2*np.sin(theta/2))
rtheta = r * np.cos(theta/2)
if side == 0:
y0 = yc - np.sqrt(rtheta**2 / (((y2-y1)/(x2-x1))**2 + 1))
else:
y0 = yc + np.sqrt(rtheta**2 / (((y2-y1)/(x2-x1))**2 + 1))
x0 = xc - ((y2-y1)*(y0-yc)/(x2-x1))
return x0, y0, r

# calculate corresponding point on the curve
def getNewPoint(r, x0, y0, x1, y1):
if (y1 - y0)>0:
y1_ = y0 + np.sqrt(r**2 / (((x0-x1)/(y0-y1))**2 + 1))
else:
y1_ = y0 - np.sqrt(r**2 / (((x0-x1)/(y0-y1))**2 + 1))
x1_ = x0 - ((x0-x1)*(y0-y1_)/(y0-y1))
return x1_, y1_

def getNewPoint2(r, x0, y0, xs, ys, theta_diff, side=0):
theta_s = np.arcsin((ys-y0)/r)
if side==0:
theta_s = np.pi - theta_s
dy = r * np.sin(theta_s - theta_diff)
dx = r * np.cos(theta_s - theta_diff)
else:
theta_s = np.pi - theta_s
dy = r * np.sin(theta_s + theta_diff)
dx = r * np.cos(theta_s + theta_diff)
return x0+dx, y0+dy


# transform a line of baseline points to corresponding curved points
def transform(pts, theta, side=0):
#theta = np.pi/2
x0, y0, r = solveCircle(pts[0][0], pts[0][1], pts[-1][0], pts[-1][1],
theta, side)

pts_ = []
xs, ys = pts[0][0], pts[0][1]
xe, ye = pts[-1][0], pts[-1][1]
l = np.sqrt((xe-xs)**2 + (ye-ys)**2)
for i in range(pts.shape[0]):
# x_, y_ = getNewPoint(r, x0, y0, pts[i][0], pts[i][1])
x, y = pts[i][0], pts[i][1]
theta_diff = theta * np.sqrt((x-xs)**2 + (y-ys)**2) / l
x_, y_= getNewPoint2(r, x0, y0, xs, ys, theta_diff, side)
pts_.append([x_,y_])
pts_ = np.array(pts_)

# plt.grid()
# plt.scatter([p[0] for p in pts], [p[1] for p in pts], marker='+')
# plt.scatter([p[0] for p in pts_], [p[1] for p in pts_], marker='o')
# plt.scatter([x0], [y0], marker='x')
# plt.show()

return pts_

# given a set of baseline points, return the warpping correspondence of image
# pts: baseline points
# theta: curve radian
# img_w, img_h: image size
# side: 0 -- bend down '^' ; 1 -- bend up 'v'
def solveCoorCorrespondence(pts, theta, img_w, img_h, bbox, side=0, isFan=True):
if side==0:
ptsL, ptsS = np.split(pts, 2)
else:
ptsS, ptsL = np.split(pts, 2)
x0L,y0L,rL = solveCircle(ptsL[0][0], ptsL[0][1], ptsL[-1][0], ptsL[-1][1],
theta, side)
if isFan:
ptsS_new = resizeLine(ptsS, ptsL, x0L, y0L)
else:
ptsS_new = ptsS

ptsL_ = transform(ptsL, theta, side)
ptsS_ = transform(ptsS_new, theta, side)

if side==0:
pts_ = np.vstack((ptsL_,ptsS_))
else:
pts_ = np.vstack((ptsS_,ptsL_))

# plt.grid()
# plt.scatter([p[0] for p in pts], [p[1] for p in pts], marker='+')
# plt.scatter([p[0] for p in pts_], [p[1] for p in pts_], marker='o')
# plt.show()
# generate all the pixel coordiantes in the image
xgs = np.arange(img_w, dtype=float)
ygs = np.arange(img_h, dtype=float)
xgs, ygs = np.meshgrid(xgs, ygs)
pgs = np.vstack((xgs.flatten(),ygs.flatten())).T

# warp coodiantes
pgt = tps.warpCoordinates(pts, pts_, pgs)

# # get transformed bounding box
# xtp = np.linspace(bbox[0][0],bbox[1][0], 10)
# ytp = np.linspace(bbox[0][1],bbox[1][1], 10)
# ptp = np.vstack((xtp,ytp)).T
# if isFan:
# ptp = resizeLine(ptp, ptsL, x0L, y0L)
#
# xbt = np.linspace(bbox[2][0],bbox[3][0], 10)
# ybt = np.linspace(bbox[2][1],bbox[3][1], 10)
# pbt = np.vstack((xbt, ybt)).T
# if isFan:
# pbt = resizeLine(pbt, ptsL, x0L, y0L)
#
# pbbox = np.vstack((ptp,pbt))
#
# pbbox_dst = tps.warpCoordinates(pts, pts_, pbbox)

# re-warp from dst image to src image
pgt_round = pgt.round()
u_min = pgt_round.T[1].min()
u_max = pgt_round.T[1].max()
v_min = pgt_round.T[0].min()
v_max = pgt_round.T[0].max()
xgt = np.arange(v_min, v_max+1, dtype=float)
ygt = np.arange(u_min, u_max+1, dtype=float)
xgt,ygt = np.meshgrid(xgt,ygt)
pgt_new = np.vstack((xgt.flatten(),ygt.flatten())).T

# dx = ((v_max - v_min) - img_w)
# dy = ((u_max - u_min) - img_h)
# pts_[:,0] = pts_[:,0] + dx
# pts_[:,1] = pts_[:,1] + dy
return pgt_new, tps.warpCoordinates(pts_, pts, pgt_new), pts_


#theta = np.pi/3
#pts = np.linspace(30, 100, num=10)
#pts = np.vstack((pts,np.array([1]*10))).T
#
#ptsx, ptsy = np.linspace(30, 100, 10), np.linspace(10, 40, 2)
#ptsx, ptsy = np.meshgrid(ptsx, ptsy)
#pts = np.vstack([ptsx.flatten(), ptsy.flatten()]).T
#
#pts1, pts2 = np.split(pts, 2)
#
#pgt_round, pgs_new = solveCoorCorrespondence(pts, theta, 128, 64, 0)
