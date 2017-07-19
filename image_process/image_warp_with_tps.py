# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import coorTransform, interpolate
from os.path import join

def getBBoxFromPoints(pbbox):
x_min, y_min = pbbox.min(0)
x_max, y_max = pbbox.max(0)
return np.array([[x_min, y_min],
[x_max, y_min],
[x_max, y_max],
[x_min, y_max]])

def getBox2D(box):
angle = box[2]*np.pi/180 + np.pi/2
b = np.cos(angle)*0.5
a = np.sin(angle)*0.5
pt = np.zeros((4,2))

pt[0][0] = box[0][0] - a*box[1][0] - b*box[1][1];
pt[0][1] = box[0][1] + b*box[1][0] - a*box[1][1];
pt[1][0]= box[0][0] + a*box[1][0] - b*box[1][1];
pt[1][1] = box[0][1] - b*box[1][0] - a*box[1][1];
pt[2][0] = 2*box[0][0] - pt[0][0];
pt[2][1] = 2*box[0][1] - pt[0][1];
pt[3][0] = 2*box[0][0] - pt[1][0];
pt[3][1] = 2*box[0][1] - pt[1][1];

return pt

def imageWarp(img, theta, p_num, pcorners, bbox, side=0, isFan=True):
# xtl, ytl = pcorners[0][0], pcorners[0][1]
# xtr, ytr = pcorners[1][0], pcorners[1][1]
# xbl, ybl = pcorners[2][0], pcorners[2][1]
# xbr, ybr = pcorners[3][0], pcorners[3][1]
# ptsx1, ptsy1 = np.linspace(xtl, xtr, num=p_num), np.linspace(ytl, ytr, num=p_num)
# ptsx2, ptsy2 = np.linspace(xbl, xbr, num=p_num), np.linspace(ybl, ybr, num=p_num)
# pts1 = np.vstack([ptsx1.flatten(), ptsy1.flatten()]).T
# pts2 = np.vstack([ptsx2.flatten(), ptsy2.flatten()]).T
# pts = np.vstack((pts1,pts2))

xtp = np.linspace(bbox[0][0],bbox[1][0], p_num)
ytp = np.linspace(bbox[0][1],bbox[1][1], p_num)
ptp = np.vstack((xtp,ytp)).T

xbt = np.linspace(bbox[3][0],bbox[2][0], p_num)
ybt = np.linspace(bbox[3][1],bbox[2][1], p_num)
pbt = np.vstack((xbt, ybt)).T

pts = np.vstack((ptp,pbt))

pgt_new, pgs_new, pbbox_dst = coorTransform.solveCoorCorrespondence(pts, theta, img.shape[1],
img.shape[0], bbox, side, isFan)
img_w_new, img_h_new = pgt_new.max(0) - pgt_new.min(0)
x_new_min, y_new_min = pgt_new.min(0)
# dx = (img_w_new - img.shape[1])
# dy = (img_h_new - img.shape[0])
# pbbox_dst[:,0] = pbbox_dst[:,0] + dx/2
# pbbox_dst[:,1] = pbbox_dst[:,1] + dy/2
pbbox_dst[:,0] = pbbox_dst[:,0] - x_new_min
pbbox_dst[:,1] = pbbox_dst[:,1] - y_new_min
return interpolate.warpImage(img, pgt_new, pgs_new), \
getBox2D(cv2.minAreaRect(pbbox_dst.astype(np.int32)))


def imageWarpRandom(img, bbox):
theta = np.pi/(np.random.randint(2,5))
# theta = np.pi/2
# print theta
side = np.random.randint(0,2)
isFan = np.random.randint(0,2)
# isFan = 0

w = img.shape[1]
h = img.shape[0]
return imageWarp(img, theta, 6, [[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]], bbox, side, isFan)
# return imageWarp(img, theta, 5, bbox, side, isFan)

def extendBBox(bbox, ratio):
xy_mean = bbox.mean(0)
for i in range(bbox.shape[0]):
bbox[i] = xy_mean + (1+ratio)*(bbox[i] - xy_mean)
return bbox




#
#file_bbox = 'd:/data/coco_100_100_bbox_train_val.txt'
#file_info = 'd:/data/coco_100_100_label_train_val.txt'
#
#f = open(file_bbox)
#fc = f.readlines()
#f.close()
#picBBox = {}
#for line in fc:
# cs = line.split()
# picBBox[cs[0]] = cs[1:]
#
#f = open(file_info)
#fc = f.readlines()
#f.close()
#picInfo = {}
#for line in fc:
# cs = line.split()
# picInfo[cs[0]] = cs[1:]
#
##
#img_name = '1000029_100_100.jpg'
#img = cv2.imread(join('d:/data/test/old_100_100/val', img_name))
#bbox = picBBox[img_name]
#bbox = np.array(np.split(np.array(bbox).astype(float),4))
#bbox = extendBBox(bbox, -0.4)
#img_new, bbox_new = imageWarpRandom(img, bbox)
#
##bbox_new = getBox2D(cv2.minAreaRect(pbbox_dst.astype(np.int32)))
#
#for p in bbox:
# cv2.circle(img, (int(p[0]), int(p[1])), 2, (0,255,0))
#
#thic = 1
#for p in bbox_new:
# cv2.circle(img_new, (int(p[0]), int(p[1])), 2, (0,255,0), thic)
# thic += 3
#cv2.imshow('img',img)
#cv2.imshow('img_new',img_new)
#cv2.waitKey()
#cv2.destroyAllWindows()
