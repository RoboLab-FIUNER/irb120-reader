#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 13:16:03 2018

@author: juan
"""

import cv2
import numpy as np
from keras.models import load_model
#import digits_ann as ANN

def imgView(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def inside(r1, r2):
  x1,y1,w1,h1 = r1
  x2,y2,w2,h2 = r2
  if (x1 > x2) and (y1 > y2) and (x1+w1 < x2+w2) and (y1+h1 < y2 + h2):
    return True
  else:
    return False

def wrap_digit(rect):
  x, y, w, h = rect
  padding = 5
  hcenter = int(x + w/2)
  vcenter = int(y + h/2)
  if (h > w):
    w = h
    x = hcenter - int(w/2)
  else:
    h = w
    y = vcenter - int(h/2)
  return (x-padding, y-padding, w+padding, h+padding)

def getPred(model, sample):
#    sample[np.newaxis,:,:,np.newaxis]/255.0
    sample = np.expand_dims(np.expand_dims(sample,axis =0), axis=-1)/255.0
    pred_t = model.predict(sample)
    label = np.argmax(pred_t)
    prob = pred_t[0,label]
    return (label, prob)


def mod(img,alpha,beta):
     img.astype('int64')
     temp = alpha * img
     temp = temp + beta
     temp = np.clip(temp,0,255)
     return temp.astype('uint8')

def mod2(img, phi=1, theta=1, gamma=2, maxIntensity=255):
    return ((maxIntensity/phi)*(img/(maxIntensity/theta))**2).astype('uint8')


model = load_model('./../../models/mnist_cnn.h5')


#ann, test_data = ANN.train(ANN.create_ANN(56), 20000)
font = cv2.FONT_HERSHEY_SIMPLEX

path = "../../data/p_samples/test_2.jpg"
img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
img = cv2.resize(img,(int(img.shape[1]/6),int(img.shape[0]/6)))

bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bw = cv2.GaussianBlur(bw, (7,7), 0)

bw = mod(bw, 2.5, -40)
bw = mod2(bw,gamma=3)
#imgView(bw)
ret, thbw = cv2.threshold(bw, 127, 255, cv2.THRESH_BINARY_INV)
#thbw = cv2.bitwise_not(thbw)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
thbw = cv2.dilate(thbw, kernel, iterations = 2)
thbw = cv2.erode(thbw, kernel2, iterations = 2)
#imgView(thbw)
image, cntrs, hier = cv2.findContours(thbw.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

rectangles = []

for c in cntrs:
  r = x,y,w,h = cv2.boundingRect(c)
  a = cv2.contourArea(c)
  b = (img.shape[0]-3) * (img.shape[1] - 3)

  is_inside = False
  for q in rectangles:
    if inside(r, q):
      is_inside = True
      break
  if not is_inside:
    if not a == b:
      rectangles.append(r)

for r in rectangles:
    x,y,w,h = wrap_digit(r)
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
    roi = thbw[y:y+h, x:x+w]
    try:
        sample = cv2.resize(roi.copy(),(28,28))
        digit_class = getPred(model,sample)
    except:
        continue
    cv2.putText(img, "N: "+repr(digit_class[0]) + "P: "+repr(digit_class[1]) , (x, y-2), font, 0.4, (0, 255, 0))
    print("N: "+repr(digit_class[0]) + "P: "+repr(digit_class[1]))



    cv2.imshow("thbw", thbw)
    cv2.imshow("contours", img)
    cv2.imwrite("sample.jpg", img)
    cv2.waitKey(0)
cv2.destroyAllWindows()