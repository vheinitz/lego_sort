# -*- coding: utf-8 -*-
"""
    Data generator
    -----------------

    TODO: describe

    :copyright: (c) 2018 by Aleksej Kusnezov
    :license: BSD, see LICENSE for more details.
"""
import numpy as np
import cv2
import os
from random import randrange


def createRectImg(  TailW=32, TailH=32,Nr=1, Nc=1, Rs=3, Rp=3, Rc=50 ):
    img  = np.zeros((TailW*Nc, TailH*Nr, 3), np.uint8)
    for r in range(0, Nr):
        for c in range(0, Nc):
            d1 = randrange(-Rp, Rp)
            d2 = randrange(-Rs, Rs)
            C = randrange(-Rc, Rc)
            rect = img[c * TailW:c * TailW + TailW, r * TailH:r * TailH + TailH]
            cv2.rectangle(rect, (10 + d1, 10 + d1), (20 + d2, 20 + d2), (100+C, 100+C, 100+C), -1)
    return img


def createRectImgFile( fn, TailW=32, TailH=32,Nr=1, Nc=1, Rs=3, Rp=3, Rc=50 ):
    img = createRectImg( TailW, TailH, Nr, Nc, Rs, Rp, Rc )
    cv2.imwrite(fn, img)

def createEllipseImg(  TailW=32, TailH=32,Nr=1, Nc=1, Rs=3, Rp=3, Rc=50 ):
    img  = np.zeros((TailW*Nc, TailH*Nr, 3), np.uint8)
    for r in range(0, Nr):
        for c in range(0, Nc):
            d1 = randrange(-Rp, Rp)
            d2 = randrange(-Rs, Rs)
            C = randrange(-Rc, Rc)
            rect = img[c * TailW:c * TailW + TailW, r * TailH:r * TailH + TailH]
            cv2.ellipse(rect, (TailW / 2 + d1, TailH / 2 + d1), (5 + d1, 5 + d2), 0, 0, 360,
                        (100 + C, 100 + C, 100 + C), -1)
    return img


def createEllipseImgFile( fn, TailW=32, TailH=32,Nr=1, Nc=1, Rs=3, Rp=3, Rc=50 ):
    img = createRectImg( TailW, TailH, Nr, Nc, Rs, Rp, Rc )
    cv2.imwrite(fn, img)

"""
def addSaltGray(image,n, min=0, max=255): #add salt-&-pepper noise in grayscale image

    k=0
    salt=True
    ih=image.shape[0]
    iw=image.shape[1]
    noisypixels=(ih*iw*n)/100

    for i in range(ih*iw):
        if k<noisypixels:  #keep track of noise level
                if salt==True:
                        image[r.randrange(0,ih)][r.randrange(0,iw)]=max
                        salt=False
                else:
                        image[r.randrange(0,ih)][r.randrange(0,iw)]=min
                        salt=True
                k+=1
        else:
            break
    return image

def addRndGray(image,n, min=0, max=255):

    k=0
    salt=True
    ih=image.shape[0]
    iw=image.shape[1]
    noisypixels=(ih*iw*n)/100

    for i in range(ih*iw):
        if k<noisypixels:  #keep track of noise level
            image[r.randrange(0,ih)][r.randrange(0,iw)]=r.randrange(min,max)
        else:
            break
    return image

def createEllipseImg( fn, TailW, TailH,Nr, Nc, Rs=3, Rp=3, Rc=50, saveInDir=None ):
    img = np.zeros((TailW*Nc, TailH*Nr, 3), np.uint8)
    for r in range(0, Nr):
        for c in range(0, Nc):
            rect = img[c * TailW:c * TailW + TailW, r * TailH:r * TailH + TailH]
            d1 = randrange(-Rp, Rp)
            d2 = randrange(-Rs, Rs)
            C = randrange(-Rc, Rc)
            cv2.ellipse(rect, (TailW / 2 + d1, TailH / 2 + d1), (5 + d1, 5 + d2), 0, 0, 360, (100+C, 100+C, 100+C), -1)
            if saveInDir and os.path.isdir(saveInDir):
                cv2.imwrite('%s/ellipse_%d.png' % (saveInDir, c+r*Nc),rect)
        cv2.imwrite(fn, img)

def createTextImg( fn, txt, TailW, TailH,Nr, Nc, Rs=3, Rp=3, Rc=50, saveInDir=None ):
    img  = np.zeros((TailW*Nc, TailH*Nr, 3), np.uint8)
    for r in range(0, Nr):
        for c in range(0, Nc):
            d1 = randrange(-Rp, Rp)
            d2 = randrange(0, Rs)
            C = randrange(-Rc, Rc)
            rect = img[c * TailW:c * TailW + TailW, r * TailH:r * TailH + TailH]
            font = cv2.FONT_ITALIC
            cv2.putText(rect, txt, (10 + d1, TailH-10 + d1/2), font, 0.75, (100+C, 100+C, 100+C), 1+d2/2, cv2.LINE_AA)
            if saveInDir and os.path.isdir(saveInDir):
                cv2.imwrite('%s/text_%d.png' % (saveInDir, c+r*Nc),rect)
    cv2.imwrite(fn, img)
"""