# -*- coding: utf-8 -*-
"""
    Data input/output
    -----------------

    TODO: describe

    :copyright: (c) 2018 by Aleksej Kusnezov
    :license: BSD, see LICENSE for more details.
"""

import numpy as np
import cv2


def getDataFromTailImg(  fn, TailW, TailH ,Nr, Nc, procchain=None, xoffset=0, yoffset=0, xborder=0, yborder=0, vdbg=False ):
    data = []
    img = cv2.imread(fn)
    skipvdbg = False
    for r in range(0, Nr):
        for c in range(0, Nc):
            rect = img[
                   (yoffset + yborder * r) + (r * TailH):(yoffset + yborder * r) + (r * TailH + TailH),
                   (xoffset + xborder * c) + (c * TailW):(xoffset + xborder * c) + (c * TailW + TailW)
                   ]
            if vdbg and not skipvdbg:
                cv2.imshow("getDataFromTailImg", rect)
                k=cv2.waitKey(100)
                if k>0 and chr(k) == 'q':
                    skipvdbg = True

            if procchain == None:
                data.append(rect.reshape(-1))
            else:
                tmpdat = rect
                for p in procchain:
                     tmpdat = p(tmpdat)
                data.append(tmpdat.reshape(-1))
    return data

def getDataFromImgDir( dn, procchain=None ):
    data = []
    files = os.listdir(dn)
    for f in files:
        rect = cv2.imread(os.path.join(dn, f))
        if procchain == None:
            data.append(rect.reshape(-1))
        else:
            tmpdat = rect
            for p in procchain:
                 tmpdat = p(tmpdat)
            data.append(tmpdat.reshape(-1))

    return data