# -*- coding: utf-8 -*-
"""
    Processing chain
    -----------------

    TODO: describe

    :copyright: (c) 2018 by Aleksej Kusnezov
    :license: BSD, see LICENSE for more details.
"""
import numpy as np
import cv2
import os
import copy
import math
from matplotlib import pyplot as plt

class ProcChain:

    def __init__(self, name=''):
        self.chain = []
        self.debug = False
        self.debugData = []
        self.ctx = {}
        self.name = name
        self.debugView=None

    def context(self):
        return self.ctx

    def append(self, step):
        self.chain.append(step)

    def enableDebug(self, en = True):
        self.debug = True

    def process(self, img):
        step = 1
        self.debugData = []
        tmp = copy.deepcopy(img)
        if self.debug:
            if len(self.debugData) == 0:
                self.debugData.append(("START", tmp))
            else:
                self.debugData[0][1] = tmp




        for s in self.chain:
            tmp = s.process(tmp, self.ctx )
            if self.debug:
                try:
                    stepDbg = s.debugData(step) #
                    if stepDbg:
                        for e in stepDbg:
                            if len(self.debugData) <= step:
                                self.debugData.append((e[0], self.ctx[e[1]]))
                            else:
                                self.debugData[0][1] = self.ctx[e[1]]

                    else:
                        caption = "%s:%d" %(s.sname(),step)
                        self.debugData.append( (caption,tmp) )

                    n = int( len(self.debugData) )
                    self.debugView = np.zeros(( 100, n *100, 3), np.uint8)
                    for i in xrange(len(self.debugData)):
                        dtmp = self.debugData[i][1]
                        origshape = dtmp.shape
                        w = 100 if dtmp.shape[0] > 100 else dtmp.shape[0]
                        h = 100 if dtmp.shape[1] > 100 else dtmp.shape[0]
                        dtmp = cv2.resize( dtmp,( w,h))
                        if len(dtmp.shape) < 3:
                            dtmp = cv2.normalize(dtmp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                            dtmp = cv2.cvtColor(dtmp, cv2.COLOR_GRAY2RGB)

                        self.debugView[0:h,(i*100):(i*100+w)] =  dtmp #
                        frame = self.debugView[0:100,(i*100):(i*100+100)]
                        cv2.rectangle(frame, (0, 0), (100, 100), (0, 255, 0), 1)
                        cv2.putText(frame, self.debugData[i][0] , (0,100-20 ),0, 0.4, (255,0,0))
                        cv2.putText(frame, "%d:%d" % (origshape[0], origshape[1]), (0, 100 - 10), 0, 0.4, ( 255, 0, 0))
                except Exception, ex:
                    pass
            step += 1

        if self.debug and plt:
            cv2.imshow("Process chain[%s]" % self.name, self.debugView)
            #cv2.waitKey(10)
            self.debugData = []

        return tmp


class ImgProcStep:
    pass

    def process(self, img, ctx):
        pass

    def sname(self):
        return ''

    def debugData(self, num):
        return None


class ImgProcToGray(ImgProcStep):

    def __init__(self):
        pass

    def process(self, img, ctx):
        M = img
        if len(img.shape)>=3 and img.shape[2] >=3:
            M = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return M

    def sname(self):
        return 'ToGray'


class ImgProcBlur(ImgProcStep):

    def __init__(self, t=0, **kwargs):
        self.t=t
        self.s = 5
        if kwargs:
            self.s = kwargs['size']

    def process(self, img, ctx):
        M=None
        if self.t == 0:
            M = cv2.blur(img, (self.s, self.s))
        elif self.t == 1:
            M = cv2.bilateralFilter(img, 9, 75, 75)
        elif self.t == 2:
            M = cv2.medianBlur(img, 5)
        elif self.t == 3:
            M = cv2.GaussianBlur(img, (5, 5), 0)

        return M

    def sname(self):
        return 'Blur'


class ImgProcSplit(ImgProcStep):
    def __init__(self, r,g,b):
        self.r = r
        self.g = g
        self.b = b

    def process(self, img, ctx):
        ctx[self.b], ctx[self.g] , ctx[self.r] = cv2.split(img)
        return img

    def sname(self):
        return 'Split'

    def debugData(self, n):
        return [("%s R:%d" % (self.sname(),n),self.r),("%s G:%d" % (self.sname(),n),self.g),("%s B:%d" % (self.sname(),n),self.b)]


class ImgProcRoi(ImgProcStep):
    def __init__(self, x1,x2,y1,y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def process(self, img, ctx):
        return img[self.y1:self.y2,self.x1:self.x2]

    def sname(self):
        return 'ROI'


class ImgProcRoiByName(ImgProcStep):
    def __init__(self, roiname):
        self.roiname = roiname

    def process(self, img, ctx):
        x,y,w,h,= ctx[self.roiname]
        return img[y:y+h, x:x+w]

    def sname(self):
        return 'RoiByName'


class ImgProcChRed(ImgProcStep):
    def __init__(self):
        pass

    def process(self, img, ctx):
        return cv2.split(img)[0]

    def sname(self):
        return 'Red'

class ImgProcCanny(ImgProcStep):
    def __init__(self):
        pass

    def process(self, img, ctx):
        return  cv2.Canny(img,30,250)

    def sname(self):
        return 'Canny'

class ImgProcObjRoi(ImgProcStep):
    def __init__(self, roiname=None):
        self.roiname=roiname
        pass

    def process(self, img, ctx):
        sh = img.shape
        x=0
        y=0
        w=sh[1]
        h=sh[0]  # ???
        img = cv2.blur(img, (5,5))
        img =  cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0), 10)
        im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0), 1)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x),reverse=True)
        try:
            if len( cntsSorted ) >= 2:
                x, y, w, h = cv2.boundingRect(cntsSorted[1])
            else:
                x, y, w, h = cv2.boundingRect(cntsSorted[0])

            if x < 10 or y < 10 or w+x > sh[1] - 10 or h+y > sh[0] - 10:
                x, y, w, h = cv2.boundingRect(cntsSorted[0])

        except:
            pass

        if w < sh[0] / 10 or h < sh[1] / 10:  # Object too small
            x = 0
            y = 0
            w = sh[1]
            h = sh[0]  # ???

        if self.roiname:
            ctx[self.roiname] = (x, y, w, h)

        return img[y:y+h, x:x+w]


    def sname(self):
        return 'Cont'


class ImgProcChGreen(ImgProcStep):
    def __init__(self):
        pass

    def process(self, img, ctx):
        return cv2.split(img)[1]

    def sname(self):
        return 'Green'


class ImgProcChBlue(ImgProcStep):
    def __init__(self):
        pass

    def process(self, img, ctx):
        return cv2.split(img)[2]

    def sname(self):
        return 'Blue'


class ImgProcUse(ImgProcStep):

    def __init__(self, name):
        self.name=name

    def process(self, img, ctx):
        return ctx[self.name]

    def sname(self):
        return 'Use'


class ImgProcStore(ImgProcStep):

    def __init__(self, name):
        self.name = name

    def process(self, img, ctx):
        ctx[self.name] = img
        return img

    def sname(self):
        return 'Store'

class ImgProcSub(ImgProcStep):

    def __init__(self, name):
        self.name = name

    def process(self, img, ctx):
        tmp = ctx[self.name]
        sub = cv2.Sub( img, tmp )
        return sub

    def sname(self):
        return 'Sub'

class ImgProcTh(ImgProcStep):

    def __init__(self, th):
        self.th = th

    def process(self, img, ctx):
        return cv2.threshold(img, self.th,255,cv2.THRESH_BINARY)[1]


class ImgProcThInv(ImgProcStep):

    def __init__(self, th):
        self.th = th

    def process(self, img, ctx):
        return cv2.threshold(img, self.th,255,cv2.THRESH_BINARY_INV)[1]


class ImgProcThOtsu(ImgProcStep):
    def __init__(self):
        pass

    def process(self, img, ctx):
        return cv2.threshold(img, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    def sname(self):
        return 'ThOtsu'


class ImgProcThAdpt(ImgProcStep):
    def __init__(self):
        pass

    def process(self, img, ctx):
        return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    def sname(self):
        return 'ThAdpt'


class ImgProcResize(ImgProcStep):

    def __init__(self,w,h):
        self.w = w
        self.h = h

    def process(self, img, ctx):
        M = cv2.resize(img, (self.w, self.h))
        return M

    def sname(self):
        return 'Resize'


class ImgProcNorm(ImgProcStep):

    def __init__(self):
        pass

    def process(self, img, ctx):
        M = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return M

    def sname(self):
        return 'Norm'


class ImgProcBin(ImgProcStep):

    def __init__(self):
        pass

    def process(self, img, ctx):
        M = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        ret, M = cv2.threshold(M, .5, 1, cv2.THRESH_BINARY)
        return M

    def sname(self):
        return 'Bin'


class ImgProcPyrDn(ImgProcStep):

    def __init__(self):
        pass

    def process(self, img, ctx):
        M = cv2.pyrDown( img )
        return M

    def sname(self):
        return 'PyrDn'


class ImgProcBilFilter(ImgProcStep):
    def __init__(self):
        pass

    def process(self, img, ctx):
        return cv2.bilateralFilter(img, 9, 75, 75)

    def sname(self):
        return 'BilFilter'
