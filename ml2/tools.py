# -*- coding: utf-8 -*-
"""
    Tools and utilities
    -----------------

    TODO: describe

    :copyright: (c) 2018 by Aleksej Kusnezov
    :license: BSD, see LICENSE for more details.
"""
import os, sys
import cv2
import numpy as np
import copy
import math

RoiX1=0
RoiY1=0
RoiX2=0
RoiY2=0
RoiPnt=0

wname = "Select Roi (Exit with q)"
movement_detected_old_avg=-1
movement_detected_nth=0

def movement_detected( img, th=5, nth=2 ):
    global movement_detected_old_avg
    global movement_detected_nth
    movement_detected_nth += 1
    if movement_detected_nth % nth == 0:
        return False
    blr = cv2.blur(img, (5, 5))
    res = False
    av = cv2.mean(blr)[0]
    if movement_detected_old_avg == -1:
        res = False
    elif movement_detected_old_avg - av > th:
        res = True
    movement_detected_old_avg = av
    return res




def select_roi(img, col):

    cv2.namedWindow(wname)
    frame = copy.copy(img)
    def mouse_event(event,x,y,flags,param):
        global RoiX1, RoiY1
        global RoiX2, RoiY2
        global RoiPnt
        global frame

        if event == cv2.EVENT_LBUTTONUP:
            if RoiPnt == 0:
                RoiX1, RoiY1 = x, y
                RoiPnt=1
            elif RoiPnt == 1:
                RoiX2, RoiY2 = x, y
                RoiPnt = 0

        elif event == cv2.EVENT_MOUSEMOVE:
            if RoiPnt == 1:
                RoiX2, RoiY2 = x, y

    cv2.setMouseCallback(wname, mouse_event)

    while True:
        frame = copy.copy(img)
        cv2.rectangle(frame, (RoiX1, RoiY1), (RoiX2, RoiY2), col, 1, 1)
        cv2.imshow(wname, frame)

        k = cv2.waitKey(100) & 0xFF
        if k == ord('q'):
            break

    cv2.destroyWindow(wname)
    return RoiX1, RoiY1, RoiX2, RoiY2

if __name__ == '__main__':
   pass