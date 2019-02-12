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

RoiX1=0
RoiY1=0
RoiX2=0
RoiY2=0
RoiPnt=0

def select_roi(video):

    cv2.namedWindow("Select Roi")
    ret, frame = video.read()
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

    cv2.setMouseCallback("Select Roi", mouse_event)

    while True:
        ret, frame = video.read()
        cv2.rectangle(frame, (RoiX1, RoiY1), (RoiX2, RoiY2), (0, 255, 0), 1, 1)
        cv2.imshow("Select Roi", frame)

        k = cv2.waitKey(100) & 0xFF
        if k == ord('q'):
            break

    cv2.destroyWindow("Select Roi")
    return RoiX1, RoiY1, RoiX2, RoiY2

if __name__ == '__main__':
    select_roi( cv2.VideoCapture(1) )