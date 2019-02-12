import os, sys
import cv2
import numpy as np
from ml2 import featex
from ml2 import procchain
from ml2 import classifier



cap = cv2.VideoCapture(0)
pc = procchain.ProcChain()
pc.append(procchain.ImgProcToGray())
pc.append(procchain.ImgProcRoi(0, 200, 0, 200))

cl = classifier.Classifier()

ret = cl.setBaseDir("c:/tmp/hand");


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imshow("VideoIn", frame)
    simg = pc.process(frame)
    cv2.imshow("VideoStore", simg)
    k = cv2.waitKey(100)& 0xFF
    if k  == ord('q'):
        break
    elif k  == ord('o'):
        t.addItem(simg,'open_hand')
    elif k  == ord('c'):
        t.addItem(simg,'close_hand')
    elif k  == ord('a'):
        t.addItem(simg,'any')

pc1 = procchain.ProcChain()
fe = featex.FeatEx()

#fe.append(rafael.Moments())
#fe.append(rafael.HuMoments())
#fe.append(rafael.Corners())
fe.append(featex.Pixels())

pc1.append(procchain.ImgProcToGray())
pc1.append(procchain.ImgProcResize(32, 32))
pc1.append(procchain.ImgProcStore("result"))

cl.learn(pc1,fe)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imshow("VideoIn", frame)
    roi = pc.process(frame)
    cv2.imshow("VideoStore", roi)

    r = cl.test(roi)
    print r
    k = cv2.waitKey(100)& 0xFF
    if k  == ord('e'):
        break
