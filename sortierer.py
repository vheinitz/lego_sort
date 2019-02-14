import os, sys
import cv2
import numpy as np
from ml2 import featex
from ml2 import procchain
from ml2 import classifier
from ml2 import tools
from kikuboard import kikuboard
import time
import copy



BaseDir = "Testmodell1"
cl = classifier.Classifier()

RoiX1=0
RoiY1=0
RoiX2=640
RoiY2=480

DetectorRoiX1=0
DetectorRoiY1=0
DetectorRoiX2=640
DetectorRoiY2=480

try:
    cfg = open( os.path.join(BaseDir,"roi.txt"), 'r' ).readlines()
    n = cfg[0].split(' ')
    RoiX1 = int(n[0])
    RoiY1 = int(n[1])
    RoiX2 = int(n[2])
    RoiY2 = int(n[3])

    n = cfg[1].split(' ')
    DetectorRoiX1 = int(n[0])
    DetectorRoiY1 = int(n[1])
    DetectorRoiX2 = int(n[2])
    DetectorRoiY2 = int(n[3])

    cfg = open(os.path.join(BaseDir, "map.txt"), 'r').readlines()
    for l in cfg:
        n = int(l.split(' ')[0])
        v = int(l.split(' ')[1])
        cl.resultmap[n] = v
except:
    pass

LEDS=6   #Arduino DO6
MOTOR=7  #Arduino DO7
REED=8   #Arduino DI8

frame = None

#rect_pts = [(0,0),(0,0)]

Exposure=-8
cap = cv2.VideoCapture(1)
time.sleep(2)
cap.set(cv2.CAP_PROP_EXPOSURE, Exposure)

cl.setBaseDir(BaseDir)


pc = procchain.ProcChain("ROI")
pc.append(procchain.ImgProcToGray())
pc.append(procchain.ImgProcRoi( RoiX1, RoiX2, RoiY1, RoiY2))
pc.enableDebug(True)

pc1 = procchain.ProcChain("SVM-Data")
pc1.append(procchain.ImgProcToGray())
pc1.append(procchain.ImgProcStore( "ROI"))
pc1.append(procchain.ImgProcObjRoi("ObjRoi"))
pc1.append(procchain.ImgProcUse( "ROI"))
pc1.append(procchain.ImgProcRoiByName( "ObjRoi"))
pc1.append(procchain.ImgProcResize(32, 32))
pc1.append(procchain.ImgProcNorm())
pc1.append(procchain.ImgProcStore("result"))
pc1.enableDebug(True)


kb = kikuboard.KiKuBoard()
kb.connect()
print kb.version()
kb.set(LEDS)

pos=0
MotorOn=False
DetectorTH=80
ImageDetected=False
StepperPos=0
Class=0
ClassToStepperPos = {}


while(True):

    # Capture frame-by-frame
    ret, frame = cap.read()

    roimark = copy.copy(frame)
    detector = frame[DetectorRoiY1: DetectorRoiY2, DetectorRoiX1: DetectorRoiX2]
    #cv2.imshow("Detector", detector)
    DetectorMean = cv2.mean(detector)[0]
    print "Mean %d" % DetectorMean


    cv2.rectangle(roimark, (RoiX1, RoiY1), (RoiX2, RoiY2), (0, 255, 0), 1, 1)
    cv2.rectangle(roimark, (DetectorRoiX1, DetectorRoiY1), (DetectorRoiX2, DetectorRoiY2), (255, 0, 0), 2)
    cv2.putText(roimark, '%d' % pos, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("VideoIn", roimark)

    simg = pc.process(frame)
    #cv2.imshow("VideoStore", simg)

    pc1.process(simg)
    ctx = pc1.context()
    objshape = ctx["ObjRoi"]
    AreaObject = objshape[2] * objshape[3]
    Areaimg = (simg.shape[0] * simg.shape[1])

    if AreaObject < (Areaimg *0.9):
        print objshape

    k = cv2.waitKey(100) & 0xFF

    if MotorOn and not ImageDetected and DetectorMean < DetectorTH:   # Append Image
        cl.addItem(simg, "%d" % pos)
        ImageDetected = True

    if ImageDetected and DetectorMean > DetectorTH:   # Append Image
        ImageDetected = False

    if k == ord('q'):               # Anlernen beenden
        break

    elif k == ord('r'):             # Roi setzen
        (RoiX1, RoiY1, RoiX2, RoiY2)=tools.select_roi(roimark, (0, 255, 0))

    elif k == ord('d'):             # Detector Roi setzen
        (DetectorRoiX1, DetectorRoiY1, DetectorRoiX2, DetectorRoiY2) = tools.select_roi(roimark, (0, 0, 255))

    elif k == ord('a'):             # Append Image
        cl.addItem(simg,"%d" % pos)

    elif k == ord('+'):
        pos += 50
        kb.stepper(1, 50)
        print(pos)

    elif k == ord('-'):
        pos -= 50
        kb.stepper(1, -50)
        print(pos)

    elif k == ord('3'):
        kb.set(LEDS)
        kb.set(MOTOR)
        MotorOn = True

    elif k == ord('4'):
        kb.reset(LEDS)
        kb.reset(MOTOR)
        MotorOn = False

fe = featex.FeatEx()
fe.append(featex.Pixels())

try:
    cl.learn(pc1, fe)
except:
    pass

try:
    cfg = open( os.path.join(BaseDir,"roi.txt"), 'w' )
    cfg.write( "%d %d %d %d\n%d %d %d %d" %
               (RoiX1,
                RoiY1,
                RoiX2,
                RoiY2,
                DetectorRoiX1,
                DetectorRoiY1,
                DetectorRoiX2,
                DetectorRoiY2)
               )
    cfg.close()
    cfg = open(os.path.join(BaseDir, "map.txt"), 'w')
    for k in cl.resultmap:

        cfg.write("%d %d\n" % (k,cl.resultmap[k])
              )
    cfg.close()
except:
    pass

cv2.destroyAllWindows()

ImageDetected=False

r = ''
while(True):
    # Capture frame-by-frame

    ret, frame = cap.read()

    roimark = copy.copy(frame)
    cv2.rectangle(roimark, (RoiX1, RoiY1), (RoiX2, RoiY2), (0, 255, 0), 1, 1)
    cv2.rectangle(roimark, (DetectorRoiX1, DetectorRoiY1), (DetectorRoiX2, DetectorRoiY2), (255, 0, 0), 2)
    cv2.putText(roimark, '%s' % r, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("VideoIn", roimark)
    roi = pc.process(frame)
    detector = frame[DetectorRoiY1: DetectorRoiY2, DetectorRoiX1: DetectorRoiX2]
    #cv2.imshow("VideoStore", roi)

    try:

        DetectorMean = cv2.mean(detector)[0]
        print "Mean %d" % DetectorMean

        if  DetectorMean < DetectorTH:  # Append Image
            ImageDetected = True
            kb.reset(MOTOR)
            r = cl.test(roi)
            if not r == '':
                while True:
                    kb.stepper(1, 10)

                    kb.poll(t=0.2)
                    time.sleep(0.2)
                    s = kb.getDI(REED)
                    if s == 0:
                        print "0-Stelle gefunden"
                        break
                kb.stepper(1, int(r) )
                time.sleep(3)
                kb.set(MOTOR)
                kb.set(LED)
                r=''

    except Exception, ex:
        print str(ex)
        pass

    k = cv2.waitKey(100)& 0xFF
    if k == ord('e'):
        break


kb.reset(LEDS)