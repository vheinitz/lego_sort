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



BaseDir = "c:/tmp/muenzen_1"
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

LEDS=6
MOTOR=7

frame = None

rect_pts = [(0,0),(0,0)]

def SelectRoi(event, x, y, flags, param):
    global RoiX1, RoiY1, RoiX2, RoiY2
    if event == cv2.EVENT_LBUTTONDOWN:
        (RoiX1,RoiY1) = (x, y)
        (RoiX2,RoiY2) = (-1,-1)

    if event == cv2.EVENT_LBUTTONUP:
        (RoiX2, RoiY2) = (x, y)

    # draw a rectangle around the region of interest
    if RoiX2 == -1:
        cv2.rectangle(frame, (RoiX1,RoiY1), (x, y), (0, 255, 0), 2)
    else:
        cv2.rectangle(frame, (RoiX1, RoiY1), (RoiX2,RoiY2), (0, 255, 0), 2)
    cv2.imshow("VideoIn", frame)


def SelectRoiDetector(event, x, y, flags, param):
    global DetectorRoiX1, DetectorRoiY1, DetectorRoiX2, DetectorRoiY2
    if event == cv2.EVENT_LBUTTONDOWN:
        (DetectorRoiX1, DetectorRoiY1) = (x, y)
        (DetectorRoiX2, DetectorRoiY2) = (-1,-1)

    if event == cv2.EVENT_LBUTTONUP:
        (DetectorRoiX2, DetectorRoiY2) = (x, y)

    # draw a rectangle around the region of interest
    if DetectorRoiX2 == -1:
        cv2.rectangle(frame, (DetectorRoiX1, DetectorRoiY1), (x, y), (255, 0, 0), 2)
    else:
        cv2.rectangle(frame, (DetectorRoiX1, DetectorRoiY1), (DetectorRoiX2, DetectorRoiY2), (255, 0, 0), 2)
    cv2.imshow("VideoIn", frame)

cv2.namedWindow("VideoIn")


Exposure=-8
cap = cv2.VideoCapture(1)
time.sleep(2)
cap.set(cv2.CAP_PROP_EXPOSURE, Exposure)


cl.setBaseDir(BaseDir)
#RoiX1, RoiY1, RoiX2, RoiY2 = cl.roi()

#if RoiX1 is None:
#    RoiX1, RoiY1, RoiX2, RoiY2 = tools.select_roi(cap)
#    cl.set_roi(RoiX1, RoiY1, RoiX2, RoiY2)


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
        cv2.setMouseCallback("VideoIn", SelectRoi)

    elif k == ord('d'):             # Detector Roi setzen
        cv2.setMouseCallback("VideoIn", SelectRoiDetector)

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

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imshow("VideoIn", frame)
    roi = pc.process(frame)
    #cv2.imshow("VideoStore", roi)

    try:
        r = cl.test(roi)
        print r
        if not r == '0':
            while True:
                kb.stepper(1, 20)
                kb.poll(t=0.6)
                s = kb.getDI(7)
                print s
                if s == 0:
                    break
            kb.stepper(1, int(r) )

    except Exception, ex:
        print str(ex)
        pass

    k = cv2.waitKey(100)& 0xFF
    if k == ord('e'):
        break
    elif k == ord('b'):
        bg = frame.astype(np.float)

kb.reset(LEDS)