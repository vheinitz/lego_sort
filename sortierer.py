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


Exposure=-8
cap = cv2.VideoCapture(1)   #Kamera einschalten
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
time.sleep(1)

pos=0
MotorOn=False
DetectorTH=75
ImageDetected=False
StepperPos=0
Class=0
ClassToStepperPos = {}

fe = featex.FeatEx()
fe.append(featex.Pixels())

try:
    cl.learn(pc1, fe)
except:
    pass

#Hier beginnt die Erkennung
while(True):
    ret, Kamerbild = cap.read() #Bild von der Kamera lesen
    cv2.imshow("VideoIn", Kamerbild) #Bild anzeigen lassen


    Ausgabebild = copy.copy(Kamerbild)  # Kamerabild in Ausgabebild kopieren
    detector = Kamerbild[DetectorRoiY1: DetectorRoiY2, DetectorRoiX1: DetectorRoiX2]
    # cv2.imshow("Detector", detector)
    DetektorMittelwert = cv2.mean(detector)[0]
    print "DetektorMittelwert: %d" % DetektorMittelwert

    cv2.rectangle(Ausgabebild, (RoiX1, RoiY1), (RoiX2, RoiY2), (0, 255, 0), 1, 1)
    cv2.rectangle(Ausgabebild, (DetectorRoiX1, DetectorRoiY1), (DetectorRoiX2, DetectorRoiY2), (255, 0, 0), 2)
    cv2.putText(Ausgabebild, '%d' % pos, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("VideoIn", Ausgabebild)
    cv2.waitKey(10)

    if DetektorMittelwert<DetectorTH :
        try:
            roi = pc.process(Kamerbild)  #ROI ausschneiden
            Abteilungswinkel = cl.test(roi)
            print Abteilungswinkel
            if not Abteilungswinkel == '0':
                while True:
                    kb.stepper(1, 20)
                    kb.poll(t=0.6)
                    s = kb.getDI(8)
                    print s
                    if s == 0:
                        break
                kb.stepper(1, int(Abteilungswinkel) )

        except Exception, ex:
            print str(ex)
            pass

        k = cv2.waitKey(100)& 0xFF
        if k == ord('e'):
            break
        elif k == ord('b'):
            bg = Kamerbild.astype(np.float)

kb.reset(LEDS)