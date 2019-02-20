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
import Einstellungen




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
    cfg = open( os.path.join(Einstellungen.BaseDir,"roi.txt"), 'r' ).readlines()
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

    cfg = open(os.path.join(Einstellungen.BaseDir, "map.txt"), 'r').readlines()
    for l in cfg:
        n = int(l.split(' ')[0])
        v = int(l.split(' ')[1])
        cl.resultmap[n] = v
except:
    pass

LEDS=6   #Arduino DO6
MOTOR=7  #Arduino DO7
REED=8   #Arduino DI8

Kamerbild = None

Exposure=-8
cap = cv2.VideoCapture(Einstellungen.CameraID)   #Kamera einschalten
time.sleep(2)
cap.set(cv2.CAP_PROP_EXPOSURE, Exposure)


cl.setBaseDir(Einstellungen.BaseDir)



pc = procchain.ProcChain("ROI")      #Neue Prozesskette mit dem Namen ROI fuer die Grauumwandlung und ROI herausschneiden
pc.append(procchain.ImgProcToGray()) #Neue Grauumwandlung
pc.append(procchain.ImgProcRoi( RoiX1, RoiX2, RoiY1, RoiY2)) #ROI herausschneiden
pc.enableDebug(True)                                         #Ausgabe der Schritte aktivieren zum Testen

pc1 = procchain.ProcChain("SVM-Data")       #Neue Prozesskette zum Umwandeln des Bildes in SVM-Taugliche Daten
pc1.append(procchain.ImgProcToGray())       #Graustufen
pc1.append(procchain.ImgProcStore( "ROI"))  #Speichere akt. Bild in der Kette unter dem Namen ROI
pc1.append(procchain.ImgProcObjRoi("ObjRoi")) #ROI/Rahmen vom Objekt finden
pc1.append(procchain.ImgProcUse( "ROI"))    #Gespeicheres Bild mit dem Namen ROI als akt. benutzen
pc1.append(procchain.ImgProcRoiByName( "ObjRoi")) #Aus dem Bild den Ausschnitt gefunden in "ObjRoi" herausschneiden
pc1.append(procchain.ImgProcResize(32, 32))  #Ausschnitt auf 32x32 Pixel verkleinern
pc1.append(procchain.ImgProcNorm())          #Bild normalisieren, alle Werte zwischen 0 und 1
pc1.append(procchain.ImgProcStore("result")) #Bild als "result" im Kontextobjekt speichern
pc1.enableDebug(True)


kb = kikuboard.KiKuBoard()                   #KiKu (Kinderkurs) Platine Objekt zum Steuern vom Band und LEDs
kb.connect()                                 #Mit Arduino verbinden
print kb.version()                           #Version ausgeben
kb.set(LEDS)                                 #LEDs einschalten

Fachwinkel=0                                 #Variable fuer den Fachwinkel
MotorOn=False                                #Bandmotor an oder aus. Beim Start aus
DetectorTH=75
ObjektImDetektor=False
StepperPos=0
Class=0
ClassToStepperPos = {}


while(True):

    ret, Kamerbild = cap.read()   #Das Bild von der Kamera in der Variable Kamerbild speichern

    Ausgabebild = copy.copy(Kamerbild) #Kamerabild in Ausgabebild kopieren
    detector = Kamerbild[DetectorRoiY1: DetectorRoiY2, DetectorRoiX1: DetectorRoiX2]
    #cv2.imshow("Detector", detector)

    ObjektImDetektor = tools.movement_detected(detector, 3, 4)

    cv2.rectangle(Ausgabebild, (RoiX1, RoiY1), (RoiX2, RoiY2), (0, 255, 0), 1, 1)
    cv2.rectangle(Ausgabebild, (DetectorRoiX1, DetectorRoiY1), (DetectorRoiX2, DetectorRoiY2), (255, 0, 0), 2)
    cv2.putText(Ausgabebild, 'Fachwinkel: %d, Objekt: %s' %
                (Fachwinkel, str(ObjektImDetektor)),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(Ausgabebild, 'Beenden: q, ROI: r, Detektor: d, Bild hinzuf.: a, Winkel: +/-, Band: Leertaste',
                (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_4)
    cv2.putText(Ausgabebild,
                'Fachwinkel 0-Pos: 0',
                (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_4)

    cv2.imshow("VideoIn", Ausgabebild)   #Bild mit ROI, Detektor, Info und Hilfe zeigen



    simg = pc.process(Kamerbild)
    cv2.imshow("VideoStore", simg)

    pc1.process(simg)
    ctx = pc1.context()
    objshape = ctx["ObjRoi"]
    AreaObject = objshape[2] * objshape[3]
    Areaimg = (simg.shape[0] * simg.shape[1])

    if AreaObject < (Areaimg *0.9):
        print objshape

    k = cv2.waitKey(100) & 0xFF

    if MotorOn and not ObjektImDetektor:   # Append Image
        cl.addItem(simg, "%d" % Fachwinkel)
        ObjektImDetektor = True

    if ObjektImDetektor and DetektorMittelwert > DetectorTH:   # Append Image
        ObjektImDetektor = False

    if k == ord('q'):               # Anlernen beenden
        break

    elif k == ord('r'):             # Roi setzen
        (RoiX1, RoiY1, RoiX2, RoiY2)=tools.select_roi(Kamerbild, (0, 255, 0))

    elif k == ord('d'):             # Detector Roi setzen
        (DetectorRoiX1, DetectorRoiY1, DetectorRoiX2, DetectorRoiY2) = tools.select_roi(Kamerbild, (0, 0, 255))

    elif k == ord('a'):             # Append Image
        cl.addItem(simg,"%d" % Fachwinkel)

    elif k == ord('+'):
        Fachwinkel += 20
        kb.stepper(1, 20)

    elif k == ord('-'):
        Fachwinkel -= 20
        kb.stepper(1, -20)

    elif k == ord('0'):
        Fachwinkel = 0

    elif k == ord(' '):
        if MotorOn:
            kb.reset(LEDS)
            kb.reset(MOTOR)
            MotorOn = False

        else:
            kb.set(LEDS)
            kb.set(MOTOR)
            MotorOn = True

fe = featex.FeatEx()
fe.append(featex.Pixels())

try:
    cl.learn(pc1, fe)
except:
    pass

try:
    cfg = open( os.path.join(Einstellungen.BaseDir,"roi.txt"), 'w' )
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
    cfg = open(os.path.join(Einstellungen.BaseDir, "map.txt"), 'w')
    for k in cl.resultmap:

        cfg.write("%d %d\n" % (k,cl.resultmap[k])
              )
    cfg.close()
except:
    pass

cv2.destroyAllWindows()

