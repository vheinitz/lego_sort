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
import Einstellungen            #Gemeinsame Datei Einstellungen importieren


Klassifikator = classifier.Classifier()

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
        Klassifikator.resultmap[n] = v
except:
    pass

LEDS=6   #Arduino DO6
MOTOR=7  #Arduino DO7

Exposure=-8
cap = cv2.VideoCapture(Einstellungen.CameraID)   #Kamera einschalten
time.sleep(2)
cap.set(cv2.CAP_PROP_EXPOSURE, Exposure)

Klassifikator.setBaseDir(Einstellungen.BaseDir)


ProzessKetteROI = procchain.ProcChain("ROI")
ProzessKetteROI.append(procchain.ImgProcToGray())
ProzessKetteROI.append(procchain.ImgProcRoi(RoiX1, RoiX2, RoiY1, RoiY2))
ProzessKetteROI.enableDebug(True)

ProzessKetteSVMData = procchain.ProcChain("SVM-Data")
ProzessKetteSVMData.append(procchain.ImgProcToGray())
ProzessKetteSVMData.append(procchain.ImgProcStore("ROI"))
ProzessKetteSVMData.append(procchain.ImgProcObjRoi("ObjRoi"))
ProzessKetteSVMData.append(procchain.ImgProcUse("ROI"))
ProzessKetteSVMData.append(procchain.ImgProcRoiByName("ObjRoi"))
ProzessKetteSVMData.append(procchain.ImgProcResize(32, 32))
ProzessKetteSVMData.append(procchain.ImgProcNorm())
ProzessKetteSVMData.append(procchain.ImgProcStore("result"))
ProzessKetteSVMData.enableDebug(True)


KiKuBoardInstanz = kikuboard.KiKuBoard()                   #KiKu (Kinderkurs) Platine Objekt zum Steuern vom Band und LEDs
KiKuBoardInstanz.connect()                                 #Mit Arduino verbinden
print KiKuBoardInstanz.version()                           #Version ausgeben
KiKuBoardInstanz.set(LEDS)                                 #LEDs einschalten

Fachwinkel=0                                 #Variable fuer den Fachwinkel
MotorOn=False                                #Bandmotor an oder aus. Beim Start aus
DetectorTH=5                                 #Detektor-Schranke (experimentell ermittelt)
DifferenzZwischenNBild=4                     #Objekt feststellen indem die Mittelwertdifferenzen zwischen den N. Bild verglichen werden

FeatureExtractionInstanz = featex.FeatEx()        # FE Instanz erzeugen
FeatureExtractionInstanz.append(featex.Pixels())  # Einzelne Pixel als Merkmale nutzen

try:
    Klassifikator.learn(ProzessKetteSVMData, FeatureExtractionInstanz)
except:
    pass


#Hier beginnt die Erkennung
while(True):
    ret, Kamerbild = cap.read() #Bild von der Kamera lesen
    cv2.imshow("VideoIn", Kamerbild) #Bild anzeigen lassen


    Ausgabebild = copy.copy(Kamerbild)  # Kamerabild in Ausgabebild kopieren
    DetectorROI = Kamerbild[DetectorRoiY1: DetectorRoiY2, DetectorRoiX1: DetectorRoiX2]
    ObjektImDetektor = tools.movement_detected(DetectorROI, DetectorTH, DifferenzZwischenNBild)



    if MotorOn and ObjektImDetektor:  # Objekt festgestellt -> Bild fuer das eingestelle Fachwinkel hinzufuegen
        roi = ProzessKetteROI.process(Kamerbild)  # ROI ausschneiden
        Fachwinkel = Klassifikator.test(roi)
        tools.movement_detector_reset()  # Verhindern, dass ein Objekt mehrmals erkannt wird, Objekt aus dem Feld rausfahren lassen
        KiKuBoardInstanz.reset(MOTOR)   #Band Anhalten
        kb.stepper(1, int(Fachwinkel))
        KiKuBoardInstanz.set(MOTOR)     #Band Starten
        time.sleep(3)                   #Warten, bis objekt vom Band weg ist
        KiKuBoardInstanz.stepper(1, -int(Fachwinkel)) #Fach auf 0-Position zurueck drehen

    cv2.rectangle(Ausgabebild, (RoiX1, RoiY1), (RoiX2, RoiY2), (0, 255, 0), 1, 1)
    cv2.rectangle(Ausgabebild, (DetectorRoiX1, DetectorRoiY1), (DetectorRoiX2, DetectorRoiY2), (255, 0, 0), 2)
    cv2.putText(Ausgabebild, 'Drehen zum Fachwinkel: %d, Objekt: %s' %
                (Fachwinkel, str(ObjektImDetektor)),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(Ausgabebild, 'Beenden: q, Winkel: +/-, Band: Leertaste, Fachwinkel 0-Pos: 0',
                (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_4)

    cv2.imshow("VideoIn", Ausgabebild)

    Taste = cv2.waitKey(10) & 0xFF

    if Taste == ord('q'):  # Anlernen beenden
        break

    elif Taste == ord('+'):  # Fachwinkel vergroessern, Fach drehen
        KiKuBoardInstanz.stepper(1, 20)

    elif Taste == ord('-'):  # Fachwinkel verkleinern, Fach drehen
        KiKuBoardInstanz.stepper(1, -20)

    elif Taste == ord('0'):  # Fachwinkel auf 0 setzen
        Fachwinkel = 0

    elif Taste == ord(' '):  # Band an-/aus- schalten
        if MotorOn:
            KiKuBoardInstanz.reset(LEDS)
            KiKuBoardInstanz.reset(MOTOR)
            MotorOn = False

        else:
            KiKuBoardInstanz.set(LEDS)
            KiKuBoardInstanz.set(MOTOR)
            MotorOn = True

KiKuBoardInstanz.reset(LEDS)
KiKuBoardInstanz.reset(MOTOR)