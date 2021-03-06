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
import Einstellungen           #Gemeinsame Datei Einstellungen importieren




Klassifikator = classifier.Classifier()
Klassifikator.setBaseDir(Einstellungen.BaseDir)    #Verzeichnis in dem die Bilder gespeichert und geladen werden.

RoiX1=0
RoiY1=0
RoiX2=640
RoiY2=480

DetectorRoiX1=0
DetectorRoiY1=0
DetectorRoiX2=640
DetectorRoiY2=480


#ROI koordinaten laden
(RoiX1,RoiY1,RoiX2,RoiY2) = Klassifikator.roi()
(DetectorRoiX1,DetectorRoiY1,DetectorRoiX2,DetectorRoiY2) = Klassifikator.roi("det_roi.txt")



LEDS=6   #Arduino DO6
MOTOR=7  #Arduino DO7

Kamerbild = None


cap = cv2.VideoCapture(Einstellungen.CameraID)   #Kamera einschalten
time.sleep(2)
cap.set(cv2.CAP_PROP_EXPOSURE, Einstellungen.Exposure)

ProzessKetteSVMData = procchain.ProcChain("SVM-Data")       #Neue Prozesskette zum Umwandeln des Bildes in SVM-Taugliche Daten
ProzessKetteSVMData.append(procchain.ImgProcToGray())       #Graustufen
ProzessKetteSVMData.append(procchain.ImgProcStore("ROI"))  #Speichere akt. Bild in der Kette unter dem Namen ROI
ProzessKetteSVMData.append(procchain.ImgProcObjRoi("ObjRoi")) #ROI/Rahmen vom Objekt finden
ProzessKetteSVMData.append(procchain.ImgProcUse("ROI"))    #Gespeicheres Bild mit dem Namen ROI als akt. benutzen
ProzessKetteSVMData.append(procchain.ImgProcRoiByName("ObjRoi")) #Aus dem Bild den Ausschnitt gefunden in "ObjRoi" herausschneiden
ProzessKetteSVMData.append(procchain.ImgProcResize(32, 32))  #Ausschnitt auf 32x32 Pixel verkleinern
ProzessKetteSVMData.append(procchain.ImgProcNorm())          #Bild normalisieren, alle Werte zwischen 0 und 1
ProzessKetteSVMData.append(procchain.ImgProcStore("result")) #Bild als "result" im Kontextobjekt speichern
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

while(True):                      #Endlosschleife beginnen

    ret, Kamerbild = cap.read()   #Das Bild von der Kamera in der Variable Kamerbild speichern

    Ausgabebild = copy.copy(Kamerbild) #Kamerabild in Ausgabebild kopieren
    DetectorROI = Kamerbild[DetectorRoiY1: DetectorRoiY2, DetectorRoiX1: DetectorRoiX2]

    ObjektImDetektor = tools.movement_detected(DetectorROI, DetectorTH, DifferenzZwischenNBild)

    cv2.rectangle(Ausgabebild, (RoiX1, RoiY1), (RoiX2, RoiY2), (0, 255, 0), 1, 1)
    cv2.rectangle(Ausgabebild, (DetectorRoiX1, DetectorRoiY1), (DetectorRoiX2, DetectorRoiY2), (255, 0, 0), 2)
    cv2.putText(Ausgabebild, 'Fachwinkel: %d, Objekt: %s' %
                (Fachwinkel, str(ObjektImDetektor)),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(Ausgabebild, 'Beenden: q, ROI: r, Detektor: d, Bild hinzuf.: a, Winkel: +/-, Band: Leertaste',
                (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_4)
    cv2.putText(Ausgabebild,
                'Fachwinkel 0-Pos: 0, Lernen: l',
                (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_4)

    cv2.imshow("VideoIn", Ausgabebild)   #Bild mit ROI, Detektor, Info und Hilfe zeigen

    ROIBild = Kamerbild[RoiY1: RoiY2, RoiX1: RoiX2]

    #ProzessKetteSVMData.process(ROIBild)                      #Evtl. zum Testen SVM-Prozesskette anzeigen


    if MotorOn and ObjektImDetektor:   #Objekt festgestellt -> Bild fuer das eingestelle Fachwinkel hinzufuegen
        Klassifikator.addItem(ROIBild, "%d" % Fachwinkel)
        tools.movement_detector_reset() #Verhindern, dass ein Objekt mehrmals erkannt wird, Objekt aus dem Feld rausfahren lassen
        time.sleep(2)                  #Programm anhalten, so dass Objekt aus dem Detektor ROI rausfaehrt

    Taste = cv2.waitKey(100) & 0xFF

    if Taste == ord('q'):               # Anlernen beenden
        break

    elif Taste == ord('r'):             # Roi setzen
        (RoiX1, RoiY1, RoiX2, RoiY2)=tools.select_roi(Kamerbild, (0, 255, 0))
        Klassifikator.set_roi(RoiX1, RoiY1, RoiX2, RoiY2)

    elif Taste == ord('d'):             # Detector Roi setzen
        (DetectorRoiX1, DetectorRoiY1, DetectorRoiX2, DetectorRoiY2) = tools.select_roi(Kamerbild, (0, 0, 255))
        Klassifikator.set_roi( DetectorRoiX1, DetectorRoiY1, DetectorRoiX2, DetectorRoiY2, "det_roi.txt")

    elif Taste == ord('a'):             # Append Image
        Klassifikator.addItem(ROIBild, "%d" % Fachwinkel)

    elif Taste == ord('+'):             #Fachwinkel vergroessern, Fach drehen
        if MotorOn:                     #Band evtl ausschalten, weil Falscherkennung
            KiKuBoardInstanz.reset(LEDS)
            KiKuBoardInstanz.reset(MOTOR)
            MotorOn = False
        Fachwinkel += 20
        KiKuBoardInstanz.stepper(1, 20)

    elif Taste == ord('-'):             #Fachwinkel verkleinern, Fach drehen
        if MotorOn:                     #Band evtl ausschalten, weil Falscherkennung
            KiKuBoardInstanz.reset(LEDS)
            KiKuBoardInstanz.reset(MOTOR)
            MotorOn = False
        Fachwinkel -= 20
        KiKuBoardInstanz.stepper(1, -20)

    elif Taste == ord('0'):             #Fachwinkel auf 0 setzen
        Fachwinkel = 0

    elif Taste == ord('l'):             #Fachwinkel auf 0 setzen
        try:
            Klassifikator.learn(ProzessKetteSVMData,
                                FeatureExtractionInstanz)  # Klassifikator anlernen mit allen Bildern, die gespeichert wurden                                                     # map.txt schliessen
        except:
            pass


    elif Taste == ord(' '):             #Band an-/aus- schalten
        if MotorOn:
            KiKuBoardInstanz.reset(LEDS)
            KiKuBoardInstanz.reset(MOTOR)
            MotorOn = False

        else:
            KiKuBoardInstanz.set(LEDS)
            KiKuBoardInstanz.set(MOTOR)
            MotorOn = True

#While Schleife zu Ende, weil q-Taste



cv2.destroyAllWindows()                                                 #Alle OpenCV-Fenster schliessen

