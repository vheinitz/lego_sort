import cv2
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

Klassifikator.setBaseDir(Einstellungen.BaseDir)

(RoiX1,RoiY1,RoiX2,RoiY2) = Klassifikator.roi()
(DetectorRoiX1,DetectorRoiY1,DetectorRoiX2,DetectorRoiY2) = Klassifikator.roi("det_roi.txt")

LEDS=6   #Arduino DO6
MOTOR=7  #Arduino DO7

Exposure=-8
cap = cv2.VideoCapture(Einstellungen.CameraID)   #Kamera einschalten
time.sleep(2)
cap.set(cv2.CAP_PROP_EXPOSURE, Exposure)

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

KiKuBoardInstanz = kikuboard.KiKuBoard()                      # KiKu (Kinderkurs) Platine Objekt zum Steuern vom Band und LEDs
KiKuBoardInstanz.connect()                                    # Mit Arduino verbinden
print KiKuBoardInstanz.version()                              # Version ausgeben
KiKuBoardInstanz.set(LEDS)                                    # LEDs einschalten

Fachwinkel=0                                                # Variable fuer den Fachwinkel
MotorOn=False                                               # Bandmotor an oder aus. Beim Start aus
DetectorTH=5                                                # Detektor-Schranke (experimentell ermittelt)
DifferenzZwischenNBild=4                                    # Objekt feststellen indem die Mittelwertdifferenzen zwischen den N. Bild verglichen werden

FeatureExtractionInstanz = featex.FeatEx()                  # FE Instanz erzeugen
FeatureExtractionInstanz.append(featex.Pixels())            # Einzelne Pixel als Merkmale nutzen

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
    cv2.rectangle(Ausgabebild, (RoiX1, RoiY1), (RoiX2, RoiY2), (0, 255, 0), 1, 1)
    cv2.rectangle(Ausgabebild, (DetectorRoiX1, DetectorRoiY1), (DetectorRoiX2, DetectorRoiY2), (255, 0, 0), 2)
    cv2.putText(Ausgabebild, 'Beenden: q, Winkel: +/-, Band: Leertaste, Fachwinkel 0-Pos: 0',
                (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_4)

    cv2.imshow("VideoIn", Ausgabebild)

    if MotorOn and ObjektImDetektor:  # Objekt festgestellt -> Bild fuer das eingestelle Fachwinkel hinzufuegen
        ROIBild = Kamerbild[RoiY1: RoiY2, RoiX1: RoiX2]  # ROI ausschneiden
        Fachwinkel = Klassifikator.test(ROIBild)
        cv2.putText(Ausgabebild, 'Drehen zum Fachwinkel: %s, Objekt: %s' %
                    (Fachwinkel, str(ObjektImDetektor)),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("VideoIn", Ausgabebild)
        cv2.waitKey(10)

        tools.movement_detector_reset()  # Verhindern, dass ein Objekt mehrmals erkannt wird, Objekt aus dem Feld rausfahren lassen
        KiKuBoardInstanz.reset(MOTOR)   #Band Anhalten
        KiKuBoardInstanz.stepper(1, int(Fachwinkel))
        KiKuBoardInstanz.set(MOTOR)     #Band Starten
        time.sleep(3)                   #Warten, bis objekt vom Band weg ist
        KiKuBoardInstanz.stepper(1, -int(Fachwinkel)) #Fach auf 0-Position zurueck drehen
        time.sleep(3)  # Warten, bis objekt vom Band weg ist

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