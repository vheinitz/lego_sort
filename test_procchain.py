# -*- coding: utf-8 -*-
import os, sys              # Dateien, Verzeichnisse
import cv2                  # OpenCV - Computer Vision Bibliothek
import numpy as np          # Notwendig fue OpenCV/Python
from ml2 import procchain   # Process-Chain Kette Der Funktionen fuer die Bildmanipulation


cap = cv2.VideoCapture(0)   #Bilder werden von der Kamera eingelesen

pc = procchain.ProcChain()  #Eine Instanz von Process-chain erzeugen

######## Alle moeglichen Operationen #############
pc.append(procchain.ImgProcToGray())
#pc.append(procchain.ImgProcBlur())
#pc.append(procchain.ImgProcSplit('r', 'g', 'b'))
#pc.append(procchain.ImgProcUse('r'))
#pc.append(procchain.ImgProcChRed())
#pc.append(procchain.ImgProcChGreen())
#pc.append(procchain.ImgProcChBlue())
#pc.append(procchain.ImgProcStore())
#pc.append(procchain.ImgProcTh(120))
#pc.append(procchain.ImgProcThInv(120))
#pc.append(procchain.ImgProcThOtsu())
#pc.append(procchain.ImgProcThAdpt())
#pc.append(procchain.ImgProcResize(64,64))
#pc.append(procchain.ImgProcNorm())
#pc.append(procchain.ImgProcBin())
pc.append(procchain.ImgProcPyrDn())
pc.append(procchain.ImgProcPyrDn())
pc.append(procchain.ImgProcPyrDn())
pc.append(procchain.ImgProcBilFilter())
#pc.append(procchain.ImgProcRoi(100, 300, 100, 300))

pc.enableDebug(True)

while(True):
    ret, frame = cap.read()           #Bild von Kamera einnlesen

    out = pc.process(frame)           #Prozess-Kette auf das Bild anwenden

    cv2.imshow("Kamera", frame)
    cv2.imshow("Ausgabe", out)
    k = cv2.waitKey(10)

    if k != 255:
        break

