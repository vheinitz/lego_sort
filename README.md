# lego_sort

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Camera-based sorter from 2019: a conveyor belt carries objects (Lego bricks or
coins) past a webcam, an SVM classifies each one, and a stepper motor turns the
bin to the learned angle. Hardware is driven through the KiKu board (Arduino).

Status: Python 2 code, last changed in February 2019.

## Files

| Path | Content |
|------|---------|
| `Sorierer_lernen.py` | learning mode: set ROIs, record images per bin angle, train |
| `sortierer.py` | sorting mode: train from the stored images, then classify and sort |
| `Einstellungen.py` | `BaseDir` (model/image directory), `CameraID`, `Exposure` |
| `ml2/` | small helper package: image processing chain (`procchain`), feature extraction (`featex`), SVM wrapper (`classifier`), motion detector and ROI selection (`tools`) |
| `kikuboard/kikuboard.py` | serial driver for the KiKu board (CH340 USB adapter, 38400 baud) |
| `test_featex.py`, `test_procchain.py` | interactive camera experiments, not unit tests |
| `Variablen.py` | older variant of the sorting script; not used by the others and not runnable as is (`pc1` undefined, hard-coded `c:/tmp/muenzen_1`) |

## How it works

- Object detection: mean-value difference in a detector ROI between frames
  (`tools.movement_detected`).
- Processing chain: grayscale, object bounding box, crop, resize to 32×32,
  normalize; the raw pixels are the features.
- Classifier: `sklearn.svm.SVC(gamma=0.001, C=100)`. Class name is the bin
  angle; training images are stored as PNG in `BaseDir/<angle>/`, ROIs in
  `BaseDir/roi.txt` and `det_roi.txt`, the model in `BaseDir/er.model`.
  The model is retrained from the images on every start.
- Board: digital output 6 = LEDs, 7 = belt motor, stepper 1 = bin. The board
  must run the "ASP" firmware (`version()` checks for `ASP Ver`), see the
  `kiku_robotics` repository.

## Requirements

Python 2, OpenCV (`cv2`), NumPy, scikit-learn old enough to provide
`sklearn.externals.joblib` (removed in 0.23), pyserial, matplotlib (imported
by `ml2/procchain.py`), a webcam and the KiKu board.

## Usage

Set `BaseDir` and `CameraID` in `Einstellungen.py`, then:

```bash
python Sorierer_lernen.py   # learn
python sortierer.py         # sort
```

Keys in learning mode: `r` set ROI, `d` set detector ROI, `a` add image,
`+`/`-` turn bin by 20 steps, `0` set current angle as zero, space belt on/off,
`l` train, `q` quit. With the belt running, an image is added automatically
whenever the detector fires. Sorting mode accepts `+`/`-`, `0`, space and `q`.

## License

[MIT](LICENSE) © Valentin Heinitz. The headers in `ml2/` still say "BSD, see
LICENSE".
