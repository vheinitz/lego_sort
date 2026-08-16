# lego_sort — sort Lego bricks by machine learning (Jugend forscht) 🧱

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3-blue.svg)](#)

A **Jugend forscht** project by the author's students: **automatically sorting
Lego bricks** using a small machine-learning / feature-extraction pipeline and
an Arduino-controlled sorter.

## Architecture

| Component | Path | Purpose |
|-----------|------|---------|
| ML framework | `ml2/` | lightweight feature-extraction + classifier toolkit (`featex`, `classifier`, `procchain`, `datagen`, `dataio`, `tools`) |
| Feature tests | `test_featex.py`, `test_procchain.py` | unit tests for the pipeline |
| Sorter | `sortierer.py`, `Sorierer_lernen.py` | the brick sorter and its learning mode |
| Hardware bridge | `kikuboard/kikuboard.py` | Arduino control (the KiKu board) |
| Config | `Einstellungen.py`, `Variablen.py` | settings and shared variables |

## The idea

A camera identifies each brick's shape/colour (features → classifier), and the
sorter routes it into the correct bin — a self-contained machine-learning
experiment using only a small custom `ml2` toolkit rather than a heavyweight
framework, ideal for teaching the concepts.

## Requirements

- Python 3, OpenCV, NumPy, an Arduino board

## License

[MIT](LICENSE) © Valentin Heinitz

*Jugend forscht school project by the author's students.*
