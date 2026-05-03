# Studying the Behavior of a Fixed Bed Catalytic Reactor with PINNs

> **Research and Innovation Project (PIR) — 4th Year Engineering**
> Institut National des Sciences Appliquées de Toulouse (INSA Toulouse)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![DeepXDE](https://img.shields.io/badge/DeepXDE-Latest-green)](https://deepxde.readthedocs.io/)

---

## Overview

This project applies **Physics-Informed Neural Networks (PINNs)** to simulate coupled fluid flow and heat transfer in a simplified 2D fixed bed catalytic reactor. Results are benchmarked against CFD reference simulations for velocity $(u, v)$, pressure $p$, and temperature $T$.

---

## Poster

![Research Poster](Poster-PINNs.png)

---

## Results

| | CFD Reference | PINNs Prediction |
|---|:---:|:---:|
| **Velocity $u$** | ![CFD u](Img/CFD/u0.png) | ![PINNs u](Img/PINNs/u2.png) |
| **Velocity $v$** | ![CFD v](Img/CFD/v0.png) | ![PINNs v](Img/PINNs/v2.png) |
| **Pressure $p$** | ![CFD p](Img/CFD/p0.png) | ![PINNs p](Img/PINNs/p2.png) |
| **Temperature $T$** | ![CFD T](Img/CFD/t0.png) | ![PINNs T](Img/PINNs/t2.png) |

---

## Implementation

Two notebooks are provided:

| Notebook | Framework |
|----------|-----------|
| [`2D_deepxde.ipynb`](2D_deepxde.ipynb) | DeepXDE |
| [`2D_tensorflow.ipynb`](2D_tensorflow.ipynb) | TensorFlow 2.x |

```bash
pip install tensorflow deepxde numpy matplotlib
```

---

## Project Report

**[Full Report (PDF)](Thesis_PINNs_in_Catalytic_Reactor.pdf)**

---

## Team

| Name | Role |
|------|------|
| Mai Dinh Nam | Researcher |
| Nguyen Phuc Luan | Researcher |
| Thai Doan Kien | Researcher |
| Mrs. Marie | Academic Advisor |
| Ms. Liantsoa | Academic Advisor |
