# Studying the Behavior of a Fixed Bed Catalytic Reactor with PINNs

> **Research and Innovation Project (PIR) — 4th Year Engineering**
> Institut National des Sciences Appliquées de Toulouse (INSA Toulouse)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![DeepXDE](https://img.shields.io/badge/DeepXDE-Latest-green)](https://deepxde.readthedocs.io/)

---

## Overview

This project applies **Physics-Informed Neural Networks (PINNs)** to model the steady-state behavior of a simplified 2D fixed bed catalytic reactor — a channel flow between two parallel plates (length 0.5 m, gap 0.1 m). Rather than using a traditional mesh-based CFD solver, we embed the governing dimensionless Navier–Stokes and energy equations directly into the neural network's loss function, enabling the model to learn physically consistent solutions with minimal data.

Results are benchmarked against CFD reference simulations for all four field variables: velocity components $(u, v)$, pressure $p$, and temperature $T$.

---

## Motivation

Catalytic reactors involve tightly coupled fluid flow and heat transfer phenomena. Classical CFD approaches rely on fine spatial meshes and iterative numerical solvers that can be computationally expensive. PINNs offer a mesh-free, data-light alternative by encoding physical laws as soft constraints during neural network training — making them an attractive candidate for fast surrogate modeling in process engineering.

---

## Physical Model

### System Assumptions

- Incompressible, **steady-state** flow (no time dependence)
- No external body forces beyond boundary conditions
- Geometry: 2D rectangular channel — length **0.5 m**, gap **0.1 m**
- Working fluid: air

### Governing Equations (Dimensionless Form)

All variables are normalized to the range $[0, 1]$ to improve numerical stability during optimization.

**Continuity equation:**

$$\frac{\partial u^*}{\partial x^*} + \frac{\partial v^*}{\partial y^*} = 0$$

**Momentum equations (Navier–Stokes):**

$$u^* \frac{\partial u^*}{\partial x^*} + v^* \frac{\partial u^*}{\partial y^*} = -\frac{\partial p^*}{\partial x^*} + \frac{1}{Re}\left(\frac{\partial^2 u^*}{\partial {x^*}^{2}} + \frac{\partial^2 u^*}{\partial {y^*}^{2}}\right)$$

$$u^* \frac{\partial v^*}{\partial x^*} + v^* \frac{\partial v^*}{\partial y^*} = -\frac{\partial p^*}{\partial y^*} + \frac{1}{Re}\left(\frac{\partial^2 v^*}{\partial {x^*}^{2}} + \frac{\partial^2 v^*}{\partial {y^*}^{2}}\right)$$

**Energy equation:**

$$u^* \frac{\partial T^*}{\partial x^*} + v^* \frac{\partial T^*}{\partial y^*} = \frac{1}{Re \cdot Pr}\left(\frac{\partial^2 T^*}{\partial {x^*}^{2}} + \frac{\partial^2 T^*}{\partial {y^*}^{2}}\right)$$

where $Re$ is the Reynolds number and $Pr$ is the Prandtl number.

---

## Neural Network Approach

The network takes spatial coordinates $(x^*, y^*)$ as input and simultaneously outputs $(u^*, v^*, p^*, T^*)$. Automatic differentiation computes the PDE residuals, which are incorporated into the loss:

$$\mathcal{L} = \mathcal{L}_{\text{PDE}} + \mathcal{L}_{\text{BC}}$$

### Training Strategy

A two-phase hybrid optimization was used:

| Phase | Optimizer | Steps | Role |
|-------|-----------|-------|------|
| 1 | **Adam** | 200,000 | Global exploration — rapidly drives loss toward a low region |
| 2 | **L-BFGS** | 5,242 | Local refinement — converges to the minimum at step **205,542** |

This combination proved essential: Adam provides robust initial convergence while L-BFGS achieves the sharp final minimization.

---

## Results

PINNs predictions are compared side-by-side with CFD reference solutions.

### Velocity — $u$ (streamwise)

| CFD Reference | PINNs Prediction |
|:---:|:---:|
| ![CFD u](Img/CFD/u0.png) | ![PINNs u](Img/PINNs/u2.png) |

### Velocity — $v$ (transverse)

| CFD Reference | PINNs Prediction |
|:---:|:---:|
| ![CFD v](Img/CFD/v0.png) | ![PINNs v](Img/PINNs/v2.png) |

### Pressure — $p$

| CFD Reference | PINNs Prediction |
|:---:|:---:|
| ![CFD p](Img/CFD/p0.png) | ![PINNs p](Img/PINNs/p2.png) |

### Temperature — $T$

| CFD Reference | PINNs Prediction |
|:---:|:---:|
| ![CFD T](Img/CFD/t0.png) | ![PINNs T](Img/PINNs/t2.png) |

The trained PINNs model successfully reproduces the overall flow patterns and thermal gradients. Minor discrepancies in velocity and pressure fields are observed but remain within an acceptable range, consistent with known optimization limitations of PINNs on coupled multi-physics problems.

---

## Poster

![Research Poster](Poster-PINNs.png)

---

## Implementation

Two implementations are provided:

| Notebook | Framework | Description |
|----------|-----------|-------------|
| [`2D_deepxde.ipynb`](2D_deepxde.ipynb) | DeepXDE | High-level PINNs API — concise and easy to extend |
| [`2D_tensorflow.ipynb`](2D_tensorflow.ipynb) | TensorFlow 2.x | Low-level implementation — full transparency over architecture and training |

### Prerequisites

```bash
pip install tensorflow deepxde numpy matplotlib
```

---

## Repository Structure

```
PINNs-in-Catalytic-Reactor/
├── 2D_deepxde.ipynb                        # PINNs via DeepXDE
├── 2D_tensorflow.ipynb                     # PINNs via raw TensorFlow
├── Img/
│   ├── CFD/                                # CFD reference: u, v, p, T
│   └── PINNs/                              # PINNs output: u, v, p, T
├── Poster-PINNs.png                        # Research poster
├── Thesis_PINNs_in_Catalytic_Reactor.pdf   # Full project report
└── Proposition-de-sujet-GMM-PIR.pdf        # Original project proposal
```

---

## Project Report

**[Full Report (PDF)](Thesis_PINNs_in_Catalytic_Reactor.pdf)**

Submitted as a *Projet Innovation Recherche (PIR)* in the 4th year of the engineering program at **INSA Toulouse**.

---

## Team

| Name | Role |
|------|------|
| Mai Dinh Nam | Researcher |
| Nguyen Phuc Luan | Researcher |
| Thai Doan Kien | Researcher |
| Mrs. Marie | Academic Advisor |
| Ms. Liantsoa | Academic Advisor |

---

## Key Findings

- PINNs successfully model **coupled fluid flow and heat transfer** in a 2D channel without any mesh or labeled simulation data
- A **hybrid Adam + L-BFGS** training strategy is critical for convergence on multi-physics problems
- Dimensionless formulation improves optimization stability significantly
- Results validate PINNs as a viable surrogate for CFD in process engineering contexts

---

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686–707.
- Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021). *DeepXDE: A deep learning library for solving differential equations.* SIAM Review, 63(1), 208–228.
