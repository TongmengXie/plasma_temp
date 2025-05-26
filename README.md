# Physics-Informed Neural Networks for Tokamak Plasma Evolution

This project demonstrates the application of Physics-Informed Neural Networks (PINNs) to predict the temporal evolution of plasma temperature profiles in tokamak fusion devices. The model combines data-driven learning with physical constraints from plasma transport equations.

## Overview

The project implements a PINN to learn and predict the spatio-temporal evolution of plasma temperature, constrained by the heat diffusion equation:

```
∂T(r,t)/∂t = D ∇²T(r,t) + S(r,t)
```

where:
- T(r,t) is the temperature profile
- r is the minor radius
- t is time
- D is thermal diffusivity
- S(r,t) represents heating/cooling sources

## Project Structure

```
.
├── codes/
│   ├── PINN_Plasma_Evolution_Prediction.ipynb   # Main PINN implementation
│   ├── pinn_model.pth                          # Trained model weights
│   └── realistic_plasma_data.py                # Realistic data generation
├── figs_tabs/                                  # Generated figures and tables
├── models/                                     # Model checkpoints
└── README.md                                   # This file
```

## Features

- Physics-informed neural network implementation in PyTorch
- Realistic plasma temperature profile generation including:
  - Core temperature peaking
  - Edge pedestal
  - Time-dependent heating and cooling phases
- Comparison with traditional Gaussian Process regression
- Visualization of predictions and PDE residuals

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- scikit-learn (for Gaussian Process comparison)

## Usage

1. Open `codes/PINN_Plasma_Evolution_Prediction.ipynb` in Jupyter
2. Run all cells to:
   - Generate training data
   - Train the PINN model
   - Compare with Gaussian Process regression
   - Visualize results

## Results

The model demonstrates:
- Accurate prediction of plasma temperature evolution
- Physics-consistent solutions (verified by PDE residuals)
- Comparison between physics-informed and pure data-driven approaches

Results are visualized in `figs_tabs/` including PDE residual plots and prediction comparisons.

## References

1. Raissi et al., "Physics-informed neural networks," J. Comput. Phys., 2019
2. Karniadakis et al., "Physics-informed machine learning," Nat. Rev. Phys., 2021
3. Seo et al., "Avoiding fusion plasma tearing instability with deep reinforcement learning," Nature, 2023
