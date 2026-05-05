# Fourier Features based Physics-Informed Neural Network (PINN) for Steady-State Navier-Stokes Flow

**Access the full convergence plots from here: [Weights and Biases](https://api.wandb.ai/links/mehul22-iiser-thiruvananthapuram/wdoooei9)**

- **To re-run the notebook, please change the data paths wherever needed.**

---

## Model Performance & Validation with Ground Truth

The network was rigorously evaluated against a high-fidelity Finite Element Method (FEM) ground truth generated via FEniCS. 

| Physical Field | Relative $L_2$ Error |
| :--- | :--- |
| **U-Velocity ($u$)** | **1.16%** |
| **V-Velocity ($v$)** | **4.97%** |
| **Pressure ($p$)** | **4.87%** |

<p align="center">
  <img src="predictio_30000.png" alt="Model Predictions" width="80%">
  <br>
  <em>Figure 1: High-fidelity flow field predictions (U, V, P) closely matching the FEniCS FEM ground truth.</em>
</p>

<p align="center">
  <img src="error_report_30000.png" alt="Spatial Error Maps" width="100%">
  <br>
  <em>Figure 2: Absolute spatial error maps demonstrating successful resolution of upstream pressure and boundary layer physics.</em>
</p>

<p align="center">
  <img src="sidebyside.png" alt="Side by Side Comparison" width="100%">
  <br>
  <em>Figure 3: Side-by-side comparison of the FEniCS generated Ground Truth vs the PINN predictions.</em>
</p>

---

## Project Overview

This repository contains the implementation of a continuous, mesh-free Physics-Informed Neural Network designed to solve the steady-state incompressible Navier-Stokes equations for fluid flow around a 2D stationary obstacle (an ellipse/superellipse) in a rectangular channel at $Re=4$. 

The architecture is engineered to overcome strict mathematical bottlenecks associated with PDE optimization, specifically targeting stability, spectral bias, and exact boundary enforcement.

---

## Implementation Details

The `Main.ipynb` notebook implements several advanced Scientific Machine Learning (SciML) techniques to stabilize the PDE optimization landscape:

### 1. Defeating Spectral Bias via Spatial Fourier Features
Standard coordinate-based MLPs inherently favor smooth, low-frequency functions, causing them to "smear" high-frequency localized physics like fluid wakes and stagnation points. 
* **Implementation:** Before passing the $(x, y)$ coordinates to the MLP, they are projected into a 64-dimensional high-frequency latent space using a static Gaussian matrix. This mathematically bypasses spectral bias and allows the network to capture sharp gradients.

### 2. Solving Mode Collapse via Non-Dimensionalization
Standard optimizers often fail to balance the minute pressure gradients against larger convective velocity gradients, leading to degenerate solutions (e.g., a zero-pressure field).
* **Implementation:** The governing Navier-Stokes equations were rigorously non-dimensionalized using characteristic scaling parameters. By forcing the dynamic pressure to an $\mathcal{O}(1)$ scale, the PDE loss function was homogenized, balancing the backpropagated gradients and successfully recovering the coupled velocity-pressure physics.

### 3. Exact Boundary Enforcement via Signed Distance Functions (SDF)
To strictly enforce the complex geometry of the obstacle without relying entirely on soft loss penalties:
* **Implementation:** The analytical Signed Distance Function (SDF) of the obstacle was computed and concatenated as an input feature to the network. This provides explicit geometric awareness, accelerating the convergence of the no-slip boundary constraints.

### 4. Mitigating Data Deserts via Biased LHS Sampling
Random uniform sampling creates empty spaces where the network fails to learn the physics.
* **Implementation:** Latin Hypercube Sampling (LHS) was used for space-filling coverage. The sampling was heavily biased, forcing **40% of the collocation points** to spawn strictly along the boundary of the obstacle to accurately resolve the boundary layer interactions.

---

## Repository Structure

* `Main.ipynb`: The core Jupyter Notebook containing the data pipeline, FEniCS data loading, network architecture (Fourier + MLP), custom PDE loss functions, and the training loop.
* `Ground_Truth_fenics.py`: The Finite Element Method (FEM) script used to generate the high-fidelity ground truth dataset for rigorous quantitative validation.
* `predictio_30000.png` & `error_report_30000.png`: Visual evaluation artifacts.

---

## Future Work:

1. Further work on to apply better state of the art techniques for this problem like operator learning approaches and adding the "time dimension" to the problem.
2. Further develop the work for applications to Digital Twins, where I aim to make the model more adaptive and learn continually from the incoming data without parameter-heavy re-training.
