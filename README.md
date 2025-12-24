
# Spectral Analysis and Structural Dynamics — Eigenvalues in Launch Stability

A companion implementation for the paper "Spectral Analysis and Structural Dynamics: The Role of Eigenvalues in Spacecraft Launch Stability". This repository contains code to reproduce the numerical examples and visualizations used for modal analysis, stability checks, and pogo (structural–propulsion) coupling studies.

**Repository contents:**
- `eigenvalue_analysis.py`: Main script implementing matrix assembly, eigenvalue analysis, damping estimation, Hurwitz stability check, Pogo frequency separation and simulation, ERA-based modal identification, and plotting routines.
- `13524112_Richard Samuel Simanullang_Makalah Algeo.pdf`: The paper describing the theory, methodology, and results (see the Paper section below).
- `results/`: Folder intended for storing generated figures and data outputs.

**Highlights**
- Solve generalized eigenvalue problems for a lumped DOF launch-vehicle model.
- Compute natural frequencies, mode shapes, and modal damping ratios.
- Perform Hurwitz stability checks via state-space eigenanalysis.
- Evaluate NASA-style Pogo frequency separation and simulate coupled structural–propulsion oscillations.
- Simple ERA (Eigensystem Realization Algorithm) routine for modal identification from impulse responses.

## Requirements
- Python 3.8+
- numpy
- scipy
- matplotlib

Install dependencies with:

```bash
pip install -r requirements.txt
```

(If you don't have a `requirements.txt`, run `pip install numpy scipy matplotlib`.)

## Quick Start / Usage

Run the main analysis script:

```bash
python eigenvalue_analysis.py
```

What the script does when run:
- Constructs simplified mass `M`, stiffness `K`, and damping `C` matrices for a 5-DOF lumped model.
- Solves the generalized eigenvalue problem `K v = λ M v` to obtain modal frequencies and shapes.
- Computes damping ratios from the state-space eigenvalues and evaluates Hurwitz stability.
- Performs Pogo safety checks against a nominal propulsion frequency and runs two time-domain simulations (resonant and safe cases).
- Saves visualizations to PNG files in the repository root: `mode_shapes.png`, `pogo_analysis.png`, `stability_analysis.png`, and `pogo_response.png`.

## Mapping to the paper
- The theoretical background, assumptions, and derivations are in the paper file: `13524112_Richard Samuel Simanullang_Makalah Algeo.pdf`.
- In code:
	- Matrix assembly: `create_launch_vehicle_matrices()`
	- Eigen analysis: `solve_eigenvalue_problem()`
	- Damping & stability: `calculate_damping_ratios()`, `check_stability()`
	- Pogo checks & simulation: `pogo_frequency_separation()`, `simulate_pogo_response()`
	- ERA modal ID: `era_modal_identification()`
	- Visualization helpers: `visualize_mode_shapes()`, `visualize_pogo_analysis()`, `visualize_stability()`

## Results
Running the script generates:
- `mode_shapes.png` — normalized first mode shapes
- `pogo_analysis.png` — frequency comparison and safety assessment
- `stability_analysis.png` — eigenvalue scatter plot (complex plane)
- `pogo_response.png` — time-domain comparison of resonant vs safe cases

## Paper
The repository includes the PDF of the paper: `13524112_Richard Samuel Simanullang_Makalah Algeo.pdf`. Refer to it for derivations, experiments, and discussion that motivated the implemented examples.

## Next steps I can take
- Extract the paper abstract, introduction, and key figures to include in this README (I can parse the PDF if you want me to),
- Add a `requirements.txt`, or convert the project to a proper package/CLI,
- Run the script here and attach the generated images to `results/`.

If you'd like, I can extract specific sections from the PDF and expand the README (abstract, methodology, key equations, conclusions).

---

Created from the repository code and the included paper file.

