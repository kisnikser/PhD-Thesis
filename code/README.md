# Code

## Structure

```
code/
├── hessian/      # Chapter 2: Hessian spectrum
├── landscape/    # Chapter 3: landscape convergence
├── scaling/      # Chapter 4: scaling laws
├── shared/       # common utilities (data, models)
└── output/       # experiment results
```

## Installation

```bash
git clone https://github.com/kisnikser/PhD-Thesis.git
cd PhD-Thesis/code
pip install -r requirements.txt
```

### Chapter 2: Hessian

```bash
cd hessian
python run_experiments.py      # MLP
python run_experiments_cnn.py  # CNN
python visualize.py
```

### Chapter 3: Landscapes

```bash
cd landscape
python run_experiments.py           # criteria Δ₁, Δ₂
python visualize.py
python compute_spectrum_data.py     # Hessian spectrum
python plot_spectrum.py
python compute_surface_data.py      # 2D/3D surface
python plot_surface_2d.py
python visualize_surface_3d.py
```

### Chapter 4: Scaling Laws and Sample Size Criteria

```bash
cd scaling

# Experiment 1: Curvature vs Sufficient Sample Size
python run_experiments.py      # computes Delta_1, Delta_2, curvature proxy M_G
python visualize.py            # plots: Delta(m), M_G vs m*, correlation analysis

# Experiment 2: Scaling Law Interpretation
python run_scaling_law.py      # computes train/test gap, curvature
python visualize_scaling_law.py # plots: E_hat(m), scaling fit, C_A vs M_G
```

## Output files

Results are saved in `output/`:
- `hessian/` — `hessian_figures.pdf`
- `hessian_cnn/` — `hessian_figures.pdf`
- `landscape/` — `landscape_convergence.pdf`, `hessian_spectrum.pdf`, `loss_surface_2d.pdf`, `loss_surface_3d.pdf`
- `scaling/` — `exp1_delta2_convergence.pdf`, `exp1_depth_analysis.pdf`, `exp1_m_star_vs_params.pdf`
