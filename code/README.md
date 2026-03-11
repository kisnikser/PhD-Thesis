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

### Chapter 4: Scaling

```bash
cd scaling
python run_experiments.py      # m*(ε)
python visualize.py
python run_scaling_law.py      # MLP vs CNN
python visualize_scaling_law.py
```

## Output files

Results are saved in `output/`:
- `hessian/` — `hessian_figures.pdf`
- `hessian_cnn/` — `hessian_figures.pdf`
- `landscape/` — `landscape_convergence.pdf`, `hessian_spectrum.pdf`, `loss_surface_2d.pdf`, `loss_surface_3d.pdf`
- `scaling/` — `delta_vs_k.pdf`, `sufficient_sample_size.pdf`, `mlp_vs_cnn_loss.pdf`, `scaling_law_fit.pdf`
