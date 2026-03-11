"""
Plot Hessian eigenvalue spectra from precomputed data.
Shows spectra at different training epochs.
"""
import argparse
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Path to spectrum_data.json")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir) if args.out_dir else _repo_root / "code" / "output" / "landscape"
    data_file = Path(args.data) if args.data else out_dir / "spectrum_data.json"
    
    with open(data_file) as f:
        data = json.load(f)
    
    spectra = data["spectra"]
    epochs = sorted([int(e) for e in spectra.keys()])
    
    # Colors for different epochs
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(epochs)))
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for i, epoch in enumerate(epochs):
        eigenvalues = np.array(spectra[str(epoch)]["eigenvalues"])
        loss = spectra[str(epoch)]["loss"]
        indices = np.arange(1, len(eigenvalues) + 1)
        
        label = f"Epoch {epoch} (Loss={loss:.2f})"
        ax.semilogy(indices, eigenvalues, "o-", markersize=6, 
                    color=colors[i], label=label, linewidth=2)
    
    ax.set_xlabel("Eigenvalue index $i$", fontsize=16)
    ax.set_ylabel("Eigenvalue $\\lambda_i$", fontsize=16)
    # ax.set_title("Hessian Eigenvalue Spectrum During Training", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_dir / "hessian_spectrum.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "hessian_spectrum.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Spectrum plot saved to {out_dir / 'hessian_spectrum.pdf'}")


if __name__ == "__main__":
    main()
