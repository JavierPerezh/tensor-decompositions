import os
from pathlib import Path
import matplotlib.pyplot as plt

# Paleta apta para daltónicos (Okabe-Ito)
COLORS = {
    "cp": "#0072B2",       # Azul para CP / PARAFAC
    "pca": "#D55E00",      # Naranja para PCA aplanado
    "tucker": "#009E73",   # Verde para Tucker / HOSVD
    "hooi": "#CC79A7",     # Magenta para HOOI
    "bound": "#000000",    # Negro para cotas teóricas (ej. sqrt(3) o pi/4)
}

COLOR_CP = COLORS["cp"]
COLOR_PCA = COLORS["pca"]
COLOR_TUCKER = COLORS["tucker"]
COLOR_EXTRA = COLORS["hooi"]
COLOR_SKY = "#56B4E9"
GEN_DIR = Path("paper") / "generated"


def apply_paper_style():
    """Configura Matplotlib para exportar figuras limpias estilo LaTeX."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })


def save_paper_figure(fig: plt.Figure, filename: str, out_dir: str = "figures_export"):
    """Guarda la figura en PDF vectorial lista para subir a Overleaf/figures/."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{filename}.pdf")
    fig.savefig(path, format="pdf", bbox_inches="tight")
    print(f"[OK] Figura exportada en: {path}")


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Compatibility wrapper for Persona-A experiments."""
    save_paper_figure(fig, filename)