import os
import json
import urllib.request
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorly as tl
from tensorly.decomposition import parafac, non_negative_parafac, tucker
import io
import zipfile

from src.tensor_utils import (
    unfold_kb,
    cp_to_tensor_kb,
    match_and_score_factors,
    corcondia,
)
from src.plot_style import COLORS, apply_paper_style, save_paper_figure


# ==============================================================================
# EXPERIMENTO E1: ROTACIÓN VS. UNICIDAD (PCA APLANADO VS. TUCKER VS. CP-ALS)
# ==============================================================================

def _extract_flattened_pca_factors(X: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[float]]:
    """
    Aplica PCA estándar sobre el despliegue mode-1 X_{(1)} (centrado por encuestados),
    remodela los r primeros loadings a matrices J x K bajo Kolda-Bader,
    y extrae su mejor aproximación de rango 1 vía SVD.
    """
    I, J, K = X.shape
    X1 = unfold_kb(X, mode=0)
    X1_centered = X1 - np.mean(X1, axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X1_centered, full_matrices=False)
    V_tilde = Vt[:r].T  # (J*K, r)

    B_pca = np.zeros((J, r))
    C_pca = np.zeros((K, r))
    reshaped_mats = []
    non_rank1_ratios = []

    for m in range(r):
        # En Kolda-Bader mode-1, el índice es j + k*J -> reshape(K, J).T recupera M_m (J x K)
        M_m = V_tilde[:, m].reshape(K, J).T
        reshaped_mats.append(M_m)
        u, s, vt = np.linalg.svd(M_m, full_matrices=False)
        B_pca[:, m] = u[:, 0]
        C_pca[:, m] = vt[0, :]
        # Energía relativa fuera del primer valor singular (0 si es rango 1 exacto)
        leak = float(np.sqrt(np.sum(s[1:] ** 2)) / (np.linalg.norm(s) + 1e-15))
        non_rank1_ratios.append(leak)

    return B_pca, C_pca, reshaped_mats, non_rank1_ratios


def run_experiment_e1(seed: int = 42, n_trials: int = 25) -> dict:
    """
    Experimento E1: Compara recuperación de B y C mediante PCA aplanado, Tucker/HOSVD
    y CP-ALS bajo ruido creciente (incluyendo el caso base I=50, J=8, K=4, r=3 al 10%).
    """
    apply_paper_style()
    rng = np.random.default_rng(seed)
    I, J, K, r = 50, 8, 4, 3
    noise_levels = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

    results = {
        "noise_levels": noise_levels,
        "cp_B_mean": [], "cp_B_std": [], "cp_C_mean": [], "cp_C_std": [],
        "tucker_B_mean": [], "tucker_B_std": [], "tucker_C_mean": [], "tucker_C_std": [],
        "pca_B_mean": [], "pca_B_std": [], "pca_C_mean": [], "pca_C_std": [],
        "pca_rank1_leak_10pct": None,
    }

    snapshot_10pct = None

    for nl in noise_levels:
        phi_cp_B, phi_cp_C = [], []
        phi_tk_B, phi_tk_C = [], []
        phi_pca_B, phi_pca_C = [], []
        leaks_this_level = []

        for trial in range(n_trials):
            # Factores realistas: A gaussiano, B con estructura positiva/mixta, C tendencias temporales
            A_true = rng.standard_normal((I, r))
            B_true = rng.standard_normal((J, r))
            C_true = rng.uniform(0.3, 1.5, size=(K, r)) + 0.2 * rng.standard_normal((K, r))

            X_clean = cp_to_tensor_kb(A_true, B_true, C_true)
            E = rng.standard_normal((I, J, K))
            sigma = nl * (np.linalg.norm(X_clean) / np.linalg.norm(E))
            X = X_clean + sigma * E

            # (a) PCA sobre X_{(1)}
            B_pca, C_pca, reshaped_pca, leaks = _extract_flattened_pca_factors(X, r)
            b_sc, _ = match_and_score_factors(B_true, B_pca)
            c_sc, _ = match_and_score_factors(C_true, C_pca)
            phi_pca_B.append(b_sc)
            phi_pca_C.append(c_sc)
            leaks_this_level.append(np.mean(leaks))

            # (b) Tucker / HOSVD (solo fija subespacios)
            _, tk_factors = tucker(X, rank=[r, r, r], init="svd")
            b_tk, _ = match_and_score_factors(B_true, tk_factors[1])
            c_tk, _ = match_and_score_factors(C_true, tk_factors[2])
            phi_tk_B.append(b_tk)
            phi_tk_C.append(c_tk)

            # (c) CP-ALS
            cp_res = parafac(X, rank=r, init="svd", n_iter_max=500, tol=1e-8, random_state=seed + trial)
            _, cp_factors = cp_res
            b_cp, perm_B = match_and_score_factors(B_true, cp_factors[1])
            c_cp, _ = match_and_score_factors(C_true, cp_factors[2])
            phi_cp_B.append(b_cp)
            phi_cp_C.append(c_cp)

            if np.isclose(nl, 0.10) and trial == 0:
                # Guardar caso ilustrativo al 10% de ruido para mapas de calor J x K
                B_cp_ord = cp_factors[1][:, perm_B]
                C_cp_ord = cp_factors[2][:, perm_B]
                true_mats = [np.outer(B_true[:, m], C_true[:, m]) for m in range(r)]
                cp_mats = []
                for m in range(r):
                    M_cp = np.outer(B_cp_ord[:, m], C_cp_ord[:, m])
                    # Alinear signo y escala con true_mats[m] para comparación visual
                    sign = np.sign(np.sum(true_mats[m] * M_cp)) or 1.0
                    M_cp = sign * M_cp * (np.linalg.norm(true_mats[m]) / (np.linalg.norm(M_cp) + 1e-15))
                    cp_mats.append(M_cp)

                pca_mats = []
                for m in range(r):
                    M_p = reshaped_pca[m]
                    # Emparejar con la componente verdadera más cercana
                    sims = [abs(np.sum(true_mats[q] * M_p)) / (np.linalg.norm(true_mats[q]) * np.linalg.norm(M_p) + 1e-15) for q in range(r)]
                    best_q = int(np.argmax(sims))
                    sign = np.sign(np.sum(true_mats[best_q] * M_p)) or 1.0
                    M_p_scaled = sign * M_p * (np.linalg.norm(true_mats[best_q]) / (np.linalg.norm(M_p) + 1e-15))
                    pca_mats.append((best_q, M_p_scaled, leaks[m]))
                pca_mats.sort(key=lambda item: item[0])

                snapshot_10pct = {
                    "true_mats": true_mats,
                    "cp_mats": cp_mats,
                    "pca_mats": [(item[1], item[2]) for item in pca_mats],
                }

        for prefix, arr_B, arr_C in [
            ("cp", phi_cp_B, phi_cp_C),
            ("tucker", phi_tk_B, phi_tk_C),
            ("pca", phi_pca_B, phi_pca_C),
        ]:
            results[f"{prefix}_B_mean"].append(float(np.mean(arr_B)))
            results[f"{prefix}_B_std"].append(float(np.std(arr_B)))
            results[f"{prefix}_C_mean"].append(float(np.mean(arr_C)))
            results[f"{prefix}_C_std"].append(float(np.std(arr_C)))

        if np.isclose(nl, 0.10):
            results["pca_rank1_leak_10pct"] = float(np.mean(leaks_this_level))

    # --- Figura 1: Curvas de congruencia phi vs. nivel de ruido ---
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    pct = [100 * x for x in noise_levels]

    for ax, mode_key, title in [
        (axes[0], "B", r"Item Loadings ($\mathbf{B} \in \mathbb{R}^{8 \times 3}$)"),
        (axes[1], "C", r"Wave Intensities ($\mathbf{C} \in \mathbb{R}^{4 \times 3}$)"),
    ]:
        for key, label, color, marker in [
            ("cp", "CP-ALS (Unconstrained)", COLORS["cp"], "o"),
            ("tucker", "Tucker / HOSVD (Subspace basis)", COLORS["tucker"], "s"),
            ("pca", r"Flattened PCA on $\mathbf{X}_{(1)}$", COLORS["pca"], "^"),
        ]:
            m_arr = np.array(results[f"{key}_{mode_key}_mean"])
            s_arr = np.array(results[f"{key}_{mode_key}_std"])
            ax.plot(pct, m_arr, marker=marker, color=color, linewidth=2, label=label)
            ax.fill_between(pct, np.clip(m_arr - s_arr, 0, 1), np.clip(m_arr + s_arr, 0, 1), color=color, alpha=0.15)

        ax.set_title(title)
        ax.set_xlabel("Relative Noise Level (%)")
        ax.set_ylim(0.45, 1.03)
        if mode_key == "B":
            ax.set_ylabel(r"Mean Tucker Congruence $\phi$")
        ax.legend(loc="lower left")

    fig.tight_layout()
    save_paper_figure(fig, "fig_E1_congruence")
    plt.close(fig)

    # --- Figura 2: Mapas de calor J x K (8 x 4) al 10% de ruido ---
    if snapshot_10pct is not None:
        fig2, axes2 = plt.subplots(3, 3, figsize=(9.5, 7.2), layout="constrained")
        vmax = max(np.max(np.abs(M)) for M in snapshot_10pct["true_mats"])

        for m in range(r):
            # Fila 0: Verdadero b_m c_m^T
            im = axes2[0, m].imshow(snapshot_10pct["true_mats"][m], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            axes2[0, m].set_title(f"True Component {m+1}\n(Rank = 1)")
            if m == 0:
                axes2[0, m].set_ylabel("Item $j$ (1..8)")

            # Fila 1: CP-ALS
            axes2[1, m].imshow(snapshot_10pct["cp_mats"][m], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            axes2[1, m].set_title(f"CP-ALS Component {m+1}\n(Rank = 1, leak = 0.0%)")
            if m == 0:
                axes2[1, m].set_ylabel("Item $j$ (1..8)")

            # Fila 2: PCA aplanado
            M_pca, leak_m = snapshot_10pct["pca_mats"][m]
            axes2[2, m].imshow(M_pca, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            axes2[2, m].set_title(f"Flattened PCA {m+1}\n(Non-rank-1 energy = {100*leak_m:.1f}%)")
            axes2[2, m].set_xlabel("Wave $k$ (1..4)")
            if m == 0:
                axes2[2, m].set_ylabel("Item $j$ (1..8)")

        fig2.colorbar(im, ax=axes2.ravel().tolist(), shrink=0.65, label="Loading Value")
        save_paper_figure(fig2, "fig_E1_reshaped_loadings")
        plt.close(fig2)

    print("\n=== RESUMEN EXPERIMENTO E1 (10% RUIDO) ===")
    idx10 = noise_levels.index(0.10)
    print(f"CP-ALS     -> phi_B = {results['cp_B_mean'][idx10]:.4f} ± {results['cp_B_std'][idx10]:.4f} | phi_C = {results['cp_C_mean'][idx10]:.4f} ± {results['cp_C_std'][idx10]:.4f}")
    print(f"Tucker     -> phi_B = {results['tucker_B_mean'][idx10]:.4f} ± {results['tucker_B_std'][idx10]:.4f} | phi_C = {results['tucker_C_mean'][idx10]:.4f} ± {results['tucker_C_std'][idx10]:.4f}")
    print(f"PCA Aplan. -> phi_B = {results['pca_B_mean'][idx10]:.4f} ± {results['pca_B_std'][idx10]:.4f} | phi_C = {results['pca_C_mean'][idx10]:.4f} ± {results['pca_C_std'][idx10]:.4f}")
    print(f"Energía fuera de rango-1 en PCA aplanado (10% ruido): {100*results['pca_rank1_leak_10pct']:.2f}%")
    return results


# ==============================================================================
# EXPERIMENTO E4(a): FLUORESCENCIA DE RASMUS BRO (5 x 201 x 61)
# ==============================================================================

def load_bro_fluorescence_data(data_dir: str = "data") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga el dataset clásico de fluorescencia de Rasmus Bro (5 muestras x 201 emisión x 61 excitación).
    """
    os.makedirs(data_dir, exist_ok=True)
    mat_path = os.path.join(data_dir, "amino.mat")

    if not os.path.exists(mat_path):
        archive_url = "https://sid.erda.dk/share_redirect/AWqL5T6JCZ"
        downloaded_mat = None

        try:
            print(f"Descargando dataset de fluorescencia de Bro desde {archive_url} ...")
            archive_bytes = urllib.request.urlopen(archive_url, timeout=60).read()
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as outer_zip:
                nested_bytes = outer_zip.read("Amino_Acid_fluo.zip")
            with zipfile.ZipFile(io.BytesIO(nested_bytes)) as inner_zip:
                downloaded_mat = inner_zip.read("amino.mat")

            with open(mat_path, "wb") as mat_file:
                mat_file.write(downloaded_mat)
            print(f"Dataset guardado en: {mat_path}")
        except Exception as e:
            if os.path.exists(mat_path):
                os.remove(mat_path)
            raise RuntimeError(
                "No se pudo descargar amino.mat automáticamente. "
                "Comprueba tu conexión o coloca el archivo manualmente como data/amino.mat."
            ) from e

    mat = sio.loadmat(mat_path)

    if "X" in mat:
        X_values = np.asarray(mat["X"], dtype=float)
        if X_values.shape == (5, 12261):
            X = X_values.reshape((5, 201, 61), order="F")
        elif X_values.shape == (5, 61, 201):
            X = np.transpose(X_values, (0, 2, 1))
        elif X_values.ndim == 3:
            X = X_values.astype(float)
        else:
            X = None
    else:
        X = None

    if X is None:
        X = None
        for key, val in mat.items():
            if isinstance(val, np.ndarray) and val.ndim == 3:
                X = val.astype(float)
                break

    if X is None:
        raise ValueError("No se encontró el tensor de fluorescencia dentro de data/amino.mat.")

    if X.shape == (5, 61, 201):
        X = np.transpose(X, (0, 2, 1))
    if X.shape != (5, 201, 61):
        raise ValueError(f"Forma inesperada del tensor de fluorescencia: {X.shape}")

    em_axis = mat["EmAx"].ravel() if "EmAx" in mat else np.linspace(250, 450, 201)
    ex_axis = mat["ExAx"].ravel() if "ExAx" in mat else np.linspace(240, 300, 61)
    return X, em_axis, ex_axis


def run_experiment_e4a(seed: int = 42) -> dict:
    """
    Experimento E4(a): Descomposición CP-ALS y NN-CP sobre fluorescencia de Bro (5x201x61).
    Verifica unicidad empírica entre múltiples inicializaciones y recuperación de espectros físicos.
    """
    apply_paper_style()
    X, em_axis, ex_axis = load_bro_fluorescence_data()

    # 1. Estabilidad / Unicidad empírica entre 10 inicializaciones aleatorias de CP-ALS sin restricciones
    runs = []
    for s in range(10):
        weights, factors = parafac(X, rank=3, init="random", n_iter_max=1000, tol=1e-9, random_state=seed + s)
        err = np.linalg.norm(X - tl.cp_to_tensor((weights, factors))) / np.linalg.norm(X)
        runs.append((err, weights, factors))

    runs.sort(key=lambda item: item[0])
    best_err, best_w, best_factors = runs[0]

    # Comparar congruencia de los espectros de emisión y excitación entre todas las corridas vs la mejor
    em_congruences = []
    ex_congruences = []
    for err, _, facs in runs[1:]:
        phi_em, _ = match_and_score_factors(best_factors[1], facs[1])
        phi_ex, _ = match_and_score_factors(best_factors[2], facs[2])
        em_congruences.append(phi_em)
        ex_congruences.append(phi_ex)

    # 2. CP con restricción de no negatividad (NN-CP)
    nn_w, nn_factors = non_negative_parafac(X, rank=3, init="svd", n_iter_max=1000, tol=1e-9, random_state=seed)
    nn_err = np.linalg.norm(X - tl.cp_to_tensor((nn_w, nn_factors))) / np.linalg.norm(X)

    # Alinear columnas de NN-CP con best_factors (unconstrained CP) y ordenar por pico de emisión
    phi_unconst_vs_nn_em, perm_nn = match_and_score_factors(best_factors[1], nn_factors[1])
    phi_unconst_vs_nn_ex, _ = match_and_score_factors(best_factors[2], nn_factors[2])

    A_cp, B_cp, C_cp = best_factors
    A_nn = nn_factors[0][:, perm_nn]
    B_nn = nn_factors[1][:, perm_nn]
    C_nn = nn_factors[2][:, perm_nn]

    # Corregir signo de CP sin restricciones para que los picos espectrales sean positivos
    for m in range(3):
        if np.sum(B_cp[:, m]) < 0:
            B_cp[:, m] *= -1
            A_cp[:, m] *= -1
        if np.sum(C_cp[:, m]) < 0:
            C_cp[:, m] *= -1
            A_cp[:, m] *= -1

    # Ordenar los 3 compuestos por longitud de onda del pico de emisión (Tyrosine ~305nm, Tryptophan ~350nm, Phenylalanine ~282nm)
    peak_order = np.argsort(em_axis[np.argmax(B_cp, axis=0)])
    A_cp, B_cp, C_cp = A_cp[:, peak_order], B_cp[:, peak_order], C_cp[:, peak_order]
    A_nn, B_nn, C_nn = A_nn[:, peak_order], B_nn[:, peak_order], C_nn[:, peak_order]

    # Normalizar espectros a pico = 1 para visualización limpia
    B_cp_plot = B_cp / np.max(B_cp, axis=0, keepdims=True)
    C_cp_plot = C_cp / np.max(C_cp, axis=0, keepdims=True)
    B_nn_plot = B_nn / np.max(B_nn, axis=0, keepdims=True)
    C_nn_plot = C_nn / np.max(C_nn, axis=0, keepdims=True)

    # --- Figura E4(a): 3 paneles (Muestras, Emisión, Excitación) ---
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    comp_colors = [COLORS["cp"], COLORS["tucker"], COLORS["pca"]]
    comp_names = ["Fluorophore 1 (Phe-like)", "Fluorophore 2 (Tyr-like)", "Fluorophore 3 (Trp-like)"]

    # Panel 1: Concentraciones por muestra (barras)
    x_samples = np.arange(1, 6)
    width = 0.25
    A_cp_norm = A_cp / np.max(A_cp, axis=0, keepdims=True)
    for m in range(3):
        axes[0].bar(x_samples + (m - 1) * width, A_cp_norm[:, m], width=width, color=comp_colors[m], label=comp_names[m])
    axes[0].set_title("Sample Loadings (Mode 1)")
    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel("Normalized Concentration")
    axes[0].set_xticks(x_samples)
    axes[0].legend(fontsize=8)

    # Panel 2: Espectros de Emisión (201 longitudes de onda)
    for m in range(3):
        axes[1].plot(em_axis, B_cp_plot[:, m], color=comp_colors[m], linewidth=2, label=f"CP {m+1}")
        axes[1].plot(em_axis, B_nn_plot[:, m], color=comp_colors[m], linestyle="--", linewidth=1.3, alpha=0.85)
    axes[1].axhline(0, color="black", linewidth=0.7, alpha=0.5)
    axes[1].set_title("Emission Spectra (Mode 2)\nSolid: Unconstrained CP | Dashed: NN-CP")
    axes[1].set_xlabel("Emission Wavelength (nm)")
    axes[1].set_ylabel("Normalized Intensity")

    # Panel 3: Espectros de Excitación (61 longitudes de onda)
    for m in range(3):
        axes[2].plot(ex_axis, C_cp_plot[:, m], color=comp_colors[m], linewidth=2, label=comp_names[m])
        axes[2].plot(ex_axis, C_nn_plot[:, m], color=comp_colors[m], linestyle="--", linewidth=1.3, alpha=0.85)
    axes[2].axhline(0, color="black", linewidth=0.7, alpha=0.5)
    axes[2].set_title("Excitation Spectra (Mode 3)\nSolid: Unconstrained CP | Dashed: NN-CP")
    axes[2].set_xlabel("Excitation Wavelength (nm)")
    axes[2].set_ylabel("Normalized Intensity")

    fig.tight_layout()
    save_paper_figure(fig, "fig_E4a_fluorescence")
    plt.close(fig)

    summary = {
        "cp_rel_error": float(best_err),
        "nn_cp_rel_error": float(nn_err),
        "restart_congruence_em_mean": float(np.mean(em_congruences)),
        "restart_congruence_ex_mean": float(np.mean(ex_congruences)),
        "unconstrained_vs_nn_phi_em": float(phi_unconst_vs_nn_em),
        "unconstrained_vs_nn_phi_ex": float(phi_unconst_vs_nn_ex),
        "corcondia_r3": corcondia(X, best_w, best_factors),
    }
    print("\n=== RESUMEN EXPERIMENTO E4(a) FLUORESCENCIA ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.6f}")
    return summary


# ==============================================================================
# EXPERIMENTO E4(b): CIENCIAS SOCIALES (WDI PAÍS x INDICADOR x AÑO)
# ==============================================================================

def load_social_science_tensor(data_dir: str = "data") -> tuple[np.ndarray, list[str], list[str], list[int]]:
    """
    Descarga y construye un tensor socioeconómico 3D (País x Indicador x Año)
    desde la API abierta del Banco Mundial (WDI) y lo guarda en data/wdi_social_tensor.npz.
    """
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, "wdi_social_tensor.npz")

    countries = [
        "USA", "CAN", "MEX", "BRA", "ARG", "CHL", "COL", "PER",
        "GBR", "DEU", "FRA", "ITA", "ESP", "SWE", "POL", "TUR",
        "ZAF", "NGA", "EGY", "KEN", "IND", "CHN", "JPN", "KOR",
        "IDN", "THA", "VNM", "AUS", "NZL", "SAU",
    ]
    indicators = {
        "NY.GDP.PCAP.KD": "GDP per capita",
        "SP.DYN.LE00.IN": "Life expectancy",
        "SP.DYN.IMRT.IN": "Infant mortality",
        "SP.URB.TOTL.IN.ZS": "Urban pop (%)",
        "IT.NET.USER.ZS": "Internet users (%)",
        "EG.FEC.RNEW.ZS": "Renewable energy (%)",
        "SL.TLF.CACT.FE.ZS": "Female labor (%)",
        "SH.XPD.CHEX.GD.ZS": "Health exp (% GDP)",
    }
    years = list(range(2012, 2022))  # 10 años: 2012..2021

    if os.path.exists(cache_path):
        loaded = np.load(cache_path, allow_pickle=True)
        return (
            loaded["X"],
            loaded["countries"].tolist(),
            loaded["indicator_names"].tolist(),
            loaded["years"].tolist(),
        )

    ind_codes = list(indicators.keys())
    ind_names = list(indicators.values())
    X = np.full((len(countries), len(ind_codes), len(years)), np.nan)
    country_str = ";".join(countries)

    print("Descargando indicadores del Banco Mundial (WDI) para E4(b)...")
    for j, code in enumerate(ind_codes):
        url = (
            f"https://api.worldbank.org/v2/country/{country_str}/indicator/{code}"
            f"?date={years[0]}:{years[-1]}&format=json&per_page=2000"
        )
        with urllib.request.urlopen(url, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        records = payload[1]
        for rec in records:
            iso3 = rec.get("countryiso3code")
            yr = int(rec.get("date"))
            val = rec.get("value")
            if iso3 in countries and yr in years and val is not None:
                i = countries.index(iso3)
                k = years.index(yr)
                X[i, j, k] = float(val)

    # Interpolación limpia a lo largo del tiempo (modo 2) para evitar NaN aislados
    for i in range(len(countries)):
        for j in range(len(ind_codes)):
            series = pd.Series(X[i, j, :])
            if series.isna().all():
                X[i, j, :] = np.nanmean(X[:, j, :])
            else:
                X[i, j, :] = series.interpolate(method="linear", limit_direction="both").to_numpy()

    np.savez(
        cache_path,
        X=X,
        countries=np.array(countries),
        indicator_names=np.array(ind_names),
        years=np.array(years),
    )
    return X, countries, ind_names, years


def preprocess_three_way(X: np.ndarray, scheme: str = "j_center_scale") -> np.ndarray:
    """
    Preprocesamiento tridimensional estándar (Bro & Smilde):
      - 'j_center_scale': centra a través de los países (modo 0) para cada (indicador, año)
        y escala por la desviación estándar global de cada indicador (modo 1).
      - 'double_center_scale': centra tanto a través de países (modo 0) como de años (modo 2),
        eliminando niveles base por país y tendencias globales, y escala por indicador.
    """
    X_proc = X.copy().astype(float)
    # Transformación logarítmica sobre PIB per cápita (indicador 0) para reducir asimetría
    X_proc[:, 0, :] = np.log10(np.maximum(X_proc[:, 0, :], 1.0))

    if scheme == "j_center_scale":
        X_proc = X_proc - np.mean(X_proc, axis=0, keepdims=True)
    elif scheme == "double_center_scale":
        X_proc = X_proc - np.mean(X_proc, axis=0, keepdims=True)
        X_proc = X_proc - np.mean(X_proc, axis=2, keepdims=True)
    else:
        raise ValueError("Esquema de preprocesamiento no reconocido.")

    # Escalado dentro del modo 1 (indicadores) para que todos tengan varianza comparable
    std_j = np.std(X_proc, axis=(0, 2), keepdims=True)
    X_proc = X_proc / (std_j + 1e-12)
    return X_proc


def run_experiment_e4b(seed: int = 42) -> dict:
    """
    Experimento E4(b): Análisis multiway sobre datos socioeconómicos del Banco Mundial.
    Evalúa preprocesamiento 3D, diagnóstico CORCONDIA para r=1..5 y compara
    PCA aplanado vs. Tucker3 vs. CP/PARAFAC.
    """
    apply_paper_style()
    X_raw, countries, ind_names, years = load_social_science_tensor()

    # 1. Comparar dos esquemas de centrado/escalado + selección de rango vía CORCONDIA
    ranks_grid = [1, 2, 3, 4, 5]
    cc_scheme_A = []
    cc_scheme_B = []
    rel_err_A = []

    X_A = preprocess_three_way(X_raw, scheme="j_center_scale")
    X_B = preprocess_three_way(X_raw, scheme="double_center_scale")

    models_A = {}
    for r in ranks_grid:
        w_A, fac_A = parafac(X_A, rank=r, init="svd", n_iter_max=1000, tol=1e-8, random_state=seed)
        cc_A = corcondia(X_A, w_A, fac_A)
        err_A = np.linalg.norm(X_A - tl.cp_to_tensor((w_A, fac_A))) / np.linalg.norm(X_A)
        cc_scheme_A.append(cc_A)
        rel_err_A.append(err_A)
        models_A[r] = (w_A, fac_A)

        w_B, fac_B = parafac(X_B, rank=r, init="svd", n_iter_max=1000, tol=1e-8, random_state=seed)
        cc_B = corcondia(X_B, w_B, fac_B)
        cc_scheme_B.append(cc_B)

    # Elegir r óptimo: mayor r con CORCONDIA > 60% en el esquema estándar
    valid_r = [r for r, cc in zip(ranks_grid, cc_scheme_A) if cc > 60.0]
    r_opt = max(valid_r) if valid_r else 2

    w_opt, fac_opt = models_A[r_opt]
    A_cp, B_cp, C_cp = fac_opt

    # Orientar signos de CP para interpretación positiva del desarrollo socioeconómico
    for m in range(r_opt):
        if B_cp[0, m] < 0:  # GDP per capita positivo
            B_cp[:, m] *= -1
            A_cp[:, m] *= -1

    # 2. Comparar con Tucker3 de rango (r_opt, r_opt, r_opt) y PCA aplanado en X_{(1)}
    tk_core, tk_fac = tucker(X_A, rank=[r_opt, r_opt, r_opt], init="svd")
    tk_err = np.linalg.norm(X_A - tl.tucker_to_tensor((tk_core, tk_fac))) / np.linalg.norm(X_A)

    B_pca, C_pca, reshaped_pca, pca_leaks = _extract_flattened_pca_factors(X_A, r_opt)
    X1_A = unfold_kb(X_A, 0)
    U_p, s_p, Vt_p = np.linalg.svd(X1_A, full_matrices=False)
    X1_pca_rec = (U_p[:, :r_opt] * s_p[:r_opt]) @ Vt_p[:r_opt, :]
    pca_err = np.linalg.norm(X1_A - X1_pca_rec) / np.linalg.norm(X1_A)

    # --- Figura E4(b): Curva CORCONDIA + Comparación de loadings de indicadores y años ---
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0))

    # Panel 1: CORCONDIA vs r bajo ambos esquemas de centrado
    axes[0].plot(ranks_grid, cc_scheme_A, marker="o", color=COLORS["cp"], linewidth=2, label="Mode-0 Center + Mode-1 Scale")
    axes[0].plot(ranks_grid, cc_scheme_B, marker="s", color=COLORS["tucker"], linewidth=2, linestyle="--", label="Double Center (Mode 0 & 2)")
    axes[0].axhline(60.0, color=COLORS["bound"], linestyle=":", linewidth=1.2, label="60% Consistency Threshold")
    axes[0].set_title("CORCONDIA Rank Selection")
    axes[0].set_xlabel("Number of CP Components ($r$)")
    axes[0].set_ylabel("Core Consistency (%)")
    axes[0].set_xticks(ranks_grid)
    axes[0].set_ylim(-20, 108)
    axes[0].legend(fontsize=8, loc="lower left")

    # Panel 2: Loadings de indicadores (Modo 2) en CP (r = r_opt)
    y_pos = np.arange(len(ind_names))
    bar_h = 0.35
    for m in range(min(r_opt, 2)):
        offset = (m - 0.5) * bar_h if r_opt >= 2 else 0.0
        axes[1].barh(y_pos + offset, B_cp[:, m], height=bar_h, color=[COLORS["cp"], COLORS["pca"]][m], label=f"CP Comp. {m+1}")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(ind_names, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_title(f"CP Indicator Loadings ($r={r_opt}$)")
    axes[1].set_xlabel("Normalized Loading")
    axes[1].legend(fontsize=8)

    # Panel 3: Evolución temporal (Modo 3) en CP vs PCA aplanado (rango-1 aproximado)
    for m in range(min(r_opt, 2)):
        c_cp_m = C_cp[:, m] / (np.linalg.norm(C_cp[:, m]) + 1e-15)
        c_pca_m = C_pca[:, m] / (np.linalg.norm(C_pca[:, m]) + 1e-15)
        if np.sum(c_cp_m * c_pca_m) < 0:
            c_pca_m *= -1
        col = [COLORS["cp"], COLORS["pca"]][m]
        axes[2].plot(years, c_cp_m, marker="o", color=col, linewidth=2, label=f"CP Time Mode {m+1}")
        axes[2].plot(years, c_pca_m, marker="^", color=col, linestyle="--", alpha=0.75, label=f"PCA Approx. {m+1}")
    axes[2].set_title("Temporal Profiles (Mode 3)")
    axes[2].set_xlabel("Year")
    axes[2].set_ylabel("Normalized Trajectory")
    axes[2].legend(fontsize=7.5)

    fig.tight_layout()
    save_paper_figure(fig, "fig_E4b_social_analysis")
    plt.close(fig)

    print("\n=== RESUMEN EXPERIMENTO E4(b) CIENCIAS SOCIALES (WDI) ===")
    print(f"CORCONDIA (Mode-0 Center) por r=1..5: {[round(x, 2) for x in cc_scheme_A]}")
    print(f"CORCONDIA (Double Center) por r=1..5: {[round(x, 2) for x in cc_scheme_B]}")
    print(f"Rango óptimo seleccionado: r = {r_opt}")
    print(f"Error relativo de reconstrucción (r={r_opt}): CP = {rel_err_A[r_opt-1]:.4f} | Tucker3 = {tk_err:.4f} | PCA aplanado = {pca_err:.4f}")
    print(f"Energía fuera de rango-1 en PCA aplanado (E4b): {[round(100*x, 1) for x in pca_leaks]}%")
    print("\nTabla de Loadings de Indicadores (CP):")
    df_loadings = pd.DataFrame(B_cp, index=ind_names, columns=[f"CP_{m+1}" for m in range(r_opt)])
    print(df_loadings.round(4).to_string())

    return {
        "r_opt": r_opt,
        "corcondia_A": cc_scheme_A,
        "corcondia_B": cc_scheme_B,
        "cp_rel_error": float(rel_err_A[r_opt - 1]),
        "tucker_rel_error": float(tk_err),
        "pca_rel_error": float(pca_err),
        "pca_non_rank1_pct": [float(100 * x) for x in pca_leaks],
        "indicator_loadings": df_loadings.to_dict(),
    }


if __name__ == "__main__":
    run_experiment_e1()
    run_experiment_e4a()
    run_experiment_e4b()