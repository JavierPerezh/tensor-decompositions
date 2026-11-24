"""
experiments_A.py -- reproducible experiments of Persona A.

    smoke_test()        Preliminary checks of Section 7.3 of the proposal (D, Tucker,
                        CP-ALS on D, HOSVD vs HOOI on a 10x10x10 tensor).
    s_epsilon_check()   Numerical check of the identity ||S_eps - D|| = eps sqrt(3+eps^2)
                        and of the 1/eps growth of the two rank-one terms.
    experiment_E2()     HOSVD versus the optimum (HOOI) on random tensors + an
                        adversarial search for bad cases.
    experiment_E3()     Degeneracy of rank-2 CP-ALS on D and on real rank-3 2x2x2
                        tensors; Monte Carlo estimate of P(rank 2) vs pi/4.
    strassen_check()    Strassen's algorithm as a rank-7 CP decomposition of the
                        2x2 matrix-multiplication tensor (Section 7).

Every function returns (and prints) a summary dictionary. `run_all()` also writes
paper/generated/results_A.tex (LaTeX macros \\resA...) and the auto-generated tables,
so no number in Sections 3 and 6a is transcribed by hand.

Run from the repository root:   python -m src.experiments_A
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from . import tensor_utils as tu
from .plot_style import (COLOR_CP, COLOR_EXTRA, COLOR_PCA, COLOR_SKY,
                         COLOR_TUCKER, GEN_DIR, plt, save_figure)

SEED = tu.SEED
SQRT3 = np.sqrt(3.0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _print(title: str, d: dict) -> None:
    print(f"\n=== {title} ===")
    for k, v in d.items():
        print(f"  {k:38s} {v}")


def _fmt(x, nd=3):
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}".replace(",", "{,}")
    return f"{x:.{nd}f}"


def _sci(x, nd=1):
    if not np.isfinite(x):
        return r"\infty" if x > 0 else r"-\infty"
    m, e = f"{x:.{nd}e}".split("e")
    return f"{m}\\times 10^{{{int(e)}}}"


# ---------------------------------------------------------------------------
# 0. Smoke test (Section 7.3 of the proposal)
# ---------------------------------------------------------------------------
def smoke_test(seed: int = SEED) -> dict:
    D = tu.w_state()
    out = {}
    out["mlrank_D"] = tu.multilinear_rank(D)
    out["unfoldings_equal"] = bool(np.allclose(tu.unfold_kb(D, 0), tu.unfold_kb(D, 1))
                                   and np.allclose(tu.unfold_kb(D, 0), tu.unfold_kb(D, 2)))
    G, F, _ = tu.hosvd(D, (2, 2, 2))
    out["tucker222_fit_error"] = float(np.linalg.norm(D - tu.tucker_to_tensor(G, F)))
    out["disc_D"] = float(tu.slice_pencil_discriminant(D))

    # Rank-2 CP-ALS on D, one random start, plain ALS
    _, err, wmax = tu.cp_als(D, 2, n_iter=5000, seed=seed)
    errs_abs = err * np.linalg.norm(D)
    for it in (10, 100, 1000, 5000):
        out[f"cpD_err_{it}"] = float(errs_abs[it - 1])
        out[f"cpD_wmax_{it}"] = float(wmax[it - 1])

    # HOSVD vs HOOI on a random 10x10x10 tensor, multilinear rank (3,3,3)
    rng = np.random.default_rng(seed)
    T = rng.standard_normal((10, 10, 10))
    G, F, spectra = tu.hosvd(T, (3, 3, 3))
    e_h = np.linalg.norm(T - tu.tucker_to_tensor(G, F))
    e_o = tu.tucker_best_error(T, (3, 3, 3), n_random=5, seed=seed)
    out["prelim_hosvd_err"] = float(e_h)
    out["prelim_hooi_err"] = float(e_o)
    out["prelim_ratio"] = float(e_h / e_o)
    out["prelim_ratio_single_hooi"] = float(e_h / tu.hooi(T, (3, 3, 3))[2])
    _print("Smoke test (proposal Sec. 7.3)", out)
    return out


# ---------------------------------------------------------------------------
# 1. S_epsilon
# ---------------------------------------------------------------------------
def s_epsilon_check() -> dict:
    D = tu.w_state()
    eps = np.logspace(-6, 0, 61)
    err, formula, n1, n2 = [], [], [], []
    for e in eps:
        S, a, b = tu.s_epsilon(e)
        err.append(np.linalg.norm(S - D))
        formula.append(e * np.sqrt(3 + e ** 2))
        n1.append(a)
        n2.append(b)
    err, formula, n1, n2 = map(np.array, (err, formula, n1, n2))
    # rank check: the two terms are rank one by construction; D has rank 3
    mask = eps >= 1e-4  # below this, cancellation in floating point dominates
    out = {
        "max_rel_dev_from_formula": float(np.max(np.abs(err[mask] - formula[mask]) / formula[mask])),
        "eps_x_term_norm_at_1e-6": float(eps[0] * n1[0]),
        "term_norm_formula_ok": bool(np.allclose(n1, (1 + eps ** 2) ** 1.5 / eps)),
    }

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.loglog(eps[mask], err[mask], "o", ms=3, color=COLOR_CP,
              label=r"$\|S_\varepsilon-\mathcal{D}\|_F$ (computed)")
    ax.loglog(eps, formula, "-", color="k", lw=0.8,
              label=r"$\varepsilon\sqrt{3+\varepsilon^2}$")
    ax.loglog(eps, n1, "-", color=COLOR_PCA,
              label=r"norm of each rank-one term of $S_\varepsilon$")
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("Frobenius norm")
    ax.legend(loc="center right", frameon=False)
    save_figure(fig, "S_eps_convergence")
    _print("S_eps check", out)
    return out


# ---------------------------------------------------------------------------
# 2. Experiment E2 -- HOSVD versus the optimum
# ---------------------------------------------------------------------------
E2_CONFIGS = [((10, 10, 10), (3, 3, 3)),
              ((20, 20, 20), (5, 5, 5)),
              ((30, 20, 10), (6, 4, 2)),
              ((15, 15, 15), (2, 5, 8))]
E2_FAMILIES = ["gaussian", "lowrank_noise", "graded"]
E2_LABEL = {"gaussian": "Gaussian i.i.d.",
            "lowrank_noise": "low rank + 30\\% noise",
            "graded": "graded spectrum"}
E2_LABEL_PLOT = {"gaussian": "Gaussian\ni.i.d.",
                 "lowrank_noise": "low rank\n+ noise",
                 "graded": "graded\nspectrum"}


def _e2_tensor(family, shape, ranks, rng):
    if family == "gaussian":
        return rng.standard_normal(shape)
    if family == "lowrank_noise":
        G = rng.standard_normal(ranks)
        F = [np.linalg.qr(rng.standard_normal((s, r)))[0] for s, r in zip(shape, ranks)]
        X = tu.tucker_to_tensor(G, F)
        N = rng.standard_normal(shape)
        return X / np.linalg.norm(X) + 0.3 * N / np.linalg.norm(N)
    if family == "graded":
        # full-size Tucker tensor whose core decays geometrically in every mode,
        # so that each unfolding has a slowly decaying spectrum (no clean gap)
        G = rng.standard_normal(shape)
        for n, s in enumerate(shape):
            w = 0.8 ** np.arange(s)
            G = tu.mode_dot(G, np.diag(w), n)
        F = [np.linalg.qr(rng.standard_normal((s, s)))[0] for s in shape]
        return tu.tucker_to_tensor(G, F)
    raise ValueError(family)


def experiment_E2(n_trials: int = 30, seed: int = SEED, adversarial: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    records = {f: {"lower": [], "upper": []} for f in E2_FAMILIES}
    bound_violations = 0
    hooi_worse_than_hosvd = 0
    for shape, ranks in E2_CONFIGS:
        for fam in E2_FAMILIES:
            lo, up = [], []
            for t in range(n_trials):
                T = _e2_tensor(fam, shape, ranks, rng)
                G, F, spectra = tu.hosvd(T, ranks)
                e_h = np.linalg.norm(T - tu.tucker_to_tensor(G, F))
                tails = tu.hosvd_tail_bounds(spectra, ranks)
                e_o = tu.tucker_best_error(T, ranks, n_random=3,
                                           seed=int(rng.integers(1 << 30)))
                if e_h ** 2 > np.sum(tails ** 2) * (1 + 1e-10):
                    bound_violations += 1
                if e_o > e_h * (1 + 1e-12):
                    hooi_worse_than_hosvd += 1
                lo.append(e_h / e_o)              # <= true ratio  (HOOI error >= optimum)
                up.append(e_h / np.max(tails))    # >= true ratio  (max tail <= optimum)
            lo, up = np.array(lo), np.array(up)
            records[fam]["lower"].extend(lo)
            records[fam]["upper"].extend(up)
            rows.append((shape, ranks, fam, np.median(lo), lo.max(), np.median(up), up.max()))

    all_lo = np.concatenate([records[f]["lower"] for f in E2_FAMILIES])
    all_up = np.concatenate([records[f]["upper"] for f in E2_FAMILIES])
    out = {
        "n_tensors": int(all_lo.size),
        "median_lower_ratio": float(np.median(all_lo)),
        "max_lower_ratio": float(all_lo.max()),
        "max_upper_ratio": float(all_up.max()),
        "bound_violations": bound_violations,
        "hooi_worse_than_hosvd": hooi_worse_than_hosvd,
    }
    for f in E2_FAMILIES:
        out[f"max_lower_{f}"] = float(np.max(records[f]["lower"]))
        out[f"median_lower_{f}"] = float(np.median(records[f]["lower"]))
        out[f"max_upper_{f}"] = float(np.max(records[f]["upper"]))

    if adversarial:
        out.update(_e2_adversarial(seed))

    _e2_figure(records, out)
    _e2_table(rows, n_trials)
    _print("Experiment E2", out)
    return out


def _best_rank1_222(T):
    """Exact (grid + local refinement) best rank-(1,1,1) error of a 2x2x2 tensor.

    sigma_max = max_{|x|=|y|=|z|=1} T(x, y, z) = max_{|x|=1} sigma_1( T(x, ., .) ),
    a one-dimensional problem; error^2 = ||T||^2 - sigma_max^2."""
    from scipy.optimize import minimize_scalar

    def s1(a):
        M = np.cos(a) * T[0] + np.sin(a) * T[1]
        return np.linalg.norm(M, 2)
    g = np.linspace(0, np.pi, 361)
    Ms = np.cos(g)[:, None, None] * T[0] + np.sin(g)[:, None, None] * T[1]
    vals = np.linalg.svd(Ms, compute_uv=False)[:, 0]
    i = int(np.argmax(vals))
    h = g[1] - g[0]
    res = minimize_scalar(lambda a: -s1(a), bounds=(g[i] - h, g[i] + h), method="bounded",
                          options={"xatol": 1e-13})
    smax = max(-res.fun, vals[i])
    return np.sqrt(max(np.sum(T ** 2) - smax ** 2, 0.0))


def _e2_adversarial(seed, n_starts: int = 25):
    """Search 2x2x2 tensors maximising HOSVD error / optimal error for ranks (1,1,1)."""
    rng = np.random.default_rng(seed + 7)

    def ratio(v):
        T = v.reshape(2, 2, 2)
        T = T / np.linalg.norm(T)
        G, F, _ = tu.hosvd(T, (1, 1, 1))
        e_h = np.linalg.norm(T - tu.tucker_to_tensor(G, F))
        e_o = _best_rank1_222(T)
        return e_h / max(e_o, 1e-14)

    best_val, best_T = 0.0, None
    for _ in range(n_starts):
        v0 = rng.standard_normal(8)
        res = minimize(lambda v: -ratio(v), v0, method="Nelder-Mead",
                       options={"maxiter": 1500, "xatol": 1e-9, "fatol": 1e-12})
        if -res.fun > best_val:
            best_val, best_T = -res.fun, res.x.reshape(2, 2, 2) / np.linalg.norm(res.x)
    return {"adversarial_max_ratio_222": float(best_val),
            "adversarial_tensor": np.round(best_T, 4).tolist()}


def _e2_figure(records, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
    colors = [COLOR_TUCKER, COLOR_CP, COLOR_EXTRA]
    ax = axes[0]
    rng = np.random.default_rng(0)
    for k, f in enumerate(E2_FAMILIES):
        lo = np.array(records[f]["lower"])
        up = np.array(records[f]["upper"])
        jit = rng.uniform(-0.12, 0.12, lo.size)
        ax.plot(k - 0.18 + jit, lo, "o", ms=2.2, alpha=0.6, color=colors[k])
        ax.plot(k + 0.18 + jit, up, "^", ms=2.2, alpha=0.6, color=colors[k], mfc="none")
    if "adversarial_max_ratio_222" in out:
        ax.axhline(out["adversarial_max_ratio_222"], color=COLOR_PCA, lw=1, ls=":")
        ax.text(-0.45, out["adversarial_max_ratio_222"] + 0.015,
                "adversarial $2{\\times}2{\\times}2$, ranks $(1,1,1)$",
                ha="left", va="bottom", fontsize=7.5, color=COLOR_PCA)
    ax.axhline(SQRT3, color="k", lw=1, ls="--")
    ax.text(-0.45, SQRT3 + 0.015, r"worst-case bound $\sqrt{3}$", ha="left", va="bottom", fontsize=7.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels([E2_LABEL_PLOT[f] for f in E2_FAMILIES], fontsize=8)
    ax.set_ylim(0.98, 2.15)
    ax.set_ylabel("error ratio")
    ax.plot([], [], "o", color="gray", ms=3, label=r"lower: $e_{\rm HOSVD}/e_{\rm HOOI}$")
    ax.plot([], [], "^", color="gray", ms=3, mfc="none",
            label=r"upper: $e_{\rm HOSVD}/\max_n \tau_n$")
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    ax.set_title("(a) error ratios (true ratio in between)")

    ax = axes[1]
    data = [np.maximum(np.array(records[f]["lower"]) - 1, 1e-16) for f in E2_FAMILIES]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=True,
                    flierprops={"ms": 2})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([E2_LABEL_PLOT[f] for f in E2_FAMILIES], fontsize=8)
    ax.set_ylabel(r"$e_{\rm HOSVD}/e_{\rm HOOI}-1$")
    ax.set_title("(b) relative excess error of the HOSVD")
    fig.tight_layout()
    save_figure(fig, "E2_hosvd_vs_hooi")


def _e2_table(rows, n_trials):
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"size & $(r_1,r_2,r_3)$ & family & median lower & max lower & max upper \\",
             r"\midrule"]
    prev = None
    for shape, ranks, fam, med_lo, max_lo, med_up, max_up in rows:
        sz = r"$%d\times%d\times%d$" % shape if shape != prev else ""
        rk = "$(%d,%d,%d)$" % ranks if shape != prev else ""
        if shape != prev and prev is not None:
            lines.append(r"\addlinespace")
        prev = shape
        lines.append(f"{sz} & {rk} & {E2_LABEL[fam]} & {med_lo:.4f} & {max_lo:.4f} & {max_up:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    (GEN_DIR / "tab_E2.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# 3. Experiment E3 -- degeneracy and the probability of rank 2
# ---------------------------------------------------------------------------
def _random_222_with_sign(rng, sign):
    while True:
        T = rng.standard_normal((2, 2, 2))
        if np.sign(tu.slice_pencil_discriminant(T)) == sign:
            return T / np.linalg.norm(T)


def _distance_to_boundary(T: np.ndarray, n_starts: int = 12, seed: int = 0) -> float:
    """Best rank-2 approximation distance via direct CP parametrization and L-BFGS-B."""
    rng = np.random.default_rng(seed)
    best = float(np.linalg.norm(T))

    for _ in range(n_starts):
        x0 = rng.standard_normal(12)

        def objective(x):
            A = x[:4].reshape(2, 2)
            B = x[4:8].reshape(2, 2)
            C = x[8:12].reshape(2, 2)
            approximation = np.einsum("ir,jr,kr->ijk", A, B, C)
            difference = approximation - T
            return 0.5 * float(np.sum(difference * difference))

        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-12},
        )
        best = min(best, float(np.sqrt(2.0 * max(res.fun, 0.0))))

    return best


def experiment_E3(n_iter: int = 5000, n_runs: int = 5, n_mc: int = 200_000,
                  seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    D = tu.w_state()
    out = {}

    # (a) CP-ALS rank 2 on D
    runs_D = [tu.cp_als(D, 2, n_iter=n_iter, seed=seed + s)[1:] for s in range(n_runs)]
    errD = np.array([r[0] for r in runs_D]) * np.linalg.norm(D)  # absolute error
    wD = np.array([r[1] for r in runs_D])
    out["D_err_final_median"] = float(np.median(errD[:, -1]))
    out["D_err_at_100_median"] = float(np.median(errD[:, 99]))
    out["D_wmax_final_median"] = float(np.median(wD[:, -1]))
    out["D_wmax_at_100_median"] = float(np.median(wD[:, 99]))
    # slope of log w vs log err in the last decade of iterations (theory: w ~ 1/err)
    sl = [np.polyfit(np.log(errD[k, 500:]), np.log(wD[k, 500:]), 1)[0] for k in range(n_runs)]
    out["D_loglog_slope_w_vs_err"] = float(np.median(sl))
    # err * w^2: equals 1/4 + O(eps^2) along the central-difference family
    # T_eps = (2 eps)^{-1}[(e1+eps e2)^{x3} - (e1-eps e2)^{x3}] = D + eps^2 e2^{x3}
    out["D_err_w2_final_median"] = float(np.median(errD[:, -1] * wD[:, -1] ** 2))
    out["D_err_w_final_median"] = float(np.median(errD[:, -1] * wD[:, -1]))

    # (b) real rank-3 tensors (Delta < 0) and control rank-2 tensors (Delta > 0)
    R3 = [_random_222_with_sign(rng, -1) for _ in range(n_runs)]
    R2 = [_random_222_with_sign(rng, +1) for _ in range(n_runs)]
    runs3 = [tu.cp_als(T, 2, n_iter=n_iter, seed=seed + 100 + k)[1:] for k, T in enumerate(R3)]
    runs2 = [tu.cp_als(T, 2, n_iter=n_iter, seed=seed + 200 + k)[1:] for k, T in enumerate(R2)]
    inf3 = np.array([_distance_to_boundary(T, n_starts=12, seed=seed + k) for k, T in enumerate(R3)])
    err3 = np.array([r[0] for r in runs3])  # ||T|| = 1, so relative = absolute
    w3 = np.array([r[1] for r in runs3])
    err2 = np.array([r[0] for r in runs2])
    w2 = np.array([r[1] for r in runs2])
    gap = err3[:, -1] - inf3
    out["R3_final_err_minus_inf_max"] = float(np.max(gap))
    out["R3_final_err_minus_inf_min"] = float(np.min(gap))
    out["R3_inf_min"] = float(inf3.min())
    out["R3_inf_max"] = float(inf3.max())
    out["R3_wmax_final_min"] = float(w3[:, -1].min())
    out["R3_wmax_final_max"] = float(w3[:, -1].max())
    out["R2_final_err_max"] = float(err2[:, -1].max())
    out["R2_wmax_final_max"] = float(w2[:, -1].max())

    _e3_degeneracy_figure(errD, wD, err3, w3, inf3, err2, w2)

    # (c) Monte Carlo: P(rank 2) for Gaussian 2x2x2 real tensors
    t0 = time.time()
    X = np.random.default_rng(seed + 1).standard_normal((n_mc, 2, 2, 2))
    T1, T2 = X[..., 0], X[..., 1]                       # frontal slices
    M = T2 @ np.linalg.inv(T1)                          # T2 T1^{-1}
    ev = np.linalg.eigvals(M)
    real_eig = np.all(np.abs(ev.imag) <= 1e-12 * (1 + np.abs(ev.real)), axis=1)
    disc = tu.slice_pencil_discriminant(X)
    agree = np.mean(real_eig == (disc > 0))
    p_hat = real_eig.mean()
    se = np.sqrt(p_hat * (1 - p_hat) / n_mc)
    out.update({
        "MC_n": n_mc,
        "MC_p_hat": float(p_hat),
        "MC_se": float(se),
        "MC_ci_low": float(p_hat - 1.96 * se),
        "MC_ci_high": float(p_hat + 1.96 * se),
        "MC_pi_over_4": float(np.pi / 4),
        "MC_z": float((p_hat - np.pi / 4) / se),
        "MC_agreement_eig_vs_disc": float(agree),
        "MC_seconds": round(time.time() - t0, 2),
    })
    _e3_mc_figure(real_eig)
    _print("Experiment E3", out)
    return out


def _e3_degeneracy_figure(errD, wD, err3, w3, inf3, err2, w2):
    it = np.arange(1, errD.shape[1] + 1)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    ax = axes[0]
    for k in range(errD.shape[0]):
        ax.loglog(it, errD[k], color=COLOR_CP, alpha=0.8, lw=1)
        ax.loglog(it, wD[k], color=COLOR_PCA, alpha=0.8, lw=1)
    ax.plot([], [], color=COLOR_CP, label=r"error $\|\mathcal{D}-\hat S\|_F$")
    ax.plot([], [], color=COLOR_PCA, label=r"$\max_i \|a_i\|\|b_i\|\|c_i\|$")
    ax.set_xlabel("ALS iteration")
    ax.set_title(r"(a) $\mathcal{D}$: error $\to 0$, norms $\to\infty$")
    ax.legend(frameon=False, fontsize=7, loc="lower left")

    ax = axes[1]
    gaps = err3 - inf3[:, None]
    for k in range(err3.shape[0]):
        ax.semilogx(it, gaps[k], color=COLOR_CP, alpha=0.8, lw=1)
    positive_gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
    if positive_gaps.size:
        ax.set_ylim(max(positive_gaps.min() * 0.5, 1e-12), max(positive_gaps.max() * 2.0, 1e-12))
        ax.set_yscale("log")
    ax.set_xlabel("ALS iteration")
    ax.set_ylabel(r"error $-$ infimum")
    ax.set_title("(b) rank-3 tensors: gap to the inf")

    ax = axes[2]
    for k in range(w3.shape[0]):
        ax.loglog(it, w3[k], color=COLOR_PCA, alpha=0.8, lw=1)
        ax.loglog(it, w2[k], color=COLOR_TUCKER, alpha=0.8, lw=1)
    ax.plot([], [], color=COLOR_PCA, label=r"rank 3 ($\Delta<0$)")
    ax.plot([], [], color=COLOR_TUCKER, label=r"rank 2 ($\Delta>0$)")
    ax.set_xlabel("ALS iteration")
    ax.set_title("(c) largest term norm")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    save_figure(fig, "E3_degeneracy")


def _e3_mc_figure(real_eig):
    n = np.arange(1, real_eig.size + 1)
    run = np.cumsum(real_eig) / n
    p = np.pi / 4
    band = 1.96 * np.sqrt(p * (1 - p) / n)
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    idx = np.unique(np.logspace(1, np.log10(n[-1]), 400).astype(int)) - 1
    ax.fill_between(n[idx], p - band[idx], p + band[idx], color=COLOR_SKY, alpha=0.3,
                    lw=0, label=r"$\pi/4 \pm 1.96\sqrt{p(1-p)/N}$")
    ax.semilogx(n[idx], run[idx], color=COLOR_CP, label="running proportion of rank 2")
    ax.axhline(p, color="k", lw=0.8, ls="--", label=r"$\pi/4$")
    ax.set_ylim(0.70, 0.87)
    ax.set_xlabel(r"number of tensors $N$")
    ax.set_ylabel("proportion")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    save_figure(fig, "E3_monte_carlo")


# ---------------------------------------------------------------------------
# 4. Application check -- Strassen's algorithm as a rank-7 CP decomposition
# ---------------------------------------------------------------------------
def matmul_tensor(n: int = 2) -> np.ndarray:
    """<n,n,n> in K^{n^2} (x) K^{n^2} (x) K^{n^2}: entry 1 at (vec E_ij, vec E_jk, vec E_ik),
    so that C = AB is c = M(vec A, vec B, .) (row-major vec)."""
    M = np.zeros((n * n,) * 3)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                M[i * n + j, j * n + k, i * n + k] = 1
    return M


STRASSEN_U = np.array([[1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
                       [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]], float).T
STRASSEN_V = np.array([[1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
                       [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]], float).T
STRASSEN_W = np.array([[1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
                       [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]], float).T


def strassen_check(seed: int = SEED, n_starts: int = 20, n_iter: int = 3000) -> dict:
    M = matmul_tensor(2)
    S = tu.cp_to_tensor([STRASSEN_U, STRASSEN_V, STRASSEN_W])
    rng = np.random.default_rng(seed)
    A, B = rng.standard_normal((2, 2)), rng.standard_normal((2, 2))
    m = (STRASSEN_U.T @ A.ravel()) * (STRASSEN_V.T @ B.ravel())   # the 7 products
    C = (STRASSEN_W @ m).reshape(2, 2)
    out = {"strassen_tensor_error": float(np.abs(M - S).max()),
           "strassen_product_error": float(np.abs(C - A @ B).max()),
           "matmul_mlrank": tu.multilinear_rank(M)}
    # can plain ALS find exact rank-7 (and rank-6) decompositions?
    for r in (6, 7):
        finals = [tu.cp_als(M, r, n_iter=n_iter, seed=seed + s)[1][-1] for s in range(n_starts)]
        out[f"als_rank{r}_min_relerr"] = float(np.min(finals))
        out[f"als_rank{r}_n_exact"] = int(np.sum(np.array(finals) < 1e-6))
    _print("Strassen check", out)
    return out


# ---------------------------------------------------------------------------
# 5. Verification protocol, step 1: every identity of Persona A's sections
# ---------------------------------------------------------------------------
def verify_identities(seed: int = SEED, n_trials: int = 200) -> dict:
    rng = np.random.default_rng(seed)
    ok = {}
    # Lemma 2.3: r_n <= rank <= min r_i r_j on random tensors of known CP rank r
    good = True
    for _ in range(n_trials):
        I, J, K = rng.integers(2, 7, 3)
        r = int(rng.integers(1, 6))
        T = tu.cp_to_tensor([rng.standard_normal((s, r)) for s in (I, J, K)])
        r1, r2, r3 = tu.multilinear_rank(T)
        good &= max(r1, r2, r3) <= r and r1 <= r2 * r3 and r2 <= r1 * r3 and r3 <= r1 * r2
    ok["lemma_2_3"] = bool(good)
    # Lemma 4.1 (both directions)
    A, B = rng.standard_normal((7, 3)), rng.standard_normal((5, 3))
    Q = rng.standard_normal((3, 3))
    A2, B2 = A @ Q, B @ np.linalg.inv(Q).T
    Qrec = np.linalg.pinv(A) @ A2
    ok["lemma_4_1"] = bool(np.allclose(A @ B.T, A2 @ B2.T) and np.allclose(B2, B @ np.linalg.inv(Qrec).T))
    # Proposition 4.3: orthonormal B' + diagonal decreasing A'^T A' => signs
    X = rng.standard_normal((8, 6))
    U, s, Vt = np.linalg.svd(X)
    S = np.diag(rng.choice([-1.0, 1.0], 3))
    A3, B3 = U[:, :3] * s[:3] @ S, Vt[:3].T @ S
    ok["prop_pca_signs"] = bool(np.allclose(A3 @ B3.T, U[:, :3] * s[:3] @ Vt[:3])
                                and np.allclose(B3.T @ B3, np.eye(3))
                                and np.allclose(A3.T @ A3, np.diag(s[:3] ** 2)))
    # Proposition 4.6: Tucker indeterminacy and its converse
    G = rng.standard_normal((2, 3, 4))
    F = [rng.standard_normal((s, r)) for s, r in zip((5, 6, 7), (2, 3, 4))]
    Qs = [rng.standard_normal((r, r)) for r in (2, 3, 4)]
    G2 = tu.multi_mode_dot(G, [np.linalg.inv(Qn) for Qn in Qs])
    F2 = [Fn @ Qn for Fn, Qn in zip(F, Qs)]
    T = tu.tucker_to_tensor(G, F)
    Qrec = [np.linalg.pinv(Fn) @ F2n for Fn, F2n in zip(F, F2)]
    ok["prop_tucker_indet"] = bool(np.allclose(T, tu.tucker_to_tensor(G2, F2))
                                   and all(np.allclose(a, b) for a, b in zip(Qrec, Qs))
                                   and tu.multilinear_rank(T) == (2, 3, 4))
    # Theorem 4.5: chain of inequalities (HOOI as a proxy for the optimum)
    good = True
    for _ in range(20):
        T = rng.standard_normal((6, 5, 4))
        rk = (2, 2, 2)
        Gh, Fh, sp = tu.hosvd(T, rk)
        eh = np.linalg.norm(T - tu.tucker_to_tensor(Gh, Fh))
        tau = tu.hosvd_tail_bounds(sp, rk)
        eo = tu.tucker_best_error(T, rk, n_random=3)
        good &= tau.max() <= eo + 1e-10 and eo <= eh + 1e-10 and eh <= np.sqrt(np.sum(tau ** 2)) + 1e-10
    ok["thm_hosvd_chain"] = bool(good)
    # Proposition 3.1 / Remark 3.4: S_eps and central difference T_eps
    D = tu.w_state()
    e1, e2 = np.eye(2)
    cube = lambda v: np.einsum("i,j,k->ijk", v, v, v)
    good = True
    for eps in (0.3, 0.05, 1e-3):
        S, _, _ = tu.s_epsilon(eps)
        Tc = (cube(e1 + eps * e2) - cube(e1 - eps * e2)) / (2 * eps)
        good &= np.isclose(np.linalg.norm(S - D), eps * np.sqrt(3 + eps ** 2))
        good &= np.allclose(Tc, D + eps ** 2 * cube(e2))
    N = D[:, :, 1] @ np.linalg.inv(D[:, :, 0])
    ok["prop_3_1_and_rem_central"] = bool(good and np.allclose(N, [[0, 1], [0, 0]]))
    # Cubic forms: 6 x^2 y = (x+y)^3 - (x-y)^3 - 2 y^3  <->  6 D/3 = 2 D
    ok["three_cubes"] = bool(np.allclose(cube(e1 + e2) - cube(e1 - e2) - 2 * cube(e2), 2 * D))
    # Proposition 3.6(1): explicit rank-2 decomposition when eigenvalues are real distinct
    good = True
    for _ in range(n_trials):
        T = rng.standard_normal((2, 2, 2))
        if tu.slice_pencil_discriminant(T) <= 0:
            continue
        lam, P = np.linalg.eig(T[:, :, 1] @ np.linalg.inv(T[:, :, 0]))
        A_, Bt = P.real, np.linalg.inv(P.real) @ T[:, :, 0]
        C_ = np.vstack([np.ones(2), lam.real])
        good &= np.allclose(tu.cp_to_tensor([A_, Bt.T, C_]), T)  # C_ is K x r
    ok["prop_222_decomposition"] = bool(good)
    # Theorem 8.1 construction (r = 4): rank-r approximants converge, slice quotient not diagonalizable
    r = 4
    def embed(core):
        Tb = np.zeros((r, r, 2)); Tb[:2, :2, :] = core
        for i in range(2, r):
            Tb[i, i, :] = 1.0
        return Tb
    Tb = embed(D)
    M = Tb[:, :, 1] @ np.linalg.inv(Tb[:, :, 0])
    jordan = np.linalg.matrix_rank(M @ M) < np.linalg.matrix_rank(M) or \
        np.linalg.matrix_rank(M - np.eye(r)) + np.linalg.matrix_rank(M) > r
    ok["thm_not_closed_construction"] = bool(
        np.isclose(np.linalg.norm(Tb - embed(tu.s_epsilon(1e-3)[0])), 1e-3 * np.sqrt(3 + 1e-6))
        and jordan)
    # Proposition 8.3: Gram identity behind the existence proof
    A_, B_ = rng.standard_normal((5, 3)), rng.standard_normal((4, 3))
    C_ = np.linalg.qr(rng.standard_normal((6, 3)))[0]
    w = np.linalg.norm(A_, axis=0) * np.linalg.norm(B_, axis=0)
    ok["prop_restore_orthonormal_gram"] = bool(
        np.isclose(np.linalg.norm(tu.cp_to_tensor([A_, B_, C_])) ** 2, np.sum(w ** 2)))
    ok.update({f"self_test_{k}": v for k, v in tu._self_test().items()})
    _print("Verification of identities (protocol step 1)", ok)
    return ok


# ---------------------------------------------------------------------------
# LaTeX macros with every number quoted in the text
# ---------------------------------------------------------------------------
def write_results_tex(sm, se, e2, e3, st) -> None:
    m = {
        # smoke test
        "resASmokeTuckerErr": _sci(sm["tucker222_fit_error"] + 1e-300, 0) if sm["tucker222_fit_error"] > 0 else "0",
        "resASmokeErrTen": _fmt(sm["cpD_err_10"]),
        "resASmokeErrFiveK": _fmt(sm["cpD_err_5000"], 4),
        "resASmokeWTen": _fmt(sm["cpD_wmax_10"], 1),
        "resASmokeWFiveK": _fmt(sm["cpD_wmax_5000"], 1),
        "resAPrelimRatio": _fmt(sm["prelim_ratio"], 4),
        "resAPrelimPct": _fmt(100 * (sm["prelim_ratio"] - 1), 1),
        "resAPrelimPctSingle": _fmt(100 * (sm["prelim_ratio_single_hooi"] - 1), 1),
        "resAPrelimHosvdErr": _fmt(sm["prelim_hosvd_err"], 2),
        "resAPrelimHooiErr": _fmt(sm["prelim_hooi_err"], 2),
        "resASmokeErrHundred": _fmt(sm["cpD_err_100"], 4),
        "resASmokeErrThousand": _fmt(sm["cpD_err_1000"], 4),
        "resASmokeWHundred": _fmt(sm["cpD_wmax_100"], 1),
        "resASmokeWThousand": _fmt(sm["cpD_wmax_1000"], 1),
        # S_eps
        "resASepsDev": _sci(se["max_rel_dev_from_formula"], 0),
        # E2
        "resEtwoN": _fmt(e2["n_tensors"]),
        "resEtwoMedianLower": _fmt(e2["median_lower_ratio"], 4),
        "resEtwoMaxLower": _fmt(e2["max_lower_ratio"], 4),
        "resEtwoMaxUpper": _fmt(e2["max_upper_ratio"], 3),
        "resEtwoMaxLowerGauss": _fmt(e2["max_lower_gaussian"], 4),
        "resEtwoMaxLowerLowrank": _fmt(e2["max_lower_lowrank_noise"], 4),
        "resEtwoMaxLowerGraded": _fmt(e2["max_lower_graded"], 4),
        "resEtwoMaxUpperGauss": _fmt(e2["max_upper_gaussian"], 3),
        "resEtwoMaxUpperLowrank": _fmt(e2["max_upper_lowrank_noise"], 3),
        "resEtwoMaxUpperGraded": _fmt(e2["max_upper_graded"], 3),
        "resEtwoViolations": str(e2["bound_violations"]),
        "resEtwoHooiWorse": str(e2["hooi_worse_than_hosvd"]),
        "resEtwoAdv": _fmt(e2.get("adversarial_max_ratio_222", float("nan")), 4),
        # E3
        "resEthreeDErrHundred": _sci(e3["D_err_at_100_median"]),
        "resEthreeDErrFinal": _sci(e3["D_err_final_median"]),
        "resEthreeDWHundred": _fmt(e3["D_wmax_at_100_median"], 1),
        "resEthreeDWFinal": _fmt(e3["D_wmax_final_median"], 1),
        "resEthreeDSlope": _fmt(e3["D_loglog_slope_w_vs_err"], 2),
        "resEthreeDErrWW": _fmt(e3["D_err_w2_final_median"], 3),
        "resEthreeDErrW": _fmt(e3["D_err_w_final_median"], 3),
        "resEthreeInfMin": _fmt(e3["R3_inf_min"], 3),
        "resEthreeInfMax": _fmt(e3["R3_inf_max"], 3),
        "resEthreeGapMax": _sci(e3["R3_final_err_minus_inf_max"]),
        "resEthreeGapMin": _sci(e3["R3_final_err_minus_inf_min"]),
        "resEthreeWThreeMin": _fmt(e3["R3_wmax_final_min"], 1),
        "resEthreeWThreeMax": _fmt(e3["R3_wmax_final_max"], 1),
        "resEthreeRTwoErr": _sci(e3["R2_final_err_max"]),
        "resEthreeRTwoW": _fmt(e3["R2_wmax_final_max"], 2),
        "resEthreeMCN": _fmt(e3["MC_n"]),
        "resEthreeMCp": _fmt(e3["MC_p_hat"], 4),
        "resEthreeMCse": _fmt(e3["MC_se"], 4),
        "resEthreeMClo": _fmt(e3["MC_ci_low"], 4),
        "resEthreeMChi": _fmt(e3["MC_ci_high"], 4),
        "resEthreeMCz": _fmt(e3["MC_z"], 2),
        "resEthreeMCagree": _fmt(100 * e3["MC_agreement_eig_vs_disc"], 3),
        # Strassen
        "resAStrassenProdErr": _sci(st["strassen_product_error"] + 1e-300, 0),
        "resAStrassenSevenExact": str(st["als_rank7_n_exact"]),
        "resAStrassenSixMin": _fmt(st["als_rank6_min_relerr"], 3),
    }
    lines = ["% AUTO-GENERATED by src/experiments_A.py -- do not edit by hand.",
             f"% seed = {SEED}"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    (GEN_DIR / "results_A.tex").write_text("\n".join(lines) + "\n")


def run_all() -> dict:
    t0 = time.time()
    checks = verify_identities()
    if not all(checks.values()):
        raise RuntimeError(f"identity check failed: {checks}")
    sm = smoke_test()
    se = s_epsilon_check()
    e2 = experiment_E2()
    e3 = experiment_E3()
    st = strassen_check()
    write_results_tex(sm, se, e2, e3, st)
    print(f"\nAll Persona-A experiments done in {time.time() - t0:.1f} s")
    return {"smoke": sm, "s_eps": se, "E2": e2, "E3": e3, "strassen": st}


if __name__ == "__main__":
    run_all()