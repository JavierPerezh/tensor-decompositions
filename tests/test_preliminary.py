import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac, tucker
from src.tensor_utils import unfold_kb, fold_kb, khatri_rao, mode_dot


def test_kolda_bader_unfolding_formula_1():
    r"""Verifica Fórmula (1): T_(1) = A (C \odot B)^T y el plegado inverso."""
    rng = np.random.default_rng(42)
    I, J, K, r = 5, 4, 3, 2
    A = rng.standard_normal((I, r))
    B = rng.standard_normal((J, r))
    C = rng.standard_normal((K, r))

    T = np.einsum("ir,jr,kr->ijk", A, B, C)
    T1_kb = unfold_kb(T, mode=0)
    T1_formula = A @ khatri_rao(C, B).T

    assert np.allclose(T1_kb, T1_formula, atol=1e-12)
    assert np.allclose(fold_kb(T1_kb, mode=0, shape=(I, J, K)), T, atol=1e-12)


def test_worked_example_D_checks():
    """Reproduce los chequeos preliminares de la Sec. 7.3 sobre el estado W (D)."""
    D = np.zeros((2, 2, 2))
    D[0, 0, 1] = D[0, 1, 0] = D[1, 0, 0] = 1.0

    # 1. Rango multilineal (2, 2, 2)
    ranks = [np.linalg.matrix_rank(unfold_kb(D, n)) for n in range(3)]
    assert ranks == [2, 2, 2]

    # 2. Ajuste Tucker (2, 2, 2) exacto
    core, factors = tucker(D, rank=[2, 2, 2])
    D_tucker = mode_dot(mode_dot(mode_dot(core, factors[0], 0), factors[1], 1), factors[2], 2)
    assert np.linalg.norm(D - D_tucker) < 1e-12

    # 3. Degeneración en CP-ALS de rango 2 (el error baja y los pesos crecen)
    cp_10 = parafac(D, rank=2, n_iter_max=10, tol=0, init="random", random_state=1, normalize_factors=True)
    cp_1000 = parafac(D, rank=2, n_iter_max=1000, tol=0, init="random", random_state=1, normalize_factors=True)
    err_10 = np.linalg.norm(D - tl.cp_to_tensor(cp_10))
    err_1000 = np.linalg.norm(D - tl.cp_to_tensor(cp_1000))
    assert err_1000 < err_10
    assert np.max(cp_1000.weights) > np.max(cp_10.weights)