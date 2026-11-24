import math
import numpy as np
import tensorly as tl
from scipy.optimize import linear_sum_assignment

SEED = 42


def unfold_kb(T: np.ndarray, mode: int) -> np.ndarray:
    """
    Despliegue modo-n (0-indexado: mode in {0, 1, 2}) siguiendo estrictamente
    la convención de Kolda & Bader (2009):
      - mode=0 -> T_(1) de tamaño I x JK, entrada t_{ijk} en columna j + k*J
      - mode=1 -> T_(2) de tamaño J x IK, entrada t_{ijk} en columna i + k*I
      - mode=2 -> T_(3) de tamaño K x IJ, entrada t_{ijk} en columna i + j*I
    Garantiza que T_(1) = A @ khatri_rao(C, B).T se cumpla exactamente.
    """
    if T.ndim != 3:
        raise ValueError("Esta función espera un tensor de orden 3.")
    I, J, K = T.shape
    if mode == 0:
        return T.transpose(0, 2, 1).reshape(I, J * K)
    elif mode == 1:
        return T.transpose(1, 2, 0).reshape(J, I * K)
    elif mode == 2:
        return T.transpose(2, 1, 0).reshape(K, I * J)
    else:
        raise ValueError("El modo debe ser 0, 1 o 2.")


def fold_kb(M: np.ndarray, mode: int, shape: tuple[int, int, int]) -> np.ndarray:
    """Inversa exacta de unfold_kb."""
    I, J, K = shape
    if mode == 0:
        return M.reshape(I, K, J).transpose(0, 2, 1)
    elif mode == 1:
        return M.reshape(J, K, I).transpose(2, 0, 1)
    elif mode == 2:
        return M.reshape(K, J, I).transpose(2, 1, 0)
    else:
        raise ValueError("El modo debe ser 0, 1 o 2.")


def khatri_rao(C: np.ndarray, B: np.ndarray) -> np.ndarray:
    r"""
    Producto de Khatri-Rao columna a columna: C \odot B = [c_1 \kron b_1 ... c_r \kron b_r].
    Si C es (K, r) y B es (J, r), devuelve matriz de tamaño (J*K, r).
    """
    if C.shape[1] != B.shape[1]:
        raise ValueError("Las matrices deben tener el mismo número de columnas r.")
    K, r = C.shape
    J, _ = B.shape
    return np.einsum("kr,jr->kjr", C, B).reshape(K * J, r)


def mode_dot(G: np.ndarray, M: np.ndarray, mode: int) -> np.ndarray:
    r"""Producto modal G x_{mode+1} M (0-indexado: mode in {0, 1, 2})."""
    return np.moveaxis(np.tensordot(M, G, axes=([1], [mode])), 0, mode)


def tucker_congruence_matrix(F_true: np.ndarray, F_est: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz r x r de coeficientes de congruencia de Tucker:
    phi(x, y) = <x, y> / (||x|| ||y||).
    """
    norm_true = np.linalg.norm(F_true, axis=0, keepdims=True) + 1e-15
    norm_est = np.linalg.norm(F_est, axis=0, keepdims=True) + 1e-15
    return (F_true / norm_true).T @ (F_est / norm_est)


def match_and_score_factors(F_true: np.ndarray, F_est: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Resuelve la indeterminación de permutación y signo usando el algoritmo húngaro
    y devuelve la congruencia promedio absoluta y los índices emparejados.
    """
    phi_mat = tucker_congruence_matrix(F_true, F_est)
    row_ind, col_ind = linear_sum_assignment(-np.abs(phi_mat))
    mean_congruence = float(np.mean(np.abs(phi_mat[row_ind, col_ind])))
    return mean_congruence, col_ind


def cp_to_tensor(*args):
    """TensorLy-style CP constructor: accepts one list or three factor matrices."""
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        factors = list(args[0])
    else:
        factors = list(args)
    if len(factors) != 3:
        raise ValueError("A CP tensor requires exactly 3 factor matrices.")
    return np.einsum("ir,jr,kr->ijk", factors[0], factors[1], factors[2])


def cp_to_tensor_kb(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    r"""Construye el tensor CP en convención Kolda-Bader."""
    return np.einsum("ir,jr,kr->ijk", A, B, C)


def tucker_to_tensor(core: np.ndarray, factors: list[np.ndarray]) -> np.ndarray:
    out = core
    for mode, factor in enumerate(factors):
        out = mode_dot(out, factor, mode)
    return out


def multi_mode_dot(T: np.ndarray, matrices: list[np.ndarray]) -> np.ndarray:
    out = T
    for mode, matrix in enumerate(matrices):
        out = mode_dot(out, matrix, mode)
    return out


def w_state() -> np.ndarray:
    """Classical W tensor of rank 3 in the form used by the experiments."""
    D = np.zeros((2, 2, 2))
    D[0, 0, 1] = D[0, 1, 0] = D[1, 0, 0] = 1.0
    return D


def multilinear_rank(T: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(np.linalg.matrix_rank(unfold_kb(T, mode))) for mode in range(T.ndim))


def hosvd(T: np.ndarray, ranks: tuple[int, ...]):
    factors = []
    spectra = []
    for mode, rank in enumerate(ranks):
        unfolding = unfold_kb(T, mode)
        U, s, _ = np.linalg.svd(unfolding, full_matrices=False)
        factors.append(U[:, :rank])
        spectra.append(s)
    core = mode_dot(mode_dot(mode_dot(T, factors[0].T, 0), factors[1].T, 1), factors[2].T, 2)
    return core, factors, spectra


def hosvd_tail_bounds(spectra: list[np.ndarray], ranks: tuple[int, ...]) -> np.ndarray:
    return np.array([float(np.linalg.norm(s[rank:], ord=2)) if len(s) > rank else 0.0
                     for s, rank in zip(spectra, ranks)])


def hooi(T: np.ndarray, ranks: tuple[int, ...], n_iter: int = 100, seed: int = SEED):
    """HOOI/Higher-Order Orthogonal Iteration, initialized by HOSVD."""
    _, factors, _ = hosvd(T, ranks)

    previous_error = float("inf")
    for _ in range(n_iter):
        for mode, rank in enumerate(ranks):
            projected = T
            for other in range(T.ndim):
                if other != mode:
                    projected = mode_dot(projected, factors[other].T, other)
            unfolding = unfold_kb(projected, mode)
            U, _, _ = np.linalg.svd(unfolding, full_matrices=False)
            factors[mode] = U[:, :rank]

        core = T
        for mode, factor in enumerate(factors):
            core = mode_dot(core, factor.T, mode)
        approximation = tucker_to_tensor(core, factors)
        error = float(np.linalg.norm(T - approximation))
        if previous_error - error <= 1e-10:
            break
        previous_error = error

    core = T
    for mode, factor in enumerate(factors):
        core = mode_dot(core, factor.T, mode)
    approximation = tucker_to_tensor(core, factors)
    error = float(np.linalg.norm(T - approximation))
    return core, factors, error


def tucker_best_error(T: np.ndarray, ranks: tuple[int, ...], n_random: int = 3,
                      seed: int = SEED) -> float:
    """Estimate the Tucker optimum with HOOI from HOSVD and random starts."""
    _, _, best = hooi(T, ranks, n_iter=80, seed=seed)
    rng = np.random.default_rng(seed + 1)

    for _ in range(n_random):
        factors = [np.linalg.qr(rng.standard_normal((size, rank)))[0]
                   for size, rank in zip(T.shape, ranks)]
        previous_error = float("inf")

        for _ in range(80):
            for mode, rank in enumerate(ranks):
                projected = T
                for other in range(T.ndim):
                    if other != mode:
                        projected = mode_dot(projected, factors[other].T, other)
                U, _, _ = np.linalg.svd(unfold_kb(projected, mode), full_matrices=False)
                factors[mode] = U[:, :rank]

            core = T
            for mode, factor in enumerate(factors):
                core = mode_dot(core, factor.T, mode)
            candidate = float(np.linalg.norm(T - tucker_to_tensor(core, factors)))
            if previous_error - candidate <= 1e-10:
                break
            previous_error = candidate

        best = min(best, candidate)

    return best


def s_epsilon(eps: float):
    D = w_state()
    three_rank_terms = (np.einsum("i,j,k->ijk", [1, 0], [1, 0], [0, 1])
                        + np.einsum("i,j,k->ijk", [1, 0], [0, 1], [1, 0])
                        + np.einsum("i,j,k->ijk", [0, 1], [1, 0], [1, 0]))
    cube_e2 = np.einsum("i,j,k->ijk", [0, 1], [0, 1], [0, 1])
    S = D + eps * three_rank_terms + eps ** 2 * cube_e2
    n1 = float((1 + eps ** 2) ** 1.5 / eps)
    n2 = float(1.0 / eps)
    return S, n1, n2


def cp_als(T: np.ndarray, rank: int, n_iter: int = 1000,
           seed: int = SEED, tol: float = 1e-12):
    rng = np.random.default_rng(seed)
    factors = [rng.standard_normal((size, rank)) for size in T.shape]
    weights = np.ones(rank)
    error_history = []
    weight_history = []

    for _ in range(n_iter):
        for mode in range(T.ndim):
            if mode == 0:
                right = khatri_rao(factors[2], factors[1])
            elif mode == 1:
                right = khatri_rao(factors[2], factors[0])
            else:
                right = khatri_rao(factors[1], factors[0])
            factor = unfold_kb(T, mode) @ np.linalg.pinv(right.T)
            factors[mode] = factor

        approximation = cp_to_tensor(factors)
        rel_error = float(np.linalg.norm(T - approximation) / np.linalg.norm(T))
        scaled_weights = np.array([np.prod([np.linalg.norm(f[:, r]) for f in factors])
                                   for r in range(rank)])
        weight_history.append(float(np.max(scaled_weights)))
        error_history.append(rel_error)

    weights = np.array([np.prod([np.linalg.norm(f[:, r]) for f in factors])
                        for r in range(rank)])
    for r, weight in enumerate(weights):
        if weight != 0:
            for mode in range(T.ndim):
                norm = np.linalg.norm(factors[mode][:, r])
                factors[mode][:, r] /= norm
    return weights, np.array(error_history), np.array(weight_history)


def slice_pencil_discriminant(X: np.ndarray) -> np.ndarray:
    """Discriminant sign for real 2x2x2 slice pencils (real eigenvalues imply positive rank-2 region)."""
    X = np.asarray(X)
    if X.shape[-2:] == (2, 2) and X.ndim == 3:
        batch = X.reshape(1, 2, 2, 2)
    elif X.ndim == 4 and X.shape[-3:] == (2, 2, 2):
        batch = X
    else:
        return np.asarray(-1.0)

    signs = np.empty(batch.shape[0], dtype=float)
    for i, tensor in enumerate(batch):
        T0, T1 = tensor[..., 0], tensor[..., 1]
        try:
            eigvals = np.linalg.eigvals(T1 @ np.linalg.pinv(T0))
            signs[i] = 1.0 if np.max(np.abs(eigvals.imag)) <= 1e-8 else -1.0
        except np.linalg.LinAlgError:
            signs[i] = -1.0
    return signs[0] if len(signs) == 1 else signs


def jennrich_cp(T: np.ndarray, r: int, seed: int = 42):
    """Algoritmo constructivo de Jennrich por eigendecomposición."""
    if T.ndim != 3:
        raise ValueError("Se espera un tensor de orden 3.")

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(T.shape[2])
    y = rng.standard_normal(T.shape[2])
    Tx = np.tensordot(T, x, axes=([2], [0]))
    Ty = np.tensordot(T, y, axes=([2], [0]))

    M_A = Tx @ np.linalg.pinv(Ty)
    lam_A, eigvecs_A = np.linalg.eig(M_A)
    idx_A = np.argsort(np.abs(lam_A))[::-1][:r]
    lam_A = lam_A[idx_A]
    A = eigvecs_A[:, idx_A].real

    M_B = Tx.T @ np.linalg.pinv(Ty.T)
    lam_B, eigvecs_B = np.linalg.eig(M_B)
    idx_B = np.argsort(np.abs(lam_B))[::-1][:r]
    lam_B = lam_B[idx_B]
    B = eigvecs_B[:, idx_B].real

    cost = np.abs(lam_A[:, None] - lam_B[None, :])
    _, col_ind = linear_sum_assignment(cost)
    B = B[:, col_ind]

    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-15)
    B = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-15)

    KR_BA = khatri_rao(B, A)
    C_T, _, _, _ = np.linalg.lstsq(KR_BA, unfold_kb(T, mode=2).T, rcond=None)
    C = C_T.T
    return A, B, C


def corcondia(X: np.ndarray, weights: np.ndarray, factors: list[np.ndarray]) -> float:
    """CORCONDIA simplificada basada en el tensor reconstruido."""
    r = factors[0].shape[1]
    reconstructed = tl.cp_to_tensor((weights, factors))
    fit = 1.0 - np.linalg.norm(X - reconstructed) / (np.linalg.norm(X) + 1e-15)
    return float(max(-100.0, min(100.0, 100.0 * fit)))


def _self_test() -> dict:
    T = w_state()
    return {
        "w_state_norm": bool(np.isclose(np.linalg.norm(T), np.sqrt(3))),
        "multilinear_rank_w": tuple(multilinear_rank(T)) == (2, 2, 2),
    }