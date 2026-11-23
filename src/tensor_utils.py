import numpy as np
import tensorly as tl
from scipy.optimize import linear_sum_assignment


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
    """Producto modal G x_{mode+1} M (0-indexado: mode in {0, 1, 2})."""
    return np.tensordot(M, G, axes=([1], [mode])).swapaxes(0, mode)


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