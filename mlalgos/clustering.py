import numpy as np
import scipy
from typing import Tuple, List

def kmeans(
    X: np.ndarray,
    k: int, 
    max_iter: int=1000
) -> Tuple[np.ndarray, np.ndarray, float]:
    '''
    Iteratively performs K-Means clustering of input X, bounded to max_iter iterations.

    Args:
        X: n x d input matrix X
        k: scalar integer k(1 < k < n), number of clusters
        max_iter: maximum number of iterations

    Returns:
        n x d membership / assignment matrix Y
        k x d matrix of cluster centers U
        scalar objective value achieved by the solutions obj_val
    '''

    n,d = X.shape

    U = np.random.rand(k,d)
    for _ in range(max_iter):
        D = scipy.spatial.distance.cdist(X, U)
        Y = np.where(D == np.min(D, axis=1).reshape(-1,1), 1, 0)
        old_U = U.copy()
        U = np.linalg.inv(Y.T @ Y + 1e-8 * np.eye(k)) @ Y.T @ X
        if np.allclose(old_U, U):
            break

    obj_val = np.linalg.norm(X - Y @ U, 'fro') ** 2 / (2 * n)

    return Y, U, obj_val


def repeatKmeans(
    X: np.ndarray, 
    k: int, 
    n_runs: int=100
) -> Tuple[np.ndarray, np.ndarray, float]:
    '''
    Performs iterative K-means clustering n_runs times and reports best solution.

    Args:
        X: n x d input matrix X
        k: scalar integer k(1 < k < n), number of clusters
        n_runs: number of runs

    Returns:
        best n x d membership / assignment matrix Y
        best k x d matrix of cluster centers U
        scalar objective value achieved by the best solutions obj_val 
    '''
    solutions = []
    for _ in range(n_runs):
        solutions.append(kmeans(X, k))

    return max(solutions, key=lambda solution: solution[2])


def chooseK(
    X: np.ndarray, 
    k_candidates: List[int]
) -> List[float]:
    '''
    Computes objective values by calling repeatKmeans for each of k_candidates.

    Args:
        X: n x d input matrix
        k_candidates: list of candidates

    Returns:
        List of objective values obtained for each k value in the candidate list
    '''
    obj_vals = []
    for k in k_candidates:
        obj_vals.append(repeatKmeans(X, k))
    
    return obj_vals
