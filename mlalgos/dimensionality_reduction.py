import numpy as np
import scipy

def PCA(
    X: np.ndarray, 
    k: int,
) -> np.ndarray:
    '''
    Performs PCA with specified number of projecting directions.

    Args:
        X: n x d input matrix
        k: number of projecting directions (1 <= k < d)

    Returns:
        k x d matrix of projecteing directions U whose rows correspond to 
        the top-k projecting directions with the largest variances
    '''
    X_cent = X - np.mean(X, axis=0)
    _, eigvec = scipy.linalg.eigh(X_cent.T @ X_cent)
    return eigvec[:, -k:].T


def projPCA(
    X_test: np.ndarray,
    mu: np.ndarray,
    U: np.ndarray
) -> np.ndarray:
    '''
    Projects features of X_test onto the directions U.

    Args:
        X_test: m x d input matrix
        mu: d x 1 training mean vector
        U: k x d projection matrix U

    Returns:
        m x k projected matrix Xproj
    '''
    return (X_test - mu.T) @ U.T