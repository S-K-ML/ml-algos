import numpy as np
from scipy.spatial.distance import cdist
from typing import Callable

def gd(
        obj_func: Callable,
        w_init: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        eta: float,
        max_iter: int,
        tol: float
    ) -> np.ndarray:
    '''
    Performs gradient descent on the objective function with respect to parameters. 

    Args:
        obj_func: objective function 
        w_init: d x 1 initial parameter vector
        X: n x d input matrix
        y: n x 1 label vector
        eta: step size, positive float
        max_iter: maximum number of iterations, positive integer
        tol: tolerance, positive float
    
    Returns:
        optimal d x 1 parameter vector
    '''
    w = w_init.copy()

    for _ in range(max_iter):
        _, obj_func_grad = obj_func(w, X, y)
        if np.sqrt((obj_func_grad ** 2).sum()) < tol: break
        w -= eta * obj_func_grad

    return w 


def linearKernel(
    X1: np.ndarray, 
    X2: np.ndarray
) -> np.ndarray:
    '''
    Computes a linear kernel.
    '''
    return X1 @ X2.T


def polyKernel(
    X1: np.ndarray, 
    X2: np.ndarray, 
    degree: int
) -> np.ndarray:
    '''
    Computes a polynomial kernel.
    '''
    return (X1 @ X2.T + 1) ** degree


def gaussKernel(
    X1: np.ndarray, 
    X2: np.ndarray, 
    width: float
) -> np.ndarray:
    '''
    Computes a Gaussian kernel.
    '''
    distances = cdist(X1, X2, 'sqeuclidean')
    return np.exp(- distances / (2*(width**2)))