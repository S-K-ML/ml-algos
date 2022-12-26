import numpy as np
import cvxopt
from typing import Tuple

cvxopt.solvers.options['show_progress'] = False

def minimizeL2(
    X: np.ndarray, 
    y: np.ndarray
) -> np.ndarray:
    '''
    Finds weights corresponding to the solution of linear regression objective function with L2 loss.

    Args:
        X: n x d input matrix
        y: n x 1 label vector

    Returns:
        optimal d x 1 vector of weights
    '''
    return np.linalg.inv(X.T @ X) @ X.T @ y


def minimizeL1(
    X: np.ndarray, 
    y: np.ndarray
) -> np.ndarray:
    '''
    Finds weights corresponding to the solution of linear regression objective function with L1 loss.

    Args:
        X: n x d input matrix
        y: n x 1 label vector

    Returns:
        optimal d x 1 vector of weights
    '''
    n = X.shape[0]
    d = X.shape[1]
    
    c = cvxopt.matrix(np.concatenate((np.zeros(d), np.ones(n))))
    G = cvxopt.matrix(
        np.concatenate(
            (
                np.concatenate((np.zeros((n, d)), X, -X), axis=0), 
                np.concatenate((-np.identity(n), -np.identity(n), -np.identity(n)), axis=0)
            ), axis=1
        )
    )
    h = cvxopt.matrix(np.concatenate((np.zeros((n,1)),y,-y)))
    sol = cvxopt.solvers.lp(c,G,h)
    return sol['x'][:d]


def minimizeLinf(
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    '''
    Finds weights corresponding to the solution of linear regression objective function with Linf loss.

    Args:
        X: n x d input matrix
        y: n x 1 label vector

    Returns:
        optimal d x 1 vector of weights
    '''
    n = X.shape[0]
    d = X.shape[1]

    c = cvxopt.matrix(np.concatenate((np.zeros(d), np.ones(1))))
    G = cvxopt.matrix(
        np.concatenate(
            (
                np.concatenate((np.zeros((1, d)), X, -X), axis=0), 
                np.concatenate((-np.ones(1).reshape(-1, 1), -np.ones(n).reshape(-1, 1), -np.ones(n).reshape(-1, 1)), axis=0)
            ),axis=1
        )
    )
    h = cvxopt.matrix(np.concatenate((np.zeros((1,1)),y,-y)))
    sol = cvxopt.solvers.lp(c,G,h)
    return sol['x'][:d]


def linearRegL2Obj(
    w: np.ndarray, 
    X: np.ndarray, 
    y: np.ndarray
) -> Tuple[float, np.ndarray]:
    '''
    Computes the scalar value and gradient of the linear regression objective function with L2 loss.

    Args:
        w: d x 1 parameters vector
        X: n x d input matrix
        y: n x 1 label vector

    Returns:
        scalar value of the objective function and its d x 1 gradient
    '''
    n = len(y)
    resd =  X @ w - y
    obj_func_val = 0.5 * (resd ** 2).mean()
    obj_func_grad = X.T @ resd / n

    return obj_func_val, obj_func_grad


def L2Loss(
    y: np.ndarray,
    y_hat: np.ndarray
) -> float:
    '''
    Computes L2 loss for regression predictions

    Args:
        y: n x 1 ground truth label vector
        y_hat: n x 1 predicted label vector
    
    Returns:
        L2 loss value
    '''
    return np.mean((y - y_hat)**2) / 2


def L1Loss(
    y: np.ndarray,
    y_hat: np.ndarray
) -> float:
    '''
    Computes L1 loss for regression predictions

    Args:
        y: n x 1 ground truth label vector
        y_hat: n x 1 predicted label vector
    
    Returns:
        L2 loss value
    '''
    return np.mean(np.abs(y - y_hat))


def LinfLoss(
    y: np.ndarray,
    y_hat: np.ndarray
) -> float:
    '''
    Computes Linf loss for regression predictions

    Args:
        y: n x 1 ground truth label vector
        y_hat: n x 1 predicted label vector
    
    Returns:
        Linf loss value
    '''
    return np.max(np.abs(y - y_hat))
