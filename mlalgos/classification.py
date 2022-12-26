import numpy as np
import scipy
import cvxopt
from typing import Tuple, Callable

def logisticRegObj(
        w: np.ndarray, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[float, np.ndarray]:
    '''
    Computes the scalar value and gradient of the logistic regression objective function.

    Args:
        w: d x 1 parameters vector
        X: n x d input matrix
        y: n x 1 label vector

    Returns:
        scalar value of the objective function and its d x 1 gradient
    '''
    n = len(y)
    z = X @ w
    y_hat = 1 / (1 + np.exp(-z))
    log_y_hat = np.logaddexp(0, -z)
    obj_func_value = (
        np.dot(y.reshape(-1), log_y_hat)[0] - 
        np.dot((1 - y).reshape(-1), -z - log_y_hat)[0] / n
    )
    obj_func_grad = X.T @ (y_hat - y) / n

    return obj_func_value, obj_func_grad


def accuracy(
        y: np.ndarray, 
        y_hat: np.ndarray
    ) -> float:
    '''
    Computes accuracy score for the binary classification predictions.

    Args:
        y: n x 1 ground truth label vector
        y_hat: n x 1 predicted label vector
    
    Returns:
        accuracy score in range [0, 1]
    '''
    return (y == y_hat).sum() / len(y)


def minBinDev(
        X: np.ndarray, 
        y: np.ndarray, 
        lamb: float
    ) -> Tuple[np.ndarray, float]:
    '''
    Finds weights corresponding to the solution of the regularized binomial deviance loss objective function.

    Args:
        X: n x d input matrix
        y: n x 1 label vector
        lamb: regularization hyperparameter, > 0

    Returns:
        Optimal d x 1 vector of weights and a scalar intercept  
    '''
    d = X.shape[1]

    obj_func = lambda u: (
        np.sum(np.logaddexp(0, -y * (X @ u[:-1][:, None] + u[-1]))) +
        0.5 * lamb * np.sum(u[:-1] ** 2)
    )

    u_init = np.ones(d + 1)
    u = scipy.optimize.minimize(obj_func, u_init)['x']
    w = u[:-1][:, None]
    w0 = u[-1]    

    return w, w0


def minHinge(
        X: np.ndarray, 
        y: np.ndarray, 
        lamb: float
    ) -> Tuple[np.ndarray, float]:
    '''
    Finds weights corresponding to the solution of the regularized hinge loss objective function.

    Args:
        X: n x d input matrix
        y: n x 1 label vector
        lamb: regularization hyperparameter, > 0

    Returns:
        Optimal d x 1 vector of weights and a scalar intercept  
    '''
    n = X.shape[0]
    d = X.shape[1]

    P = cvxopt.matrix(
        np.lib.pad(lamb * np.eye(d), ((0, n + 1), (0, n + 1))) +
        (1e-8) * np.eye(n + d + 1)
    )
    q = cvxopt.matrix(np.concatenate([np.zeros((d + 1, 1)), np.ones((n, 1))]))
    G = cvxopt.matrix(np.concatenate([
        np.lib.pad(-np.eye(n), ((0, 0), (d + 1, 0))),
        np.concatenate([-np.diag(y.T[0]) @ X, -y, -np.eye(n)], axis=1)
    ]))
    h = cvxopt.matrix(np.concatenate([np.zeros((n, 1)), -np.ones((n, 1))]))

    u = cvxopt.solvers.qp(P, q, G, h)['x']
    w = np.array(u[:d])
    w0 = u[d]

    return w, w0


def classify(
        X_test: np.ndarray, 
        w: np.ndarray, 
        w0: np.ndarray
    ) -> np.ndarray:
    '''
    Performs predictions on the test set given the trained primal form model.

    Args:
        X_test: m x d input matrix
        w: d x 1 weights vector
        w0: scalar intercept
    
    Returns:
        m x 1 prediction vector
    '''
    return np.sign(X_test @ w + w0)


def adjBinDev(
        X: np.ndarray, 
        y: np.ndarray, 
        lamb: float,
        kernel_func: Callable
    ) -> Tuple[np.ndarray, float]:
    '''
    Finds weights corresponding to the solution of the regularized adjusted binomial deviance objective function.

    Args:
        X: n x d input matrix
        y: n x 1 label vector
        lamb: regularization hyperparameter, > 0
        kernel_func: kernel function

    Returns:
        Optimal d x 1 vector of weights and a scalar intercept  
    '''
    K = kernel_func(X,X)
    d = X.shape[1]
    n = X.shape[0]
    def obj_func(u):
        a0 = u[-1]
        a = u[:-1]
        a = a[:,None]
        return np.sum(np.logaddexp(0, -y * (K @ a + a0))) + 0.5*  lamb * float(a.T @ K @ a)
    
    
    u_init = np.ones(n + 1)
    u = scipy.optimize.minimize(obj_func, u_init)['x']
    a = u[:-1][:, None]
    a0 = u[-1]    
    return a, a0
    

def adjHinge(
        X: np.ndarray, 
        y: np.ndarray, 
        lamb: float,
        kernel_func: Callable
) -> Tuple[np.ndarray, float]:
    '''
    Finds weights corresponding to the solution of the regularized adjusted hinge loss objective function.

    Args:
        X: n x d input matrix
        y: n x 1 label vector
        lamb: regularization hyperparameter, > 0
        kernel_func: kernel function

    Returns:
        Optimal d x 1 vector of weights and a scalar intercept  
    '''
    K = kernel_func(X,X)
    n = X.shape[0]
    d = X.shape[1]

    P_row1 = np.concatenate([lamb * K, np.zeros((n,n+1))], axis = 1)
    P = np.concatenate([P_row1, np.zeros((n+1, 2*n+1))], axis = 0)
    P = P + (1e-8) * np.eye(2*n+1)
    P = cvxopt.matrix(P)
    q = cvxopt.matrix(np.concatenate([np.zeros((n + 1, 1)), np.ones((n, 1))]))
    
    G = cvxopt.matrix(np.concatenate([
        np.lib.pad(-np.eye(n), ((0, 0), (n + 1, 0))),
        np.concatenate([-np.diag(y.T[0]) @ K, -y, -np.eye(n)], axis=1)
    ]))
    
    h = cvxopt.matrix(np.concatenate([np.zeros((n, 1)), -np.ones((n, 1))], axis=0))
    u = cvxopt.solvers.qp(P, q, G, h)['x']
    a = np.array(u[:n])
    a0 = u[n]

    return a, a0
    
    
def adjClassify(
        Xtest, a, a0, X, kernel_func
) -> np.ndarray:
    '''

    Args:
        X_test: m x d input matrix
        a: d x 1 vector of weights
        a0: scalar intercept
        X: n x d input matrix
        kernel_func: kernel function
        
    
    Returns:
        m x 1 prediction vector
    '''
    return np.sign(kernel_func(Xtest,X) @ a + a0)   


def dualHinge(
    X: np.ndarray, 
    y: np.ndarray, 
    lamb: float, 
    kernel_func: Callable
) -> Tuple[np.ndarray, float]:
    '''
    Finds weights corresponding to the solution of the the dual form SVM objective function.

    Args:
        X: n x d input matrix
        y: n x 1 label vector
        lamb: regularization hyperparameter, > 0
        kernel_func: kernel function
    
    Returns:
        Optimal d x 1 vector of weights and a scalar intercept  
    '''
    
    n = X.shape[0]

    delta_y = np.diag(y.T[0])
    K = kernel_func(X, X)

    P = cvxopt.matrix(delta_y @ K @ delta_y / lamb + (1e-8) * np.eye(n))
    q = cvxopt.matrix(-np.ones((n, 1)))
    G = cvxopt.matrix(np.concatenate([-np.eye(n), np.eye(n)]))
    h = cvxopt.matrix(np.concatenate([np.zeros((n, 1)), np.ones((n, 1))]))
    A = cvxopt.matrix(y.T.astype('d'))
    b = cvxopt.matrix(np.zeros((1, 1)))

    a = cvxopt.solvers.qp(P, q, G, h, A, b)['x']

    i = np.argmin(np.abs(0.5 - a))
    b = y[i] - K[i] @ delta_y @ a / lamb

    return a, b


def dualClassify(
    Xtest: np.ndarray, 
    a: np.ndarray, 
    b: float, 
    X: np.ndarray, 
    y: np.ndarray, 
    lamb: float, 
    kernel_func: Callable
) -> float:
    '''
    Performs predictions on the test set given the trained SVM dual form model.

    Args:
        Xtest: m x d test matrix
        a: n x 1 weights vector
        b: scalar intercept
        X: n x d training matrix
        y: n x 1 training label vector
        lamb: regularization hyperparameter, > 0
        kernel_func: kernel function

    Returns:
        m x 1 prediction vector
    '''
    return np.sign(kernel_func(Xtest, X) @ np.diag(y.T[0]) @ a / lamb + b)


def minMulDev(
    X: np.ndarray,
    Y: np.ndarray
) -> np.ndarray:
    '''
    Computes the optimal weights matrix W corresponding to the solution of the multinomial deviance loss.

    Args:
        X: n x d input matrix X
        Y: k x d label matrix Y

    Returns: 
        d x k weights matrix W orresponding to the solution of the multinomial deviance loss.
    '''
    d = X.shape[1]
    k = Y.shape[1]

    u = np.ones(d*k)

    def obj_func(a):
        W = a.reshape(d,k)
        return np.mean((
            scipy.special.logsumexp(X @ W, axis=1).reshape(-1,1) - 
            np.sum((X @ W)*Y, axis=1).reshape(-1,1))
        )

    W = scipy.optimize.minimize(obj_func, u)['x'].reshape(d,k)
    return W


def mulClassify(
    Xtest: np.ndarray, 
    W: np.ndarray
) -> np.ndarray:
    '''
    Classifies data points according to the provided weights.

    Args:
        Xtest: m x d input matrix
        d x k weights matrix W
    
    Returns:
        m x k prediction matrix Yhat
    '''
    XtestW = Xtest @ W
    return np.where(XtestW == np.max(XtestW, axis=1).reshape(-1, 1), 1, 0)


def mulAccuracyScore(
    Yhat: np.ndarray, 
    Y: np.ndarray
) -> float:
    '''
    Computes accuracy score of the predictions.

    Args:
        Yhat: m x k prediction matrix Yhat
        m x k label matrix Y
    
    Returns:
        accuracy of the predictions, scalar
    '''
    return np.sum(Yhat*Y) / len(Yhat)
