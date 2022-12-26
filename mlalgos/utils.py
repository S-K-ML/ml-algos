import numpy as np

def augmentX(
    X: np.ndarray
) -> np.ndarray:
    '''
    Appends a column of ones to X.

    Args:
        X: n x (d-1) input matrix

    Retuns:
        augmented n x d matrix
    '''
    n = X.shape[0]
    return np.concatenate((np.ones((n, 1)), X),  axis=1)


def unAugmentX(
    X: np.ndarray
) -> np.ndarray:
    '''
    Removes a columns of ones from X.

    Args:
        X: n x d input matrix

    Retuns:
        unaugmented n x (d-1) matrix
    '''

    return X[:, 1:]


def convertToOneHot(
    y: np.ndarray, 
    n_values: int
) -> np.ndarray:
    '''
    Performs One-Hot encoding on a label vector.

    Args:
        y: n x 1 label vector
        n_values: number of classes (c)

    Returns:
        n x c encoded matrix 
    '''
    y = y.astype(int).flatten()
    Y = np.eye(n_values)[y]
    return Y.astype(float)