import numpy as np
import matplotlib.pyplot as plt
from mlalgos.utils import *

def generateMulClassData(
    n: int,
    gen_model: int
):
    d = 2
    shift = 1.8

    X = []
    y = []
    m = n // 4
    class_label = 0
    for i in [-1, 1]:
        for j in [-1, 1]:
            if gen_model == 1:
                X.append(np.random.randn(m, d) + 
                         class_label * shift)
            elif gen_model == 2:
                X.append(np.random.randn(m, d) + 
                         shift * np.array([[i, j]]))
            else:
                raise ValueError("Unknown generative model")
            y.append(np.ones((m, 1)) * class_label)
            class_label += 1
    X = np.vstack(X)
    y = np.vstack(y)

    return augmentX(X), convertToOneHot(y, 4)


def plotMulClassPoints(
    X: np.ndarray, 
    Y: np.ndarray
):
    k = Y.shape[1]
    markers = ['o', '+', 'd', 'x', '^', 'v', 's']
    colors = ['r', 'b', 'g', 'y', 'm', 'c', 'k']
    X = X[:, 1:]
    labels = Y.argmax(axis=1)
    for i in range(k):
        Xpart = X[labels == i]

        plt.scatter(
            Xpart[:, 0], Xpart[:, 1], 
            marker=markers[i], 
             color=colors[i],
            label=f'class {i}'
        )
