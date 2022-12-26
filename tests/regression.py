import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
from mlalgos.regression import *
from mlalgos.common import gd
from mlalgos.utils import (
    augmentX,
    unAugmentX
)
from tests.config import *

def generateRegData(
    n: int,
    d: int=5,
    noise: float=0.2
):
    X = np.random.randn(n, d)  #input matrix
    X = augmentX(X)  # augment input matrix with ones
    w_true = np.random.randn(d + 1, 1)  # true model parameters
    y = X @ w_true + noise * np.random.randn(n, 1)  #ground truth label
    return X, y


def runSynthLinRegExperiments() -> Tuple[np.ndarray, np.ndarray]:
    '''
    Evaluates linear regression with L1, L2, Linf loss on the synthetic data.

    Returns:
        3x3 matrix of training and test losses for each of the 9 combinations of the regularization parameters
    '''

    n_exp = 100
    n_trn = 30
    n_tst = 1000

    trn_losses = np.zeros((3, 3))
    tst_losses = np.zeros((3, 3))

    for exp in range(1, n_exp + 1):

        X, y = generateRegData(n=n_trn+n_tst, d=5)

        X_trn = X[:n_trn]
        y_trn = y[:n_trn]

        X_tst = X[n_trn:]
        y_tst = y[n_trn:]

        w_l2 = minimizeL2(X_trn, y_trn)
        y_trn_hat_l2 = X_trn @ w_l2
        w_l1 = minimizeL1(X_trn, y_trn)
        y_trn_hat_l1 = X_trn @ w_l1
        w_linf = minimizeLinf(X_trn, y_trn)
        y_trn_hat_linf = X_trn @ w_linf

        trn_losses += (np.array([
            [L2Loss(y_trn, y_trn_hat_l2), L1Loss(y_trn, y_trn_hat_l2), LinfLoss(y_trn, y_trn_hat_l2)],
            [L2Loss(y_trn, y_trn_hat_l1), L1Loss(y_trn, y_trn_hat_l1), LinfLoss(y_trn, y_trn_hat_l1)],
            [L2Loss(y_trn, y_trn_hat_linf), L1Loss(y_trn, y_trn_hat_linf), LinfLoss(y_trn, y_trn_hat_linf)]
        ]) - trn_losses) / exp

        y_tst_hat_l2 = X_tst @ w_l2
        y_tst_hat_l1 = X_tst @ w_l1
        y_tst_hat_linf = X_tst @ w_linf

        tst_losses += (np.array([
            [L2Loss(y_tst, y_tst_hat_l2), L1Loss(y_tst, y_tst_hat_l2), LinfLoss(y_tst, y_tst_hat_l2)],
            [L2Loss(y_tst, y_tst_hat_l1), L1Loss(y_tst, y_tst_hat_l1), LinfLoss(y_tst, y_tst_hat_l1)],
            [L2Loss(y_tst, y_tst_hat_linf), L1Loss(y_tst, y_tst_hat_linf), LinfLoss(y_tst, y_tst_hat_linf)]
        ]) - tst_losses) / exp
    
    return trn_losses, tst_losses


def testLinearRegression():

	np.random.seed(RANDOM_SEED)

	X, y = generateRegData(n=30, d=1)

	plt.scatter(unAugmentX(X), y, marker='x', color='k')  # plot data points

	w_L2 = minimizeL2(X, y)
	y_hat_L2 = X @ w_L2
	w_L1 = minimizeL1(X, y)
	y_hat_L1 = X @ w_L1
	w_Linf = minimizeLinf(X, y)
	y_hat_Linf = X @ w_Linf
	w_L2_gd_init = np.random.randn(2, 1)
	w_L2_gd = gd(linearRegL2Obj, w_L2_gd_init, X, y, 0.1, 1000, 1e-10)
	y_hat_L2_gd = X @ w_L2_gd

	# compare OLS analytic solution & GD
	assert np.isclose(w_L2_gd,w_L2).all() 

	# change in objective function value
	lin_reg_L2_obj_pre, _ = linearRegL2Obj(w_L2_gd_init, X, y)
	lin_reg_L2_obj_post, _ = linearRegL2Obj(w_L2_gd, X, y)
	print(f'Linear regression L2 objective pre optimization: {lin_reg_L2_obj_pre}')
	print(f'Linear regression L2 objective post optimization: {lin_reg_L2_obj_post}')

	# plot models
	plt.plot(X[:, 1], y_hat_L2, 'b', marker=None, label='L2 Analytic')
	plt.plot(X[:, 1], y_hat_L1, 'r', marker=None, label='L1')
	plt.plot(X[:, 1], y_hat_Linf, 'g', marker=None, label='Linf')
	plt.plot(X[:, 1], y_hat_L2_gd, 'y', marker=None, label='L2 GD')
	plt.legend()
	plt.show()

	# run synthetic regression experiments
	np.random.seed(RANDOM_SEED)
	trn_losses, tst_losses = runSynthLinRegExperiments()
	print('Synthetic data experiments train set losses:')
	plt.figure(figsize=(3, 3))
	sns.heatmap(
		trn_losses,
		annot=True,
		fmt=".5f",
		linewidths=0.5,
		cbar_kws={"shrink": 0.5},
        cmap=sns.cm.rocket_r
	)
	plt.show()
	print('Synthetic data experiments test set losses:')
	plt.figure(figsize=(3, 3))    
	sns.heatmap(
		tst_losses,
		annot=True,
		fmt=".5f",
		linewidths=0.5,
		cbar_kws={"shrink": 0.5},
        cmap=sns.cm.rocket_r
	)
	plt.show()

