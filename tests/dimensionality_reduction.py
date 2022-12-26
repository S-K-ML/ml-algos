import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
from mlalgos.dimensionality_reduction import *
from mlalgos.utils import (
    unAugmentX,
    augmentX
)
from mlalgos.classification import (
    minMulDev, 
    mulClassify,
    mulAccuracyScore
)
from tests.config import RANDOM_SEED
from tests.common import generateMulClassData


def runSynthClsExperimentsPCA() -> Tuple[np.ndarray, np.ndarray]:
    '''
    Evaluates PCA implementation on synthetic dataset.

    Returns:
        2 x 2 matrix train_acc of average training accuracies
        2 x 2 matrix test_acc of average test accuracies
    '''
    n_runs = 100
    n_train = 128
    n_test = 1000

    dim_list = [1, 2]
    gen_model_list = [1, 2]

    train_acc = np.zeros([len(dim_list), len(gen_model_list)])
    test_acc = np.zeros([len(dim_list), len(gen_model_list)])

    for r, (i, k), (j, gen_model) in itertools.product(
        range(1, n_runs + 1), enumerate(dim_list), enumerate(gen_model_list)
    ):
        Xtrain, Ytrain = generateMulClassData(n=n_train, gen_model=gen_model)
        Xtest, Ytest = generateMulClassData(n=n_test, gen_model=gen_model)

        Xtrain = unAugmentX(Xtrain) # remove augmentation before PCA
        Xtest = unAugmentX(Xtest)

        U = PCA(Xtrain, k)
        Xtrain_mu = Xtrain.mean(axis=0)
        Xtrain_proj = projPCA(Xtrain, Xtrain_mu, U) #call projPCA to find the new features 
        Xtest_proj = projPCA(Xtest, Xtrain_mu, U) #call projPCA to find the new features
        
        Xtrain_proj = augmentX(Xtrain_proj) # add augmentation back 
        Xtest_proj = augmentX(Xtest_proj)
        
        W = minMulDev(Xtrain_proj, Ytrain) # from Q1
        Yhat = mulClassify(Xtrain_proj, W) # from Q1
        train_acc[i, j] += (mulAccuracyScore(Yhat, Ytrain) - train_acc[i, j]) / r
        
        Yhat = mulClassify(Xtest_proj, W)
        test_acc[i, j] += (mulAccuracyScore(Yhat, Ytest) - test_acc[i, j]) / r

    return train_acc, test_acc

def testPCA():

	np.random.seed(RANDOM_SEED)
	train_acc, test_acc = runSynthClsExperimentsPCA()
	print('Synthetic data experiments train set accuracies:')
	plt.figure(figsize=(2, 2))
	sns.heatmap(
		train_acc,
		annot=True,
		fmt=".5f",
		linewidths=0.5,
		cbar_kws={"shrink": 0.5},
	)
	plt.show()
	print('Synthetic data experiments test set accuracies:')
	plt.figure(figsize=(2, 2))
	sns.heatmap(
		test_acc,
		annot=True,
		fmt=".5f",
		linewidths=0.5,
		cbar_kws={"shrink": 0.5},
	)
	plt.show()