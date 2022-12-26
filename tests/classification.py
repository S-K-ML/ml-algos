import numpy as np
import jax
import jax.numpy as jnp
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from mlalgos.classification import *
from mlalgos.common import *
from mlalgos.utils import *
from tests.config import *
from tests.common import *

def generateBinClassData(
    n: int, 
    gen_model: int,
    d: int=2
):
    if gen_model == 0:
        c0 = np.ones([1, d])
        c1 = -np.ones([1, d])
        X0 = np.random.randn(n // 2, d) + c0
        X1 = np.random.randn(n // 2, d) + c1
        X = np.concatenate((X0, X1), axis=0)
        y = np.concatenate([np.ones([n // 2, 1]), np.zeros([n // 2, 1])], axis=0)

    elif gen_model == 1 or gen_model == 2:
        w_true = np.ones([d, 1])

        X = np.random.randn(n, d)

        if gen_model == 1:
            y = np.sign(X @ w_true)
        else:
            y = np.sign((X ** 2) @ w_true - 1)

    elif gen_model == 3:
        n_samples_out = n // 2
        n_samples_in = n - n_samples_out
        outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
        outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
        inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
        inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5

        X = np.vstack(
            [np.append(outer_circ_x, inner_circ_x), 
            np.append(outer_circ_y, inner_circ_y)]
        ).T
        X += np.random.randn(*X.shape) * 0.1

        y = np.hstack(
            [-np.ones(n_samples_out, dtype=np.intp), 
            np.ones(n_samples_in, dtype=np.intp)]
        )[:, None]

    else:
        raise ValueError("Unknown generative model")

    return X, y


def plotBinClassPoints(
    X: np.ndarray, 
    y: np.ndarray
):
    X0 = X[y.flatten() > 0]
    X1 = X[y.flatten() <= 0]

    plt.scatter(X0[:, 0], X0[:, 1], marker='x', label=f'class {int(min(y)[0])}')
    plt.scatter(X1[:, 0], X1[:, 1], marker='o', label=f'class {int(max(y)[0])}')


def runSynthExpBinClassLogReg():
    '''
    Evaluates the logistic regression implementation on synthetic dataset.

    Returns:
        4 x 3 matrix of average training accuracies and 4 x 3 matrix of average test accuracies
    '''

    n_exp = 100 # number of experiments
    max_iter = 1000 # maximum number of iterations
    tol = 1e-10 # tolerance
    m_tst = 1000 # number of test points *per class*
    m_trn_default = 100 # default number of train points *per class*
    d_default = 2 # default number of dimensions
    eta_default = 0.1 # default learning rate

    hyperparams_space = {
        'm_trn' : [10, 50, 100, 200], # number of test points per class hyperparameter values
        'd' : [1, 2, 4, 8], # number of dimensions hyperparameter values
        'eta' : [0.1, 1, 10, 100] # learning rate hyperparameter values
    }    
    n_hyperparams = 3 # number of hyperparameters
    n_hyperparam_vals = 4 # number of values for each hyperparameter

    trn_accs = np.zeros((n_hyperparam_vals, n_hyperparams))
    tst_accs = np.zeros((n_hyperparam_vals, n_hyperparams))

    for exp, hyperparam_idx, hyperparam_val_idx in itertools.product(
        range(1, n_exp + 1), range(n_hyperparams), range(n_hyperparam_vals)
    ):
        hyperparams = {
            'm_trn' : m_trn_default,
            'd' : d_default,
            'eta' : eta_default
        }
        # pick hyperparameter value and fix the rest
        hyperparam = list(hyperparams_space.keys())[hyperparam_idx]
        hyperparam_val = hyperparams_space[hyperparam][hyperparam_val_idx]
        hyperparams[hyperparam] = hyperparam_val

        # train set learning and evalutation
        X_trn, y_trn = generateBinClassData(hyperparams['m_trn'], gen_model=0, d=hyperparams['d'])
        X_trn = augmentX(X_trn)

        w_init = np.random.randn(hyperparams['d'] + 1, 1)
        w_logit = gd(logisticRegObj, w_init, X_trn, y_trn, hyperparams['eta'], max_iter, tol)

        y_trn_hat = np.heaviside(X_trn @ w_logit, 1).astype(int)

        trn_accs[hyperparam_val_idx, hyperparam_idx] += (
            accuracy(y_trn, y_trn_hat) - trn_accs[hyperparam_val_idx, hyperparam_idx]
        ) / exp

        # test set evaluation
        X_tst, y_tst = generateBinClassData(m_tst, gen_model=0, d=hyperparams['d'])
        X_tst = augmentX(X_tst)

        y_tst_hat = np.heaviside(X_tst @ w_logit, 1).astype(int)

        tst_accs[hyperparam_val_idx, hyperparam_idx] += (
            accuracy(y_tst, y_tst_hat) - tst_accs[hyperparam_val_idx, hyperparam_idx]
        ) / exp
    
    return trn_accs, tst_accs


def testBinClassLogisticRegression():
	np.random.seed(RANDOM_SEED)

	m = 100
	d = 2
	X, y = generateBinClassData(m, gen_model=0, d=d)

	# plot data points
	plotBinClassPoints(X, y)

	X = augmentX(X)

	# gradient descent
	w_gd_init = np.random.randn(d + 1, 1)
	w_gd = gd(logisticRegObj, w_gd_init, X, y, 0.1, 1000, 1e-10)
	y_hat_gd = np.heaviside(X @ w_gd, 1).astype(int)
	log_reg_acc = accuracy(y, y_hat_gd)
	print(f'Logistic regression model accuracy: {log_reg_acc}')

	# change in objective function value
	log_reg_obj_pre, log_reg_init_grad = logisticRegObj(w_gd_init, X, y)
	log_reg_obj_post, _ = logisticRegObj(w_gd, X, y)
	print(f'Logistic regression objective pre optimization: {log_reg_obj_pre}')
	print(f'Logistic regression objective post optimization: {log_reg_obj_post}')

	# compare analytic gradient and numeric
	obj_func_init = lambda w: (jnp.dot(-y.reshape(-1), jnp.log(1 / (1 + jnp.exp(-X @ w))))[0] + jnp.dot(-(1-y).reshape(-1), jnp.log(1-1/(1+jnp.exp(-X @ w))))[0]) / len(y)
	log_reg_init_grad_numeric = jax.grad(obj_func_init)(w_gd_init)
	assert np.isclose(log_reg_init_grad, log_reg_init_grad_numeric).all()
	
	# plot models
	x_grid = np.arange(-4, 4, 0.01)
	plt.plot(x_grid, (-w_gd[0]-w_gd[1]*x_grid) / w_gd[2], '--k', label='decision boundary')
	plt.legend()
	plt.show()

	# run synthetic classification experiments
	np.random.seed(RANDOM_SEED)
	trn_accs, tst_accs = runSynthExpBinClassLogReg()
	print('Synthetic data experiments train set accuracies:')
	plt.figure(figsize=(3, 4))    
	sns.heatmap(
		trn_accs,
		annot=True,
		fmt=".5f",
		linewidths=0.5,
		cbar_kws={"shrink": 0.5},
	)
	plt.show()
	print('Synthetic data experiments test set accuracies:')
	plt.figure(figsize=(3, 4))    
	sns.heatmap(
		tst_accs,
		annot=True,
		fmt=".5f",
		linewidths=0.5,
		cbar_kws={"shrink": 0.5},
	)
	plt.show()


def getRange(
    X: np.ndarray
):
    x_min = np.amin(X[:, 0]) - 0.1
    x_max = np.amax(X[:, 0]) + 0.1
    y_min = np.amin(X[:, 1]) - 0.1
    y_max = np.amax(X[:, 1]) + 0.1
    return x_min, x_max, y_min, y_max


def plotPrimalModelBinClass(
    X: np.ndarray, 
    y: np.ndarray, 
    w: np.ndarray, 
    w0: float
):

    plotBinClassPoints(X, y)

    # plot model
    x_min, x_max, y_min, y_max = getRange(X)
    grid_step = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))
    z = classify(np.c_[xx.ravel(), yy.ravel()], w, w0)

    # Put the result into a color plot
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.RdBu, alpha=0.5)
    plt.legend()
    plt.show()


def plotAdjModelBinClass(
    X: np.ndarray, 
    y: np.ndarray, 
    a: np.ndarray, 
    a0: float, 
    kernel_func: Callable
):
    plotBinClassPoints(X, y)

    # plot model
    x_min, x_max, y_min, y_max = getRange(X)
    grid_step = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))
    z = adjClassify(np.c_[xx.ravel(), yy.ravel()], a, a0, X, kernel_func)

    # Put the result into a color plot
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.RdBu, alpha=0.5)
    plt.legend()
    plt.show()


def plotDualModelBinClass(
    X: np.ndarray, 
    y: np.ndarray, 
    a: np.ndarray, 
    b: float, 
    lamb: float, 
    kernel_func: Callable, 
):

    plotBinClassPoints(X, y)

    # plot model
    x_min, x_max, y_min, y_max = getRange(X)
    grid_step = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))
    z = dualClassify(np.c_[xx.ravel(), yy.ravel()], a, b, X, y, 
                     lamb, kernel_func)

    # Put the result into a color plot
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.RdBu, alpha=0.5)
    plt.legend()
    plt.show()


def runSynthExpBinClassPrimalModels() -> Tuple[np.ndarray, np.ndarray]:
    '''
    Evaluates primal form classifiers (Binomial deviance + L2 reg & Hinge loss + L2 reg) on synthetic data.

    Returns:
        4 x 6 matrix of average training accuracies and a 4 x 6 matrix of average test accuracies 
    '''
    n_runs = 100 # number of experiments
    n_train = 100 # number of train set data points
    n_test = 1000 # number of test set data points
    lamb_list = [0.001, 0.01, 0.1, 1.] # lambda values
    gen_model_list = [1, 2, 3] # data generating models list
    
    train_acc_bindev = np.zeros([len(lamb_list), len(gen_model_list)])
    test_acc_bindev = np.zeros([len(lamb_list), len(gen_model_list)])
    train_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list)])
    test_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list)])

    for run , l_idx, gm_idx in itertools.product(
        range(1, n_runs + 1), range(len(lamb_list)), range(len(gen_model_list))
    ):

        lamb = lamb_list[l_idx]
        gen_model = gen_model_list[gm_idx]

        Xtrain, ytrain = generateBinClassData(n=n_train, gen_model=gen_model)
        Xtest, ytest = generateBinClassData(n=n_test, gen_model=gen_model)

        # Binomial Deviance + L2 reg
        # Train set learning and evaluation
        w, w0 = minBinDev(Xtrain, ytrain, lamb)
        y_trn_hat = classify(Xtrain, w, w0)
        train_acc_bindev[l_idx, gm_idx] += (
            np.mean(ytrain == y_trn_hat) - train_acc_bindev[l_idx, gm_idx]
        ) / run
        # Test set evaluation
        y_tst_hat = classify(Xtest, w, w0)
        test_acc_bindev[l_idx, gm_idx] += (
            np.mean(ytest == y_tst_hat) - test_acc_bindev[l_idx, gm_idx]
        ) / run

        # Hinge Loss + L2 reg
        # Train set learning and evaluation
        w, w0 = minHinge(Xtrain, ytrain, lamb)
        y_trn_hat = classify(Xtrain, w, w0)
        train_acc_hinge[l_idx, gm_idx] += (
            np.mean(ytrain == y_trn_hat) - train_acc_hinge[l_idx, gm_idx]
        ) / run
        # Test set evaluation
        y_tst_hat = classify(Xtest, w, w0)
        test_acc_hinge[l_idx, gm_idx] += (
            np.mean(ytest == y_tst_hat) - test_acc_hinge[l_idx, gm_idx]
        ) / run

    return (
        np.concatenate([train_acc_bindev, train_acc_hinge], axis=1),
        np.concatenate([test_acc_bindev, test_acc_hinge], axis=1)
    )


def testPrimalFormBinClassifiers():
	n = 100
	lamb = 0.1

	print('Primal form:', end='\n\n')
	for data_model in [1,2,3]:
		# Generate data
		print(f'Data model {data_model}:', end='\n\n')
		np.random.seed(RANDOM_SEED)
		Xtrain, ytrain = generateBinClassData(n=n, gen_model=data_model)

		for model, model_name in zip(
			[minBinDev, minHinge], 
			['Binomial deviance loss + L2 reg model', 'Hinge loss + L2 reg model']
		):
			# Learn and plot results
			print(f'{model_name}:')
			w, w0 = model(Xtrain, ytrain, lamb)
			plotPrimalModelBinClass(Xtrain, ytrain, w, w0)
			y_trn_hat = classify(Xtrain, w, w0)
			trn_acc = np.mean(ytrain == y_trn_hat)
			print(f'Train set accuracy: {trn_acc}', end='\n\n')

	# Synthetic experiments
	np.random.seed(RANDOM_SEED)
	trn_accs, tst_accs = runSynthExpBinClassPrimalModels()
	print('Synthetic data experiments train set accuracies:')
	plt.figure(figsize=(6, 4))        
	sns.heatmap(
		trn_accs,
		annot=True,
		fmt=".5f",
		linewidths=0.5,
		cbar_kws={"shrink": 0.5},
	)
	plt.show()
	print('Synthetic data experiments test set accuracies:')
	plt.figure(figsize=(6, 4))        
	sns.heatmap(
		tst_accs,
		annot=True,
		fmt=".5f",
		linewidths=0.5,
		cbar_kws={"shrink": 0.5},
	)
	plt.show()


def runSyntExpBinClassAdjModels():
    '''
    Runs synthetic experiments for adjoint form classifiers.
    '''
    n_runs = 10
    n_train = 100
    n_test = 1000
    lamb = 0.001

    kernel_list = [linearKernel,
        lambda X1, X2: polyKernel(X1, X2, 2),
        lambda X1, X2: polyKernel(X1, X2, 3),
        lambda X1, X2: gaussKernel(X1, X2, 1.0),
        lambda X1, X2: gaussKernel(X1, X2, 0.5),
        lambda X1, X2: gaussKernel(X1, X2, 0.1)]

    gen_model_list = [1, 2, 3]

    train_acc_bindev = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    test_acc_bindev = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    train_acc_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    test_acc_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])

    for r in range(n_runs):
        for i, kernel in enumerate(kernel_list):
            for j, gen_model in enumerate(gen_model_list):

                Xtrain, ytrain = generateBinClassData(n=n_train, gen_model=gen_model)
                Xtest, ytest = generateBinClassData(n=n_test, gen_model=gen_model)

                a, a0 = adjBinDev(Xtrain, ytrain, lamb, kernel)

                train_acc_bindev[i, j, r] = np.mean(ytrain == adjClassify(Xtrain, a, a0, Xtrain, kernel))
                test_acc_bindev[i, j, r] =  np.mean(ytest == adjClassify(Xtest, a, a0, Xtrain, kernel))

                a, a0 = adjHinge(Xtrain, ytrain, lamb, kernel)
                
                train_acc_hinge[i, j, r] = np.mean(ytrain == adjClassify(Xtrain, a, a0, Xtrain, kernel))
                test_acc_hinge[i, j, r] = np.mean(ytest == adjClassify(Xtest, a, a0, Xtrain, kernel))

    train_acc_bindev_average = np.mean(train_acc_bindev, axis=2)
    test_acc_bindev_average = np.mean(test_acc_bindev, axis=2)
    train_acc_hinge_average = np.mean(train_acc_hinge, axis=2)
    test_acc_hinge_average = np.mean(test_acc_hinge, axis=2)

    return (
        np.concatenate([train_acc_bindev_average, train_acc_hinge_average], axis=1),
        np.concatenate([test_acc_bindev_average, test_acc_hinge_average], axis=1)
    )


def testAdjointFormBinClassifiers():
	n = 100
	lamb = 0.1

	for data_model in [1,2,3]:
		# Generate data
		print(f'Data model {data_model}:', end='\n\n')
		np.random.seed(RANDOM_SEED)
		Xtrain, ytrain = generateBinClassData(n=n, gen_model=data_model)

		for kernel_func, kernel_name in zip(
			[
				linearKernel, 
				lambda X1, X2: polyKernel(X1, X2, 2),
				lambda X1, X2: polyKernel(X1, X2, 3),
				lambda X1, X2: gaussKernel(X1, X2, 0.5)
			], 
			['Linear kernel', 'Poly deg 2 kernel', 'Poly deg 3 kernel', 'Gauss width 0.5 kernel']
		):
			print(f'{kernel_name}:', end='\n\n')
			for model, model_name in zip(
				[adjBinDev, adjHinge], 
				[	
					'Adjoint Binomial deviance loss + L2 reg model ', 
					'Adjoint Hinge loss + L2 reg model'
				]
			):
				# Learn and plot results
				print(f'{model_name}:')
				a, a0 = model(Xtrain, ytrain, lamb, kernel_func)
				plotAdjModelBinClass(Xtrain, ytrain, a, a0, kernel_func)
				y_trn_hat = adjClassify(Xtrain, a, a0, Xtrain, kernel_func)
				trn_acc = np.mean(ytrain == y_trn_hat)
				print(f'Train set accuracy: {trn_acc}', end='\n\n')

	# Synthetic experiments
	np.random.seed(RANDOM_SEED)
	trn_accs, tst_accs = runSyntExpBinClassAdjModels()
	print('Synthetic data experiments train set accuracies:')
	plt.figure(figsize=(6, 6))        
	sns.heatmap(
		trn_accs,
		annot=True,
		fmt=".5f",
		linewidths=0.5,
		cbar_kws={"shrink": 0.5},
	)
	plt.show()
	print('Synthetic data experiments test set accuracies:')
	plt.figure(figsize=(6, 6))        
	sns.heatmap(
		tst_accs,
		annot=True,
		fmt=".5f",
		linewidths=0.5,
		cbar_kws={"shrink": 0.5},
	)
	plt.show()


def testDualFormBinClassifiers():
	n = 100
	lamb = 0.1    	

	print('Dual form:', end='\n\n')
	for data_model in [1,2,3]:
		# Generate data
		print(f'Data model {data_model}:', end='\n\n')
		np.random.seed(RANDOM_SEED)
		Xtrain, ytrain = generateBinClassData(n=n, gen_model=data_model)

		for kernel_func, kernel_name in zip(
			[
				linearKernel, 
				lambda X1, X2: polyKernel(X1, X2, 2),
				lambda X1, X2: polyKernel(X1, X2, 3),
				lambda X1, X2: gaussKernel(X1, X2, 0.5)
			], 
			['Linear kernel', 'Poly deg 2 kernel', 'Poly deg 3 kernel', 'Gauss width 0.5 kernel']
		):
			# Learn and plot results
			print(f'{kernel_name}:')
			a, b = dualHinge(Xtrain, ytrain, lamb, kernel_func)
			plotDualModelBinClass(Xtrain, ytrain, a, b, lamb, kernel_func)
			y_trn_hat = dualClassify(Xtrain, a, b, Xtrain, ytrain, lamb, kernel_func)
			trn_acc = np.mean(ytrain == y_trn_hat)
			print(f'Train set accuracy: {trn_acc}', end='\n\n')


def plotMulClassModel(
    X: np.ndarray, 
    Y: np.ndarray, 
    W: np.ndarray
):

    plotMulClassPoints(X, Y)

    # plot model
    x_min, x_max, y_min, y_max = getRange(unAugmentX(X))
    grid_step = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))
    Y = mulClassify(np.c_[np.ones(len(xx.ravel())), xx.ravel(), yy.ravel()], W)
    labels = Y.argmax(axis=1)

    # Put the result into a color plot
    labels = labels.reshape(xx.shape)
    plt.contourf(xx, yy, labels, 
                 colors=['r', 'r', 'b', 'b', 'g', 'g', 'y', 'y'], 
                 alpha=0.3)
    plt.legend()
    plt.show()
    return


def runSynthExpMultiClass():
    n_runs = 10
    n_test = 1000
    n_train_list = [16, 32, 64, 128]
    gen_model_list = [1, 2]

    train_acc = np.zeros([len(n_train_list), len(gen_model_list), n_runs])
    test_acc = np.zeros([len(n_train_list), len(gen_model_list), n_runs])

    for r in range(n_runs):
        for i, n_train in enumerate(n_train_list):
            for j, gen_model in enumerate(gen_model_list):

                Xtrain, Ytrain = generateMulClassData(n_train, gen_model)
                Xtest, Ytest = generateMulClassData(n_test, gen_model)
                
                W = minMulDev(Xtrain, Ytrain)

                Yhat = mulClassify(Xtrain, W)
                train_acc[i, j, r] = mulAccuracyScore(Yhat, Ytrain)

                Yhat = mulClassify(Xtest, W)
                test_acc[i, j, r] = mulAccuracyScore(Yhat, Ytest)

    train_acc = np.mean(train_acc, axis=2)
    test_acc = np.mean(test_acc, axis=2)
    
    return train_acc, test_acc


def testMulClassification():

	n = 100

	for model in [1, 2]:
		print(f'Generative model {model}:', end='\n\n')

		np.random.seed(RANDOM_SEED)
		Xtrain, Ytrain = generateMulClassData(n=n, gen_model=model)
		W = minMulDev(Xtrain, Ytrain)
		plotMulClassModel(Xtrain, Ytrain, W)
		accuracy = mulAccuracyScore(Ytrain, mulClassify(Xtrain, W))
		print(f'Accuracy: {accuracy}', end='\n\n')
		
	print()
	np.random.seed(RANDOM_SEED)
	train_acc, test_acc = runSynthExpMultiClass()
	print('Synthetic data experiments train set accuracies:')
	plt.figure(figsize=(2, 4))
	sns.heatmap(
		train_acc,
		annot=True,
		fmt=".5f",
		linewidths=0.5,
		cbar_kws={"shrink": 0.5},
	)
	plt.show()
	print('Synthetic data experiments test set accuracies:')
	plt.figure(figsize=(2, 4))
	sns.heatmap(
		test_acc,
		annot=True,
		fmt=".5f",
		linewidths=0.5,
		cbar_kws={"shrink": 0.5},
	)
	plt.show()


