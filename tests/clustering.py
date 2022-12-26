import matplotlib.pyplot as plt
from mlalgos.clustering import *
from tests.common import (
    generateMulClassData,
    plotMulClassPoints
)
from mlalgos.utils import unAugmentX
from tests.config import RANDOM_SEED

def testKmeansClustering():

	n = 100

	for model in [1, 2]:
		print(f'Generative model {model}:', end='\n\n')

		np.random.seed(RANDOM_SEED)
		Xtrain, Ytrain = generateMulClassData(n, gen_model=model)
		print('Original clusters:')
		plotMulClassPoints(Xtrain, Ytrain)
		plt.legend()
		plt.show()
		print()

		for k in [4, 3, 2]:
			print(f'{k}-means clusters:')
			Y, U, obj_val = kmeans(unAugmentX(Xtrain), k)
			plotMulClassPoints(Xtrain, Y)
			plt.legend()
			plt.show()
			print(f'Objective value: {obj_val}', end='\n\n')

	print()
	print('K selection:', end='\n\n')

	k_candidates = [2,3,4,5,6,7,8,9]
	np.random.seed(RANDOM_SEED)
	Xtrain, Ytrain = generateMulClassData(n=100, gen_model=2)
	obj_val_list = np.array(list(map(lambda x: x[2], chooseK(unAugmentX(Xtrain), k_candidates))))

	plt.plot(k_candidates, obj_val_list)
	plt.xlabel('k')
	plt.ylabel('Objective value')
	plt.show()