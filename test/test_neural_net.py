import pytest
import sys
sys.path.append('../src/')
from numpy_neural_net import neural_net
import numpy as np
import pandas as pd

@pytest.fixture(scope='module')
def	nn_mnist():
	train_data = pd.read_csv('../assets/MNIST_CSV/train.csv')
	test_data = pd.read_csv('../assets/MNIST_CSV/test.csv')
	test_data = np.array(test_data).T
	nn = neural_net(train_data)
	yield nn 

class Test_neural_net():

		@staticmethod
		@pytest.mark.parametrize(
			('input', 'expected'),
			((i, 0) for i in range(-1, -4, -1) ),
			)
		def	test_ReLU_negativeInput(input, expected):
			assert True if input and not expected else False 			
		
		@staticmethod
		def test_myOwnTest():
			assert True 
		
		@staticmethod	
		@pytest.mark.skip
		def test_skipMeFunction():
			assert False
		
		@staticmethod
		@pytest.mark.xfail
		def test_ImJustAFailureWaitingToHappen():
			assert False


		@staticmethod
		@pytest.mark.parametrize(
			('input', 'expected'),
			((i, 0) for i in range(0, -4, -1) ),
		)
		def	test_ReLU(input, expected, nn_mnist): 
			assert neural_net.ReLU(input) == expected

# this is a test commit using a different git push method 