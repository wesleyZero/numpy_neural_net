import pytest
import sys
sys.path.append('../src/')
from numpy_neural_net import neural_net
import numpy as np
import pandas as pd

@pytest.fixture
def	nn_obj_fixture():
	train_data = pd.read_csv('../assets/MNIST_CSV/train.csv')
	test_data = pd.read_csv('../assets/MNIST_CSV/test.csv')
	test_data = np.array(test_data).T
	nn = neural_net(train_data)
	yield nn 

class Test_neural_net():

		@pytest.mark.parametrize(
			('input', 'expected'),
			((i, 0) for i in range(0, -4, -1) ),
			)
		def	test_ReLU_2(self, input, expected):
			assert True if input and not expected else False 			
		
		@staticmethod
		def test_myOwnTest():
			assert True 

@pytest.mark.parametrize(
	('input', 'expected'),
	# tuple( [(i, 0) for i in range(0, -4, -1)]),
	((i, 0) for i in range(0, -4, -1) ),

)
def	test_ReLU(input, expected, nn_obj_fixture): 
	assert neural_net.ReLU(input) == expected
