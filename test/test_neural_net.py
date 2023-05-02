import pytest
# import "../src/neural_net.py"
# import ../src
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



# class Test_neural_net():
			
# 		@pytest.mark.parametrize(
# 			('input', 'expected'),
# 			(
# 				(-1, 0),
# 			),
# 		)
# 		def	test_ReLU_2(input, expected):
# 			assert input == -1
# 			assert expected == 0

@pytest.mark.parametrize(
	('input', 'expected'),
	tuple( [(i, 0) for i in range(0, -4, -1)]),
)
def	test_ReLU(input, expected, nn_obj_fixture): 
	assert neural_net.ReLU(input) == expected



train_data = pd.read_csv('../assets/MNIST_CSV/train.csv')
test_data = pd.read_csv('../assets/MNIST_CSV/test.csv')
test_data = np.array(test_data).T

# def test_accuracy()

# all_time_high = 0
# nn = neural_net(train_data)
# for i in range(100):
# 	nn.gradient_descent(iterations=100, alpha=.1, update=-1)
# 	if i > 0: last_test_accuracy = test_accuracy
# 	test_accuracy = nn.test_accuracy(test_data)
# 	print(f'accuracy = {test_accuracy}')
# 	if i > 0 and last_test_accuracy > test_accuracy: 
# 		if last_test_accuracy > all_time_high:
# 			all_time_high = last_test_accuracy
# 			print(f'/\nall time high accuracy = {all_time_high}')
# 		print(f"\t\t⬇️ accuracy {test_accuracy} ")
# 		# nn = neural_net(train_data)	

