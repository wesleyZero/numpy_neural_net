import neural_net
import pandas as pd
import numpy as np

train_data = pd.read_csv('../assets/MNIST_CSV/train.csv')
test_data = pd.read_csv('../assets/MNIST_CSV/test.csv')
test_data = np.array(test_data).T

all_time_high = 0
nn = neural_net(train_data)
for i in range(100):
	nn.gradient_descent(iterations=100, alpha=.1, update=-1)
	if i > 0: last_test_accuracy = test_accuracy
	test_accuracy = nn.test_accuracy(test_data)
	print(f'accuracy = {test_accuracy}')
	if i > 0 and last_test_accuracy > test_accuracy: 
		if last_test_accuracy > all_time_high:
			all_time_high = last_test_accuracy
			print(f'/\nall time high accuracy = {all_time_high}')
		print(f"\t\t⬇️ accuracy {test_accuracy} ")
		# nn = neural_net(train_data)	



