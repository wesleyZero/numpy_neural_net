import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt 
# import tensorflow 		# Dependency for Keras
# import keras

"""
	-----	
"""



#read
train_data = pd.read_csv('../assets/MNIST_CSV/train.csv')
test_data = pd.read_csv('../assets/MNIST_CSV/test.csv')
# print(train_data.columns)
# print(train_data['label'])

#convert
train_data = np.array(train_data)
test_data = np.array(test_data)

#transpose 
# train_data = train_data.T		# Each row will be an image
# train_data = train_data.T		# Each row will be an image 

m_train, n_train = train_data.shape
m_test, n_test = test_data.shape


# Train
y = train_data[ : , 0]
x = train_data[ : , 1:]

# print(y)



class neural_net():
	def __init__(self, dataset):
		"""
		Interpets the datashape as (numOfImages, 1 + numOfPixels) where the 
		first column is the labeled data
		"""
		# if type(dataset) != type(np.array([])):
		if type(dataset) is not np.array:
			dataset = np.array(dataset)
		self.y = dataset[ : , 0 ]
		self.x = dataset[ : , 1 : ]
		self.params = neural_net.init_params()

	@staticmethod
	def init_params():
		w1 = np.random.rand(10, 784)
		b1 = np.random.rand(10, 1)
		w2 = np.random.rand(10, 10)
		b2 = np.random.rand(10, 1)
		return w1, b1, w2, b2

	@staticmethod
	def ReLU(x):
		return max(0, x)

	def forward_prop():
		...


# neural_net.init_params()
nn = neural_net(train_data)




































print("The program ran")




