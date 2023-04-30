import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt 
# import tensorflow 		# Dependency for Keras
# import keras

"""
	-----	
"""

class neural_net():
	"""
		A neural net for image classification of 28x28 images

		The layers will be numbered as followed
		0 : input later 
		1 : hidden layer 
		2 : output later
	"""
	def __init__(self, training_data):
		"""
		Interpets the datashape as (numOfImages, 1 + numOfPixels) where the 
		first column is the labeled data
		"""
		if type(training_data) is not np.array:
			training_data = np.array(training_data)
		training_data = training_data.T #Col data is now row data
		self.y = training_data[0]		#First Row is labels	
		self.x = training_data[1:] / 255	#Rest of rows is training data
		length = len(self.x[0])
		self.init_params(length)

	def init_params(self, length):
		"""
		Makes a neural network that is 784 -> 10(hidden layer) -> 10 
		"""
		self.w1 = np.random.rand(10, 784) - .5
		self.b1 = np.random.rand(10, 1) - .5
		self.w2 = np.random.rand(10, 10) - .5
		self.b2 = np.random.rand(10, 1) - .5
		return True

	@staticmethod
	def ReLU(Z):
		return np.maximum(0,Z)

	@staticmethod	
	def softmax(Z):
		top = np.exp(Z)
		bottom = np.sum(np.exp(Z))	
		return top / bottom

	@staticmethod
	def deriv_ReLU(Z):
		return Z > 0
	
	@staticmethod
	def one_hot(Y):
		one_hot_Y = np.zeros((Y.size, Y.max() + 1))
		one_hot_Y[np.arange(Y.size), Y)] = 1
		one_hot_Y = one_hot_Y.T
		return one_hot_Y
		
	def forward_prop(self):
		self.z1 = self.w1.dot(self.x) + self.b1
		self.a1 = neural_net.ReLU(self.z1)
		self.z2 = self.w2.dot(self.a1) + self.b2
		self.a2 = neural_net.softmax(self.z2)

	def backward_prop(self):
		# output layer to hidden layer
		self.dz2 = self.a2 - self.y
		self.dw2 = self.dz2 




#read
train_data = pd.read_csv('../assets/MNIST_CSV/train.csv')
test_data = pd.read_csv('../assets/MNIST_CSV/test.csv')

nn = neural_net(train_data)
nn.forward_prop()
































print("The program ran")




