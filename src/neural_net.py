import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt 
# import tensorflow 		# Dependency for Keras
# import keras

"""
	-----	
"""

#transpose 
# train_data = train_data.T		# Each row will be an image
# train_data = train_data.T		# Each row will be an image 

# m_train, n_train = train_data.shape
# m_test, n_test = test_data.shape


# Train
# y = train_data[ : , 0]
# x = train_data[ : , 1:]

# print(y)


class neural_net():
	def __init__(self, training_data):
		"""
		Interpets the datashape as (numOfImages, 1 + numOfPixels) where the 
		first column is the labeled data
		"""
		# if type(dataset) != type(np.array([])):
		if type(training_data) is not np.array:
			training_data = np.array(training_data)
		training_data = training_data.T #Col data is now row data
		self.y = training_data[0]		#First Row is labels	
		self.x = training_data[1:]		#Rest of rows is training data
		self.init_params()

	def init_params(self):
		self.w1 = np.random.rand(10, 784)
		self.b1 = np.random.rand(10, 1)
		self.w2 = np.random.rand(10, 10)
		self.b2 = np.random.rand(10, 1)
		return True

	@staticmethod
	def ReLU(x):
		return np.maximum(0, x)	
	
	def softmax(self):
		return 
		
	def forward_prop(self):
		z1 = self.w1.dot(self.x) + self.b1
		a1 = neural_net.ReLU(z1)
		z2 = self.w2.dot(a1) + self.b2
		a2 = neural_net.ReLU(z2)



#read
train_data = pd.read_csv('../assets/MNIST_CSV/train.csv')
test_data = pd.read_csv('../assets/MNIST_CSV/test.csv')

nn = neural_net(train_data)
nn.forward_prop()
































print("The program ran")




