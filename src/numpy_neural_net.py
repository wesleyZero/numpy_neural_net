import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt 
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
		Interpets the datashape of training_data as 
		(numOfImages, 1 + numOfPixels) where the 
		first column is the labeled data
		"""
		if type(training_data) is not np.array:
			training_data = np.array(training_data)
		# np.random.shuffle(training_data)
		training_data = training_data.T #Col data is now row data
		self.y = training_data[0]		#First Row is labels	
		self.x = training_data[1:] / 255	#Rest of rows is training data
		self.init_params()

	def init_params(self):
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
		return np.maximum(Z, 0)

	@staticmethod	
	def softmax(Z):
		return np.exp(Z) / sum(np.exp(Z))

	@staticmethod
	def deriv_ReLU(Z):
		return Z > 0
	
	@staticmethod
	def one_hot(Y):
		one_hot_Y = np.zeros((Y.size, Y.max() + 1))
		one_hot_Y[np.arange(Y.size), Y] = 1
		one_hot_Y = one_hot_Y.T
		return one_hot_Y
		
	def forward_prop(self):
		self.z1 = self.w1.dot(self.x) + self.b1
		self.a1 = neural_net.ReLU(self.z1)
		self.z2 = self.w2.dot(self.a1) + self.b2
		self.a2 = neural_net.softmax(self.z2)
	
	def predict(self, test_data):
		z1 = self.w1.dot(test_data) + self.b1
		a1 = neural_net.ReLU(z1)
		z2 = self.w2.dot(a1) + self.b2
		a2 = neural_net.softmax(z2)	
		a2 = np.argmax(a2, 0)
		return a2

	def backward_prop(self):
		m = self.y.size
		self.dz2 = self.a2 - neural_net.one_hot(self.y)
		self.dw2 = 1 / m * self.dz2.dot(self.a1.T)
		self.db2 = 1 / m * np.sum(self.dz2)

		self.dz1 = self.w2.T.dot(self.dz2) * neural_net.deriv_ReLU(self.z1)
		self.dw1 = 1 / m * self.dz1.dot(self.x.T)
		self.db1 = 1 / m * np.sum(self.dz1)

	def update_params(self, alpha):
		self.w1 = self.w1 - alpha * self.dw1
		self.b1 = self.b1 - alpha * self.db1 
		self.w2 = self.w2 - alpha * self.dw2
		self.b2 = self.b2 - alpha * self.db2
	
	def get_predictions(self):
		self.predictions = np.argmax(self.a2, 0)
	
	def get_accuracy(self):
		return np.sum(self.predictions == self.y) / self.y.size

	@staticmethod
	def accuracy(predictions, y):
		return np.sum(predictions == y) / y.size

	def test_accuracy(self, test_data):
		return np.sum(self.predict(test_data) == test_data) / test_data.size
	
	def gradient_descent(self, iterations, alpha, update=10):
		for i in range(iterations):
			self.forward_prop()
			self.backward_prop()
			self.update_params(alpha)
			if update == -1: continue #skip flag
			if i % update == 0:
				self.get_predictions()
				print(f"iteration {i} training acc : {self.get_accuracy()}")
		
#read
train_data = pd.read_csv('../assets/MNIST_CSV/train.csv')
test_data = pd.read_csv('../assets/MNIST_CSV/test.csv')
test_data = np.array(test_data).T

all_time_high = 0
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



# print(f"the ReLU of 0 is {neural_net.ReLU(0)}")
