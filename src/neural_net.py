import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt 
# import tensorflow 		# Dependency for Keras
# import keras

"""
	-----	
"""




train_data = pd.read_csv('../assets/MNIST_CSV/train.csv')
test_data = pd.read_csv('../assets/MNIST_CSV/test.csv')

train_data = np.array(train_data)
train_data = train_data.T		# Each column will be a image 
m, n = train_data.shape
print(f"{m}, {n}")



np.random.shuffle(train_data)





























print("The program ran")




