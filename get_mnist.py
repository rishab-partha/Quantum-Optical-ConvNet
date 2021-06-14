'''
Extracts and Downloads the MNIST dataset as images.

@version 3.8.2021
'''

from tensorflow.keras.datasets import mnist
import imageio
import os

# Load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

os.system('mkdir Images')
os.system('cd Images')

# Create directories for each image label
for i in range(10):
	os.system('mkdir Images/number_' + str(i))

# Identify the frequency of each label
frequency = [0,0,0,0,0,0,0,0,0,0]

# Iterate through the training set
for i in range(60000):
	if ((i + 1) % 1000 == 0):
		print(i + 1)

	array = x_train[i]
	imageio.imwrite('Images/number_' + str(y_train[i]) + '/' + str(frequency[y_train[i]]) + '.png', array)

	frequency[y_train[i]] += 1

# Iterate through the Testing Set
for i in range(10000):
	if ((i + 1) % 1000 == 0):
		print(i + 60001)

	array = x_test[i]
	imageio.imwrite('Images/number_' + str(y_test[i]) + '/' + str(frequency[y_test[i]]) + '.png', array)

	frequency[y_test[i]] += 1