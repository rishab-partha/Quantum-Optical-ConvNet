'''
Generates the MNIST data in .csv formatting from a folder of images.

@version 3.8.2021
'''

import numpy as np
import glob
import cv2

# Open the file
file = open('data.csv', 'w')

# Write the headers for the file: (image label, pixels 0-783)
file.write("names,")
for i in range(783):
    file.write("col_" + str(i) + ",")
file.write("col_783 \n")

count = 0
data = np.zeros((785, 1))

# Iterate through the images
for path in glob.iglob("Images/*/*.png"):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).reshape(784)

    # Identify/Write Label
    num = int(path[14:15])
    file.write(str(num) + ",")

    # Identify/Write Pixel Values
    for i in range(783):
        file.write(str(img[i]) + ",")
    file.write(str(img[i]) + "\n")
    count += 1

    # Print update of how much has ran
    if (count % 1000) == 0:
        print(count)


    