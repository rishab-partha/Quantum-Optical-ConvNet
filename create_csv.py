import numpy as np
import glob
import cv2

file = open('data.csv', 'w')
file.write("names,")
for i in range(783):
    file.write("col_" + str(i) + ",")
file.write("col_783 \n")

count = 0
data = np.zeros((785, 1))
for path in glob.iglob("Images/*/*.png"):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).reshape(784)
    num = int(path[14:15])
    file.write(str(num) + ",")
    for i in range(783):
        file.write(str(img[i]) + ",")
    file.write(str(img[i]) + "\n")
    count += 1
    if (count % 1000) == 0:
        print(count)


    