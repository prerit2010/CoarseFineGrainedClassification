from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import cv2
import numpy as np

img= cv2.imread("00004.jpg") 
labimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

n = 0
while(n<4):
    labimg = cv2.pyrDown(labimg)
    n = n+1

feature_image=np.reshape(labimg, [-1, 3])
rows, cols, chs = labimg.shape

db = DBSCAN(eps=5, min_samples=2, metric = 'euclidean',algorithm ='auto')
db.fit(feature_image)
labels = db.labels_

plt.figure(2)
plt.subplot(2, 1, 1)
plt.imshow(img)
plt.axis('off')
plt.subplot(2, 1, 2)
plt.imshow(np.reshape(labels, [rows, cols]))
plt.axis('off')
plt.show()