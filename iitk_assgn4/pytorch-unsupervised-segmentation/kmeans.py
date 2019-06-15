from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import cv2
import os


image1read = cv2.imread('00004.jpg')
(h1, w1) = image1read.shape[:2]
# image2read = cv2.imread('image2.jpg')
# (h2, w2) = image2read.shape[:2]
# image3read = cv2.imread('image3.jpg')
# (h3, w3) = image3read.shape[:2]



image1 = cv2.cvtColor(image1read, cv2.COLOR_BGR2LAB)
# image2 = cv2.cvtColor(image2read, cv2.COLOR_BGR2LAB)
# image3 = cv2.cvtColor(image3read, cv2.COLOR_BGR2LAB)


image1 = image1.reshape((image1.shape[0] * image1.shape[1], 3))
# image2 = image2.reshape((image2.shape[0] * image2.shape[1], 3))
# image3 = image3.reshape((image3.shape[0] * image3.shape[1], 3))


# clt = MiniBatchKMeans(n_clusters = 16)
clt = KMeans(n_clusters = 6,max_iter = 700, tol=0.000001)

labels1 = clt.fit_predict(image1)
quant1 = clt.cluster_centers_.astype("uint8")[labels1]

print("1st done")
# labels2 = clt.fit_predict(image2)
# quant2 = clt.cluster_centers_.astype("uint8")[labels2]

# print "2nd done"
# labels3 = clt.fit_predict(image3)
# quant3 = clt.cluster_centers_.astype("uint8")[labels3]

# print "3rd done"


#reshape the feature vectors to images
quant1 = quant1.reshape((h1, w1, 3))
image1 = image1.reshape((h1, w1, 3))

# quant2 = quant2.reshape((h2, w2, 3))
# image2 = image2.reshape((h2, w2, 3))

# quant3 = quant3.reshape((h3, w3, 3))
# image3 = image3.reshape((h3, w3, 3))

# convert from L*a*b* to RGB
quant1 = cv2.cvtColor(quant1, cv2.COLOR_LAB2BGR)
image1 = cv2.cvtColor(image1, cv2.COLOR_LAB2BGR)

# quant2 = cv2.cvtColor(quant2, cv2.COLOR_LAB2BGR)
# image2 = cv2.cvtColor(image2, cv2.COLOR_LAB2BGR)

# quant3 = cv2.cvtColor(quant3, cv2.COLOR_LAB2BGR)
# image3 = cv2.cvtColor(image3, cv2.COLOR_LAB2BGR)

path = 'clusteredImages'
# os.mkdir(path)
cv2.imwrite(os.path.join(path , 'image1reduced_6_maxiter700.jpg'), quant1)
# cv2.imwrite(os.path.join(path , 'image2reduced.jpg'), quant2)
# cv2.imwrite(os.path.join(path , 'image3reduced.jpg'), quant3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()