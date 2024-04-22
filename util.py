import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def find_corners(im,
                 hsv_range=[(50,0,120),(75,255,255)],
                 show=False):
    
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(im_hsv,hsv_range[0],hsv_range[1])
    dest = cv2.cornerHarris(mask.astype(np.float32), 9, 5, 0.07)
    ind = np.argwhere(dest>0.2*dest.max())
    kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(ind)
    corners = kmeans.cluster_centers_
    corners = np.fliplr(corners)

    if show:
        plt.imshow(im)
        plt.scatter(corners[:,0],corners[:,1],color='r')
        plt.show()

    return corners