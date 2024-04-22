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
    ind = np.argwhere(dest>0.1*dest.max())
    kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(ind)
    corners = kmeans.cluster_centers_
    corners = np.fliplr(corners)

    if show:
        plt.imshow(im)
        plt.scatter(corners[:,0],corners[:,1],color='r')
        plt.show()

    return corners

def sort_corners(corners):
    corners_center = np.mean(corners,axis=0,keepdims=True)
    offset = corners-corners_center

    i1 = np.argwhere((offset[:,0]<0)&(offset[:,1]<0))[0,0]
    i2 = np.argwhere((offset[:,0]>0)&(offset[:,1]<0))[0,0]
    i3 = np.argwhere((offset[:,0]>0)&(offset[:,1]>0))[0,0]
    i4 = np.argwhere((offset[:,0]<0)&(offset[:,1]>0))[0,0]
    return corners[[i1,i2,i3,i4]]

def find_phi(corners_0, corners_t):
    """find phi from the shift in x and y pixel positions
    """
    # corners are in (x,y)
    # upper left -> upper right -> lower right -> lower left
    corners_0 = sort_corners(corners_0)
    corners_t = sort_corners(corners_t)

    # find the z component from how much width and height changed
    width_top_0 = np.abs(corners_0[0,0]-corners_0[1,0])
    width_bottom_0 = np.abs(corners_0[2,0]-corners_0[3,0])

    width_top_t = np.abs(corners_t[0,0]-corners_t[1,0])
    width_bottom_t = np.abs(corners_t[2,0]-corners_t[3,0])

    # compute the average between the top and bottom width ratio
    phi_z = np.mean([width_top_0/width_top_t, width_bottom_0/width_bottom_t])

    # go through the 4 corner points and generate the 4 different phi matrix
    phi_matrices = []
    for i in range(4):
        dx,dy = corners_t[i,:]-corners_0[i,:]
        phi = np.eye(3)
        phi[0,2] = dx
        phi[1,2] = dy
        phi[2,2] = phi_z

        phi_matrices.append(phi)

    return phi_matrices