import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
from sklearn.cluster import KMeans

FOCAL_LENGTH = 589 # focal length


def filter_color(rgb_image, lower_bound_color, upper_bound_color):
    #convert the image into the HSV color space
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    # cv2.imshow("hsv image",hsv_image)

    #define a mask using the lower and upper bounds of the yellow color 
    mask = cv2.inRange(hsv_image, lower_bound_color, upper_bound_color)
    # cv2.imshow("Color Masking", mask)
    return mask

def getContours(binary_image):      
    contours, _ = cv2.findContours(binary_image.copy(), 
                                            cv2.RETR_EXTERNAL,
	                                        cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_ball_contour(binary_image, rgb_image, contours):
    # black_image = np.zeros([binary_image.shape[0], binary_image.shape[1],3],'uint8')
    ball_centers = [] 
    for c in contours:
        area = cv2.contourArea(c)
        perimeter= cv2.arcLength(c, True)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if (area>10):
            cv2.drawContours(rgb_image, [c], -1, (150,250,150), 1)
            cx, cy = get_contour_center(c)
            cv2.circle(rgb_image, (cx,cy), int(radius),(0,0,255),1)
            cv2.circle(rgb_image, (cx,cy),5,(150,150,255),-1) # draw center
            ball_centers.append([cx,cy])

    cv2.imshow("RGB Image Contours", rgb_image)
    return ball_centers

def get_contour_center(contour):
    M = cv2.moments(contour)
    cx=-1
    cy=-1
    if (M['m00']!=0):
        cx= int(M['m10']/M['m00'])
        cy= int(M['m01']/M['m00'])
    return cx, cy

def ball_detection(img_frame):
    redLower = (0, 25, 25)
    redUpper = (10, 255, 255)
    rgb_image = cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)

    binary_image_mask = filter_color(rgb_image, redLower, redUpper)
    contours = getContours(binary_image_mask)
    ball_centers = draw_ball_contour(binary_image_mask, rgb_image,contours)
    distance = np.sqrt((ball_centers[0][0]-ball_centers[1][0])**2 + (ball_centers[0][1]-ball_centers[1][1])**2)
    return ball_centers, distance

def compute_3d(corners_0, Z0s, fl):
    pts_3d = []
    for i in range(len(corners_0)):
        x,y = corners_0[i,:]
        Z0 = Z0s[i].item()
        X0 = x*Z0/fl
        y0 = y*Z0/fl
        pts_3d.append([X0, y0, Z0])
    return np.array(pts_3d)


def camera_matrix(fovy, height, width):
    f = 0.5 * height / math.tan(fovy * math.pi / 360)
    K = np.array([[-f, 0, width/2],
                  [0, f, height/2],
                  [0, 0, 1]])
    return f, K

def cumulative_int(dt, x):
    return dt * np.cumsum(x)

def cumulative_trapezoid(dt, x):
    return scipy.integrate.cumulative_trapezoid(x.flatten(), dx=dt, initial=0.0)

def rot_y(deg):
    t = deg/180*np.pi
    r = np.array([[np.cos(t),0,np.sin(t)],
                  [0,1,0],
                  [-np.sin(t),0,np.cos(t)]])
    return r

def rot_x(deg):
    t = deg/180*np.pi
    r = np.array([[1,0,0],
                  [0,np.cos(t),-np.sin(t)],
                  [0,np.sin(t), np.cos(t)]])
    return r

def rot_z(deg):
    t = deg/180*np.pi
    r = np.array([[np.cos(t),-np.sin(t),0],
                  [np.sin(t), np.cos(t),0],
                  [0,0,1]])
    return r

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

    if corners.shape[0]<4:
        return None
    # upper left -> upper right -> lower right -> lower left
    # corners are in (x,y)
    corners = sort_corners(corners)

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

def get_opening_width(corners):
    width_top = np.abs(corners[0,0]-corners[1,0])
    width_bottom = np.abs(corners[2,0]-corners[3,0])

    return (width_top+width_bottom)/2

def find_phi(corners_0, corners_t):
    """find phi from the shift in x and y pixel positions
    """
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
        dx,dy = (corners_t[i,:]-corners_0[i,:])/FOCAL_LENGTH
        phi = np.eye(3)
        phi[0,2] = dx
        phi[1,2] = dy
        phi[2,2] = phi_z

        phi_matrices.append(phi)

    return phi_matrices

class PhiConstraintSolver:
    def __init__(self,dt) -> None:
        self.acc_history = None
        self.phi_history = None
        self.A = None
        self.b = None
        self.dt = dt
        self.t_array = []
        self.rot = rot_z(90).dot(rot_y(-90))

    def accumulate(self, acc, phi, curr_t, z_only=False):
        acc = self.rot.dot(acc[:,np.newaxis]).T
        self.t_array.append(curr_t)
        if self.acc_history is None:
            self.acc_history = acc
        else:
            self.acc_history = np.concatenate((self.acc_history,acc),
                                               axis=0)
            
        if self.phi_history is None:
            self.phi_history = phi[:,[2]].T
        else:
            self.phi_history = np.concatenate((self.phi_history,
                                               phi[:,[2]].T),axis=0)
            
        # construct A and b
        t = curr_t
        t2 = t**2
        phix,phiy,phiz = phi[:,2]

        print("==== phi ====")
        print(phix,phiy,phiz)
        print("=============")
        
        dt = np.mean(np.diff(self.t_array))
        if z_only is False:

            A = np.array([[phix,   -t, 0.,  0.,  0.5*t2,       0.,       0.],
                      [phiy,   0., -t,  0.,       0.,  0.5*t2,       0.],
                      [phiz-1, 0., 0.,  -t,       0.,       0.,  0.5*t2]
                    ])
            
            b = [None]*3
            for i in range(3):
                integral = cumulative_trapezoid(x=self.acc_history[:,i],dt=dt)
                double_integral = cumulative_trapezoid(x=integral,dt=dt)
                b[i] = double_integral

            b = np.array(b) # 3 by xx
            self.b = np.reshape(b.T, (-1,1))
            
        else:
            A = np.array([phiz-1, -t, 0.5*t2])[np.newaxis,:]

            integral = cumulative_trapezoid(x=self.acc_history[:,2],dt=self.dt)
            double_integral = cumulative_trapezoid(x=integral,dt=self.dt)
            self.b = double_integral.reshape((-1,1))
        
        if self.A is None:
            self.A = A
        else:
            self.A = np.concatenate((self.A, A),axis=0)

    def solve(self):
        # solve Ax=b
        ans, res, rank, s = np.linalg.lstsq(self.A, self.b, rcond=-1)
        # ans is the array of Z0,X0dot,Y0dot,Z0dot,gx,gy,gz
        return ans