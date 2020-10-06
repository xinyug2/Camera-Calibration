import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*9,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.jpg')

tot_error = 0
for fname in images:
    img = cv2.imread(fname)
    
    # resize
#    width = 320
#    height = 240
#    dim = (width, height)
#    img = cv2.resize(cv2.imread(fname), dim, interpolation = cv2.INTER_AREA)
#    cv2.imwrite(fname + '_resized.jpg', img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,9),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        # deep copy image 
        #img2 = cv2.drawChessboardCorners(img, (7,9), corners2, ret)
        
        #cv2.imwrite('border' + fname + '.jpg ', img2)
for fname in images:
    img = cv2.imread(fname)
    print(fname)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    

    mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
    dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
    
    x,y,w,h = roi
        #dst = dst[y:y+h, x:x+w]
        # undistort
        #dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
#        x,y,w,h = roi
#        dst = dst[y:y+h, x:x+w]

    cv2.imwrite('ud_' + fname + '.png ',dst)
        #i = i + 1
        
#        cv2.imwrite(fname, img)
#        cv2.waitKey(500)

#cv2.destroyAllWindows()
x_vals = []
y_vals = []
z_vals = []
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
#mean_error = 0
imgpoints2 = []
tot_error = 0
for i in range(len(objpoints)):
    img, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#    x_vals.append(tvecs[i, 0])
#    y_vals.append(tvecs[i, 1])
#    z_vals.append(tvecs[i, 2])
    imgpoints2.append(img)
#images2 = glob.glob('*.png')  

#for fname in images2:
#    img = cv2.imread(fname)
#    
#    # resize
##    width = 320
##    height = 240
##    dim = (width, height)
##    img = cv2.resize(cv2.imread(fname), dim, interpolation = cv2.INTER_AREA)
##    cv2.imwrite(fname + '_resized.jpg', img)
#    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
#    # Find the chess board corners
#    ret, corners = cv2.findChessboardCorners(gray, (7,9),None)
#
#    # If found, add object points, image points (after refining them)
#    if ret == True:
#        #objpoints.append(objp)
#
#        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
#        imgpoints2.append(corners2)
    
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random


fig = pyplot.figure()
ax = Axes3D(fig)

#sequence_containing_x_vals = list(range(-50, 50))
#sequence_containing_y_vals = list(range(-50, 50))
#sequence_containing_z_vals = list(range(-50, 50))

#random.shuffle(sequence_containing_x_vals)
#random.shuffle(sequence_containing_y_vals)
#random.shuffle(sequence_containing_z_vals)

ax.scatter(x_vals, y_vals, z_vals)
pyplot.show()
 
for i in range(len(objpoints)):
    error = cv2.norm(imgpoints[i],imgpoints2[i], cv2.NORM_L2)/len(imgpoints2[i])
    tot_error += error

print('total error:', tot_error/len(objpoints))
