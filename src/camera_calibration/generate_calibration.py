"""
Generate calibration mtx and dist from a set of saved chessboard images
"""

import random
import cv2
import os
import numpy as np
import sys

from joblib import dump

test = len(sys.argv) > 1    

dir_path = os.path.dirname(os.path.realpath(__file__))
image_path = dir_path + '/images'

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Chessboard internal dimensions
col = int(7)
row = int(6)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0),..., (6,5,0)
objp = np.zeros((row * col, 3), np.float32)
objp[:, :2] = np.mgrid[0:col, 0:row].T.reshape(-1, 2)

# count how many images contain a chessboard
success_count = 0
count = 0

calibration = []  # [left_calibration, right_calibration]

e = {}


def get_mean_error(objpoints, imgpoints):
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    return mean_error / len(objpoints)

for index in ['l', 'r']:
    print(index)
    e[index] = ''

    # load images
    image_paths = os.listdir(image_path + '/' + index)
    images = [cv2.imread(image_path + '/' + index + '/' + path)
              for path in image_paths]
    # random.shuffle(images)

    # Arrays to store object points and image points from all the images
    objpoints = []      # 3D point in real world space
    imgpoints = []      # 2D points in image plane

    for i, img in enumerate(images):
        count += 1

        # Convert images to grayscale for chessboard detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        cb_found, corners = cv2.findChessboardCorners(gray, (col, row), None)

        # If found, add object points, image points (after refining them)
        if cb_found:
            success_count += 1
            objpoints.append(objp)

            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
        
        if test:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None)
            print(index, i+1, get_mean_error(objpoints, imgpoints))
            e[index] += '(%d, %.3f) ' % (i+1, get_mean_error(objpoints, imgpoints))

    # Return the camera matrix, distortion coefficients, rotation and translation vectors
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    print(mtx, dist)
    calibration.append({
        'mtx': mtx,
        'dist': dist
    })

    # calculate error
    mean_error = get_mean_error(objpoints, imgpoints)
    print("mean error: {}".format(mean_error))
    print()
    
print(e['l'])
print(e['r'])

print('%d/%d images contained a chessboard' % (success_count, count))
# save calibration
if not test:
    dump(calibration, dir_path + '/camera_config.gz')
