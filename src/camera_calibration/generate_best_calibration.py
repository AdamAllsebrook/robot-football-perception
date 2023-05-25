"""
Generate calibration mtx and dist from a set of saved chessboard images
"""

import random
import cv2
import os
from matplotlib.font_manager import json_load
import numpy as np
import sys
from itertools import combinations

from joblib import dump

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

def sub_lists(my_list):
    subs = []
    for i in range(0, len(my_list)+1):
        temp = [list(x) for x in combinations(my_list, i)]
        if len(temp)>0:
            subs.extend(temp)
    return subs

def get_mean_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    return mean_error / len(objpoints)

def get_obj_img_points(images):
    # Arrays to store object points and image points from all the images
    objpoints = []      # 3D point in real world space
    imgpoints = []      # 2D points in image plane

    for img in images:

        # Convert images to grayscale for chessboard detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        cb_found, corners = cv2.findChessboardCorners(gray, (col, row), None)

        # If found, add object points, image points (after refining them)
        if cb_found:
            objpoints.append(objp)

            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

    return objpoints, imgpoints, gray.shape[::-1]

for index in ['l', 'r']:
    print(index)

    # load images
    image_paths = os.listdir(image_path + '/' + index)
    images = [cv2.imread(image_path + '/' + index + '/' + path)
              for path in image_paths]


    # for sublist in sub_lists(images[:5]):
    #     if len(sublist) == 0: continue
    #     objpoints, imgpoints, l = get_obj_img_points(sublist)
    #     ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    #         objpoints, imgpoints, l, None, None)
    #     mean_error = get_mean_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
    #     print(len(images), mean_error)

    all_objpoints, all_imgpoints, all_l = get_obj_img_points(images)
    all_ret, all_mtx, all_dist, all_rvecs, all_tvecs = cv2.calibrateCamera(
        all_objpoints, all_imgpoints, all_l, None, None)

    best_images = []
    best_indices = []
    for i in range(len(images)):
        errors = {}
        for j in range(len(images)):
            if j not in best_indices:
                copy = best_images.copy()
                copy.append(images[j])
                objpoints, imgpoints, l = get_obj_img_points(copy)
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints, imgpoints, l, None, None)
                mean_error = get_mean_error(all_objpoints, all_imgpoints, all_rvecs, all_tvecs, mtx, dist)
                errors[j] = mean_error

        j = min(errors, key=errors.get)
        best_indices.append(j)
        best_images.append(images[j])

        objpoints, imgpoints, l = get_obj_img_points(best_images)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, l, None, None)
        mean_error = get_mean_error(all_objpoints, all_imgpoints, all_rvecs, all_tvecs, mtx, dist)
        print(i+1, mean_error)
                    
        