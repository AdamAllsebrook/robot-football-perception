#!/usr/bin/env python3

"""
Display the MiRos camera feed, with the chessboard corners overlayed
"""

import cv2
import numpy as np

import rospy
import miro_ros_interface as mri

miro_per = mri.MiRoPerception()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

col = 7
row = 6

print(0)

def draw_chessboard_corners(cam):
    """
    Draw chessboard corners onto an image
    """
    gray = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    cb_found, corners = cv2.findChessboardCorners(gray, (col, row), None)

    # If found, add object points, image points (after refining them)
    if cb_found:

        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Draw and display the corners
        cv2.drawChessboardCorners(cam, (col, row), corners, cb_found)


if miro_per.caml is not None and miro_per.camr is not None:

    while not rospy.core.is_shutdown():
        cam_l = np.array(miro_per.caml)
        cam_r = np.array(miro_per.camr)

        draw_chessboard_corners(cam_l)
        draw_chessboard_corners(cam_r)

        concat = np.concatenate((cam_l, cam_r), axis=1)
        cv2.imshow('picture', concat)
        cv2.waitKey(1)

else:
    print('Error getting camera feed')