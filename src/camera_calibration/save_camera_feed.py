#!/usr/bin/env python3

"""
Save images from the MiRo camera feed
"""

import cv2
import os

import miro_ros_interface as mri

miro_per = mri.MiRoPerception()

dir_path = os.path.dirname(os.path.realpath(__file__))
save_path = dir_path + '/images'

# create directories if they do not already exist
if not os.path.isdir(save_path):
    os.mkdir(save_path)
if not os.path.isdir(save_path + '/l'):
    os.mkdir(save_path + '/l')
if not os.path.isdir(save_path + '/r'):
    os.mkdir(save_path + '/r')

count_l = len(os.listdir(save_path + '/l'))
count_r = len(os.listdir(save_path + '/r'))

while True:
    command = input().lower()

    # save left camera feed
    if 'l' in command:
        cv2.imwrite(save_path + '/l/%d.jpg' % count_l, miro_per.caml)
        count_l += 1

    # save right camera feed
    if 'r' in command:
        cv2.imwrite(save_path + '/r/%d.jpg' % count_r, miro_per.camr)
        count_r += 1

    # quit
    if 'q' in command:
        break
