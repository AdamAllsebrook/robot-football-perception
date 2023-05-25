#!/usr/bin/python3
"""
Display a list of ball images
"""

import cv2
import numpy as np
import sys

import rospy
import actionlib
from cv_bridge import CvBridge, CvBridgeError  # ROS -> OpenCV 

from miro_football_perception.msg import AddDebugBallAction


class DebugBalls:
    def __init__(self, img_w=50, rows=8, cols=12, scale=2, title=''):
        self.images = []
        self.circles = []
        self.img_w = img_w
        self.rows = rows
        self.cols = cols
        self.scale = scale
        self.title = title
        self.index = 0

        # Initialise CV Bridge
        self.image_converter = CvBridge()

        rospy.init_node("debug_%sed_balls_server" % title, anonymous=True)
        self.server = actionlib.SimpleActionServer('add_debug_ball_%sed' % title, AddDebugBallAction, self.execute, False)
        self.server.start()

    def add_image(self, ros_image):
        """
        Add a new image to the list
        """
        try:
            image = self.image_converter.imgmsg_to_cv2(ros_image, "rgb8")

            # crop/ pad the image to have dimension img_w x img_w 
            base = np.zeros((self.img_w, self.img_w, 3),dtype=np.uint8)
            h, w = image.shape[:2]
            if not (w > 0 and h > 0):
                return
            w, h = min(w, self.img_w), min(h, self.img_w)
            base[0:h, 0:w] = image[0:h, 0:w]
            # add the image
            # on overflow start from the beginning again
            self.index = (self.index + 1) % (self.rows * self.cols)
            if self.index >= len(self.images):
                self.images.append(base)
            else:
                self.images[self.index] = base

        except CvBridgeError as e:
            pass

    def draw(self):
        """
        Draw images on a grid
        """
        cols = []
        for i in range(0, len(self.images), self.rows):
            base = np.zeros((self.rows * self.img_w, self.img_w, 3),dtype=np.uint8)
            concat = np.concatenate(self.images[i:i+self.rows], axis=0)
            base[0:concat.shape[0], 0:concat.shape[1]] = concat
            cols.append(base)
        if len(cols) > 0:
            images_concat = np.concatenate(cols, axis=1)
            width = int(images_concat.shape[1] * self.scale)
            height = int(images_concat.shape[0] * self.scale)
            dim = (width, height)
            
            # resize image
            resized = cv2.resize(images_concat, dim, interpolation = cv2.INTER_AREA)

            cv2.imshow(self.title, resized)
            cv2.waitKey(1)

    def execute(self, goal):
        
        self.add_image(goal.ros_image)
        self.server.set_succeeded()
    
    def loop(self):
        while not rospy.core.is_shutdown():
            self.draw()
            rospy.sleep(0.1)


if __name__ == '__main__':
    main = DebugBalls(cols=7, rows=8, title=sys.argv[1])
    main.loop()