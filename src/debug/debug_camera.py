#!/usr/bin/python3
"""
Show the camera feed, ball positions and masked images
"""
import cv2
import numpy as np

import rospy
import actionlib
from cv_bridge import CvBridge, CvBridgeError  # ROS -> OpenCV converter

from miro_football_perception.msg import AddDebugCameraAction


class DebugCamera:

    def __init__(self, title='display'):
        rospy.init_node('debug_camera_server')
        self.server = actionlib.SimpleActionServer('add_debug_camera', AddDebugCameraAction, self.set_image, False)
        self.server.start()

        # Initialise CV Bridge
        self.image_converter = CvBridge()

        self.title = title
        self.images = {}

    def is_full(self):
        return len(self.images) == 2 and len(self.images[0]) == 2 and len(self.images[1]) == 2

    def set_image(self, goal):
        """
        Set a new image to be displayed
        """
        try:
            image = self.image_converter.imgmsg_to_cv2(goal.ros_image, "rgb8")
            if goal.y not in self.images:
                self.images[goal.y] = {}
            self.images[goal.y][goal.x] = image
        except CvBridgeError as e:
            pass
        self.server.set_succeeded()

    def show_images(self):
        """
        Display all images
        """
        if self.is_full():
            rows = []
            for y in range(len(self.images)):
                rows.append(np.concatenate([self.images[y][key] for key in sorted(self.images[y].keys())], axis=0))

            images_concat = np.concatenate(rows, axis=1)
            cv2.imshow(self.title, images_concat)
            cv2.waitKey(1)

    def loop(self):
        while not rospy.core.is_shutdown():
            self.show_images()
            rospy.sleep(0.1)

if __name__ == '__main__':
    main = DebugCamera()
    main.loop()