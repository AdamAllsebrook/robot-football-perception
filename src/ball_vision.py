#!/usr/bin/env python3
"""
Handles getting the camera feed from the MiRo
Uses detect_ball.py to find the position of the ball
"""
import os
import numpy as np
import cv2
import sys
import miro2 as miro
import time
from joblib import load
import rospy
import actionlib
import cProfile
import pstats

from std_msgs.msg import Header
from sensor_msgs.msg import CompressedImage  # ROS CompressedImage message
from cv_bridge import CvBridge, CvBridgeError  # ROS -> OpenCV converter
from geometry_msgs.msg import Pose2D, Point
from sensor_msgs.msg import JointState

from miro_football_perception.msg import BallPos, BallObservation
from detect_ball import DetectBall
from kalman_filter import KalmanFilter2D
from util.image_to_world import ImageToWorld
from debug.debug_pitch import BALL_L, BALL_R
from miro_football_perception.msg import AddDebugPitchObjectAction, AddDebugPitchObjectGoal
from camera_calibration.camera_config import camera_config


dir_path = os.path.dirname(os.path.realpath(__file__))


class Vision:
    TICK = 1/10  # This is the update interval for the main control loop in secs
    # Set to True to enable debug views of the cameras
    DEBUG = sys.argv[1] == 'true'

    def __init__(self):
        rospy.init_node("detect_ball", anonymous=True)
        rospy.sleep(0.5)

        # Initialise CV Bridge
        self.image_converter = CvBridge()

        # Individual robot name acts as ROS topic prefix
        topic_base_name = rospy.get_namespace()

        # Create two new subscribers to recieve camera images with attached callbapcks
        self.sub_caml = rospy.Subscriber(
            topic_base_name + "sensors/caml/compressed",
            CompressedImage,
            self.callback_caml,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.sub_camr = rospy.Subscriber(
            topic_base_name + "sensors/camr/compressed",
            CompressedImage,
            self.callback_camr,
            queue_size=1,
            tcp_nodelay=True,
        )

        # get changes in kinematics
        self.sub_log = rospy.Subscriber(
            topic_base_name + "sensors/kinematic_joints",
            JointState,
            self.callback_sensors,
            queue_size=5,
            tcp_nodelay=True
        )

        # get the current pose
        self.sub_pose = rospy.Subscriber(
            topic_base_name + "sensors/body_pose",
            Pose2D,
            self.callback_pose,
            queue_size=1
        )

        # create a publisher for the ball position
        self.pub_ball = rospy.Publisher(
            topic_base_name + 'perception/ball_position',
            BallPos,
            queue_size=10
        )

        # create a publisher for ball observations
        self.pub_observe = rospy.Publisher(
            topic_base_name + 'perception/ball_observation',
            BallObservation,
            queue_size=10
        )

        # http://wiki.ros.org/camera_calibration/Tutorials/StereoCalibration
        self.camera_config = load(
            dir_path + '/camera_calibration/camera_config.gz')
        # self.camera_config = camera_config

        # Create handle to store images
        self.input_camera = [None, None]
        # New frame notification
        self.new_frame = [False, False]

        self.kalman_filter = KalmanFilter2D(R=1, Q=10)
        # self.last_observations =  np.empty((0, 2))
        # self.max_last_observations = 10
        self.last_confidences = []
        self.max_last_confidences = 10
        self.image_to_world = ImageToWorld()
        self.ball_detector = DetectBall(self.image_to_world, debug=self.DEBUG)

        self.ball_pos = BallPos(None, Point(0, 0, 0), 0)
        self.last_ball_view = 0

        if self.DEBUG:
            self.debug_pitch_client = actionlib.SimpleActionClient('add_debug_pitch_object', AddDebugPitchObjectAction)

            self.sub_real_ball = rospy.Subscriber(
                'gazebo/ball_position',
                Point,
                self.callback_real_ball_pos,
                queue_size=1
            )
            self.real_ball_pos = None

    def callback_real_ball_pos(self, msg):
        """
        Save the real ball position (in simulation)
        """
        self.real_ball_pos = [msg.x, msg.y]

    def callback_caml(self, ros_image):  # Left camera
        self.callback_cam(ros_image, 0)

    def callback_camr(self, ros_image):  # Right camera
        self.callback_cam(ros_image, 1)

    # callback function executed upon image arrival
    def callback_cam(self, ros_image, index):
        # Silently(-ish) handle corrupted JPEG frames
        try:
            # Convert compressed ROS image to raw CV image
            image = self.image_converter.compressed_imgmsg_to_cv2(
                ros_image, "rgb8")
            # Convert from OpenCV's default BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # undistort image
            mtx, dist = self.camera_config[index]['mtx'], self.camera_config[index]['dist']
            image = cv2.undistort(image, mtx, dist, None)

            # image = cv2.remap(image, self.camera_map[index][0], self.camera_map[index][1], cv2.INTER_LINEAR)

            # Store image as class attribute for further use
            self.input_camera[index] = image
            # Raise the flag: A new frame is available for processing
            self.new_frame[index] = True

        except CvBridgeError as e:
            # Ignore corrupted frames
            pass
        except AttributeError as e:
            pass

    def callback_sensors(self, msg):
        try:
            self.image_to_world.callback_sensors(msg)
        except AttributeError as e:
            pass

    def callback_pose(self, pose):
        try:
            self.image_to_world.callback_pose(pose)
        except AttributeError as e:
            pass

    def publish_observation(self, ball):
        observation = BallObservation(
            Header(None, rospy.Time.now(), None),
            Point(
                np.float32(ball.position[0]),
                np.float32(ball.position[1]),
                0
            ),
            Point(
                np.float32(ball.image_position[0]),
                np.float32(ball.image_position[1]),
                0
            ),
            np.float32(ball.radius),
            ['l', 'r'].index(ball.stream_index),
            Point(
                np.float32(ball.camera_position[0]),
                np.float32(ball.camera_position[1]),
                np.float32(ball.camera_position[2]),
            )
        )
        self.pub_observe.publish(observation)

        # self.last_observations = np.append(self.last_observations, np.array([ball.position]), axis=0)
        # if self.last_observations.shape[0] > self.max_last_observations:
        #     self.last_observations = np.delete(self.last_observations, 0, axis=0)

    def should_reset_kf(self):#, n_std=5):
        # if self.last_observations.shape[0] < self.max_last_observations:
        #     return False
        # mean_x = np.mean(self.last_observations[:,0])
        # std_x = np.std(self.last_observations[:,0])
        # lower_x = mean_x - n_std * std_x
        # upper_x = mean_x + n_std * std_x
        # mean_y = np.mean(self.last_observations[:,1])
        # std_y = np.std(self.last_observations[:,1])
        # lower_y = mean_y - n_std * std_y
        # upper_y = mean_y + n_std * std_y

        # pos = self.kalman_filter.get()
        # return not lower_x < pos[0] < upper_x or not lower_y < pos[1] < upper_y

        # print('mean: ', np.mean(self.last_confidences))
        return np.mean(self.last_confidences) < 0.6

    # once one of the frames has been updated, look the the new ball position
    def detect_ball(self):
        self.kalman_filter.predict(self.TICK)
        if sum(self.new_frame) > 0 and self.input_camera[0] is not None and self.input_camera[1] is not None:
            self.new_frame = [False, False]
            balls = self.ball_detector.detect_balls(
                self.input_camera[0], self.input_camera[1], tick=self.TICK)
                
            ball_pos = None
            if len(balls) > 0:
                self.last_ball_view = time.time()
                for ball in balls: 
                    self.kalman_filter.update(ball.position)
                    self.publish_observation(ball)
                
            mahalanobis = self.kalman_filter.get_mahalanobis()
            if mahalanobis == 0:
                confidence = 0
            elif self.last_ball_view > self.TICK:
                confidence = 1 - np.tanh(time.time() - self.last_ball_view)
            else:
                confidence = 1 - mahalanobis
                self.last_confidences.append(confidence)
                if len(self.last_confidences) > self.max_last_confidences:
                    self.last_confidences.pop(0)

            ball_pos = self.kalman_filter.get()     
            self.ball_pos = BallPos(
                Header(None, rospy.Time.now(), None),
                Point(ball_pos[0], ball_pos[1], 0),
                confidence
            )
            self.pub_ball.publish(self.ball_pos)

            # if self.should_reset_kf():
            #     self.last_confidences = []
            #     self.kalman_filter.reset()

    # main control loop
    def loop(self):
        while not rospy.core.is_shutdown():
            self.detect_ball()
            rospy.sleep(self.TICK)


def kill():
    # record profiling
    profiler.disable()
    stats = pstats.Stats(profiler).strip_dirs()
    stats.dump_stats(dir_path + '/ball_vision.prof')


# This condition fires when the script is called directly
if __name__ == "__main__":
    main = Vision()  # Instantiate class

    profiler = cProfile.Profile()
    profiler.enable()
    rospy.on_shutdown(kill)
    main.loop()
