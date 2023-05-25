#!/usr/bin/env python3

"""
Read data about each observation the system makes, and save to a file

Data is stored in a 1D array:
    miro pose x,y,theta
    real pos x,y
    world pos x,y 
    image pos x,y 
    radius
    stream index
    timestamp (secs) 
    frame number
    real vel x,y
    percept vel x,y
    camera pos x,y,z
    head lift
    tag
"""

import rospy
import os
import numpy as np
from joblib import load, dump

from geometry_msgs.msg import Point, Pose2D
from sensor_msgs.msg import JointState

from miro_football_perception.msg import BallObservation, BallPos, Tag


dir_path = os.path.dirname(os.path.realpath(__file__))


class ReadObservations:
    DATA_PATH = dir_path + '/observation_data.gz'

    def __init__(self):
        rospy.init_node('read_observations')
        topic_base_name = rospy.get_namespace()

        self.sub_observe = rospy.Subscriber(
            topic_base_name + 'perception/ball_observation',
            BallObservation,
            self.callback_observation,
            queue_size=10
        )

        self.sub_percept_vel = rospy.Subscriber(
            topic_base_name + 'perception/ball_velocity',
            Point,
            self.callback_percept_vel,
            queue_size=10
        )

        self.sub_real_pos = rospy.Subscriber(
            '/gazebo/ball_position',
            Point,
            self.callback_real_pos,
            queue_size=10
        )

        self.sub_real_pos = rospy.Subscriber(
            '/gazebo/ball_velocity',
            Point,
            self.callback_real_vel,
            queue_size=10
        )

        self.sub_percept_ball = rospy.Subscriber(
            topic_base_name + 'perception/ball_position',
            BallPos,
            self.callback_percept_ball_pos,
            queue_size=1
        )

        self.sub_pose = rospy.Subscriber(
            topic_base_name + "sensors/body_pose",
            Pose2D,
            self.callback_pose,
            queue_size=1
        )

        self.sub_kin_joints = rospy.Subscriber(
            topic_base_name + 'sensors/kinematic_joints',
            JointState,
            self.callback_kin_joints,
            queue_size=1
        )

        self.sub_tag = rospy.Subscriber(
            '/gazebo/scenario_tag',
            Tag,
            self.callback_tag,
            queue_size=1
        )

        self.real_pos = [np.nan, np.nan]
        self.miro_pose = [np.nan, np.nan, np.nan]
        self.real_vel = [np.nan, np.nan]
        self.percept_vel = [np.nan, np.nan]
        self.head_lift = np.nan
        self.tag = 0

        self.observations = np.empty((0, 22))
        self.frame = 0

    def callback_observation(self, msg):
        """
        When a ball is spotted in either camera
        """
        observation = np.array([[
            self.miro_pose[0],
            self.miro_pose[1],
            self.miro_pose[2],
            self.real_pos[0],
            self.real_pos[1],
            msg.world_position.x,
            msg.world_position.y,
            msg.image_position.x,
            msg.image_position.y,
            msg.radius,
            msg.stream_index,
            msg.header.stamp.to_sec(),
            self.frame,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            msg.camera_position.x,
            msg.camera_position.y,
            msg.camera_position.z,
            self.head_lift,
            self.tag
        ]])
        self.observations = np.append(self.observations, observation, axis=0)

    def callback_real_pos(self, msg):
        self.real_pos = [msg.x, msg.y]

    def callback_percept_ball_pos(self, msg):
        """
        Estimation of the ball position using data from both cameras
        """
        if msg.confidence > 0.1:
            observation = np.array([[
                self.miro_pose[0],
                self.miro_pose[1],
                self.miro_pose[2],
                self.real_pos[0],
                self.real_pos[1],
                msg.position.x,
                msg.position.y,
                np.nan,
                np.nan,
                np.nan,
                2,
                msg.header.stamp.to_sec(),
                self.frame,
                self.real_vel[0],
                self.real_vel[1],
                self.percept_vel[0],
                self.percept_vel[1],
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                self.tag
            ]])
            self.observations = np.append(
                self.observations, observation, axis=0)
        
        self.frame += 1

    def callback_pose(self, msg):
        self.miro_pose = [msg.x, msg.y, msg.theta]

    def callback_real_vel(self, msg):
        self.real_vel = [msg.x, msg.y]

    def callback_percept_vel(self, msg):
        self.percept_vel = [msg.x, msg.y]

    def callback_kin_joints(self, msg):
        self.head_lift = msg.position[1]

    def callback_tag(self, msg):
        for i in range(3):
            self.blank_observation(i, msg.header.stamp.to_sec())
        self.tag = msg.tag

    def blank_observation(self, stream_index, time=np.nan):
        observation = np.array([[
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            stream_index,
            time,
            self.frame,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            self.tag
        ]])
        self.observations = np.append(
            self.observations, observation, axis=0)

    def loop(self):
        rospy.on_shutdown(self.save_data)
        while not rospy.is_shutdown():
            rospy.sleep(1/20)

    def save_data(self):
        try:
            data = load(self.DATA_PATH)
        except FileNotFoundError:
            data = []

        data = data + [self.observations]
        dump(data, self.DATA_PATH)


if __name__ == '__main__':
    main = ReadObservations()
    main.loop()
