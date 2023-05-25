#!/usr/bin/env python3
"""
ROS action servers for setting the ball position, applying force to the ball and setting the MiRo pose
"""

import rospy
import actionlib
import numpy as np

from gazebo_msgs.srv import ApplyBodyWrench, SetLinkState, SetModelState
from gazebo_msgs.msg import LinkState, ModelState
from geometry_msgs.msg import Wrench, Pose

from miro_football_perception.msg import ApplyForceToBallAction, SetBallPositionAction, SetMiRoPoseAction


class SetSimObject:
    BALL_NAME = 'ball::soccer_ball_link'

    def __init__(self):
        rospy.init_node('move_ball_server')

        rospy.wait_for_service('/gazebo/apply_body_wrench')
        rospy.wait_for_service('/gazebo/set_model_state')
        rospy.wait_for_service('/gazebo/set_link_state')

        # setup action servers
        self.force_server = actionlib.SimpleActionServer(
            'gazebo/apply_force_to_ball', ApplyForceToBallAction, self.apply_force_action, False)
        self.force_server.start()

        self.pos_server = actionlib.SimpleActionServer(
            'gazebo/set_ball_position', SetBallPositionAction, self.set_ball_position, False)
        self.pos_server.start()

        self.miro_server = actionlib.SimpleActionServer(
            'gazebo/set_miro_pose', SetMiRoPoseAction, self.set_miro_pose, False)
        self.miro_server.start()

        # create proxies for gazebo actions
        self.force = rospy.ServiceProxy(
            '/gazebo/apply_body_wrench', ApplyBodyWrench)
        self.set_link = rospy.ServiceProxy(
            '/gazebo/set_link_state', SetLinkState)
        self.set_model = rospy.ServiceProxy(
            '/gazebo/set_model_state', SetModelState)

    def apply_force_action(self, goal):
        """
        Apply a force to the ball
        """
        wrench = Wrench()
        wrench.force.x = np.sin(goal.angle) * goal.power
        wrench.force.y = np.cos(goal.angle) * goal.power
        res = self.force(body_name=self.BALL_NAME,
                         wrench=wrench, duration=rospy.Duration(goal.duration))
        self.force_server.set_succeeded()

    def set_ball_position(self, goal):
        """
        Set the world position of the ball
        """
        link_state = LinkState()
        link_state.link_name = self.BALL_NAME
        link_state.pose.position.x = goal.x
        link_state.pose.position.y = goal.y
        res = self.set_link(link_state)
        self.pos_server.set_succeeded()

    def set_miro_pose(self, goal):
        """
        Set a MiRos pose
        """
        model_state = ModelState()
        model_state.model_name = goal.name
        model_state.pose = goal.pose
        res = self.set_model(model_state)
        self.miro_server.set_succeeded()

    def loop(self):
        while not rospy.is_shutdown():
            rospy.sleep(1/20)


if __name__ == '__main__':
    main = SetSimObject()
    main.loop()
