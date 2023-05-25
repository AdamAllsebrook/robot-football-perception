#!/usr/bin/env python3

"""
Run through a set of pre-defined scenarios for system evaluation
"""

import rospy
import numpy as np
import actionlib
import sys

from tf.transformations import quaternion_about_axis
from miro_football_perception.msg import ApplyForceToBallAction, ApplyForceToBallGoal, \
    SetBallPositionAction, SetBallPositionGoal, \
    SetMiRoPoseAction, SetMiRoPoseGoal, \
    Tag
from geometry_msgs.msg import Pose, Quaternion
from std_msgs.msg import Header


loop = int(sys.argv[1])


def do_scenario(position, angle, power, clone_position=[20,0,0], duration=0.1):
    # set ball position
    goal = SetBallPositionGoal()
    goal.x = position[0]
    goal.y = position[1]
    set_position_client.send_goal(goal)
    set_position_client.wait_for_result()

    # set MiRo pose
    goal = SetMiRoPoseGoal()
    goal.name = 'miro1'
    pose = Pose()
    pose.position.x = 0
    pose.position.y = 0
    theta = np.arctan2(position[1], position[0])
    (x, y, z, w) = quaternion_about_axis(theta, (0, 0, 1))
    pose.orientation = Quaternion(x, y, z, w)
    goal.pose = pose
    set_miro_pose_client.send_goal(goal)
    set_miro_pose_client.wait_for_result()

    # set other MiRo pose
    goal = SetMiRoPoseGoal()
    goal.name = 'miro clone'
    pose = Pose()
    pose.position.x = clone_position[0]
    pose.position.y = clone_position[1]
    (x, y, z, w) = quaternion_about_axis(clone_position[2], (0, 0, 1))
    pose.orientation = Quaternion(x, y, z, w)
    goal.pose = pose
    set_miro_pose_client.send_goal(goal)
    set_miro_pose_client.wait_for_result()

    rospy.sleep(1)

    # apply force to ball
    goal = ApplyForceToBallGoal()
    goal.angle = np.float32(angle)
    goal.power = np.float32(power)
    goal.duration = np.float32(duration)
    apply_force_client.send_goal(goal)
    apply_force_client.wait_for_result()


def tag(n):
    return Tag(
        Header(None, rospy.Time.now(), None),
        n
    )


if __name__ == '__main__':
    rospy.init_node('scenarios')
    apply_force_client = actionlib.SimpleActionClient('gazebo/apply_force_to_ball', ApplyForceToBallAction)
    set_position_client = actionlib.SimpleActionClient('gazebo/set_ball_position', SetBallPositionAction)
    set_miro_pose_client = actionlib.SimpleActionClient('gazebo/set_miro_pose', SetMiRoPoseAction)
    apply_force_client.wait_for_server()
    set_position_client.wait_for_server()
    set_miro_pose_client.wait_for_server()
    pub_tag = rospy.Publisher(
        '/gazebo/scenario_tag',
        Tag,
        queue_size=1
    )
    rospy.sleep(1)

    for i in range(loop):

        pub_tag.publish(tag(0))
        do_scenario([1, -1], np.radians(0), 0.3)
        pub_tag.publish(tag(1))

        rospy.sleep(4)
        
        pub_tag.publish(tag(0))
        do_scenario([0, 1], np.radians(105), 0.2)
        pub_tag.publish(tag(2))

        rospy.sleep(4)

        pub_tag.publish(tag(0))
        do_scenario([1.5, 0.5], np.radians(240), 0.2)
        pub_tag.publish(tag(3))

        rospy.sleep(4)

        pub_tag.publish(tag(0))
        do_scenario([1, -1], np.radians(0), 0.3, clone_position=[1.5, 0.2, 0.5])
        pub_tag.publish(tag(4))

        rospy.sleep(4)
        
        pub_tag.publish(tag(0))
        do_scenario([0, 1], np.radians(105), 0.2, clone_position=[0.5, 0.5, -1.5])
        pub_tag.publish(tag(5))

        rospy.sleep(4)

        pub_tag.publish(tag(0))
        do_scenario([1.5, 0.5], np.radians(240), 0.2, clone_position=[0.7, -1, 2])
        pub_tag.publish(tag(6))

        rospy.sleep(4)

        pub_tag.publish(tag(0))
