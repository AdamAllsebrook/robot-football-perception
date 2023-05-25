#!/usr/bin/env python3

import rospy
import sys
import numpy as np
import time

from geometry_msgs.msg import TwistStamped, Pose2D
from miro_football_perception.msg import BallPos
from sensor_msgs.msg import JointState
from miro2.lib import wheel_speed2cmd_vel

#pitch_shape=(2.6, 3.9)


class LookAtBall:
    CHASE = float(sys.argv[1])
    PITCH_X = 1.95
    PITCH_Y = 1.3

    def __init__(self):
        rospy.init_node("look_at_ball", anonymous=True)
        self.rate = rospy.Rate(20)

        # Individual robot name acts as ROS topic prefix
        topic_base_name = rospy.get_namespace()

        # get changes in ball position
        self.sub_ball_pos = rospy.Subscriber(
            topic_base_name + 'perception/ball_position',
            BallPos,
            self.callback_ball_position
        )

        # get pose
        self.sub_pose = rospy.Subscriber(
            topic_base_name + 'sensors/body_pose',
            Pose2D,
            self.callback_pose
        )

        # publish movement
        self.pub_vel = rospy.Publisher(
            topic_base_name + 'control/cmd_vel',
            TwistStamped,
            queue_size=0
        )

        # publish kinematics (head movement)
        self.pub_kin = rospy.Publisher(
            topic_base_name + 'control/kinematic_joints',
            JointState,
            queue_size=0
        )

        self.ball_pos = None
        self.pose = None

        self.head_lift = np.radians(60)
        # self.head_lift = np.radians(40)
        self.head_left_delta = 1
        self.kin_joints = JointState()
        self.kin_joints.name = ["tilt", "lift", "yaw", "pitch"]
        self.kin_joints.position = [0, self.head_lift, 0, 0]

        self.turn_speed = 0.2

    def callback_ball_position(self, msg):
        self.ball_pos = msg

    def callback_pose(self, msg):
        self.pose = msg

    def drive(self, speed_l=0.1, speed_r=0.1):  # (m/sec, m/sec)
        """
        Wrapper to simplify driving MiRo by converting wheel speeds to cmd_vel
        """
        # Prepare an empty velocity command message
        msg_cmd_vel = TwistStamped()

        # Desired wheel speed (m/sec)
        wheel_speed = [speed_l, speed_r]

        # Convert wheel speed to command velocity (m/sec, Rad/sec)
        (dr, dtheta) = wheel_speed2cmd_vel(wheel_speed)

        # Update the message with the desired speed
        msg_cmd_vel.twist.linear.x = dr
        msg_cmd_vel.twist.angular.z = dtheta

        # Publish message to control/cmd_vel topic
        self.pub_vel.publish(msg_cmd_vel)

    def get_nearest_wall(self, dir):
        distances = []
        # x = pose_x + d * dir_x
        # d = (x - pose_x) / dir_x
        distances.append((self.PITCH_X - self.pose.x) / dir[0])
        distances.append((-self.PITCH_X - self.pose.x) / dir[0])
        distances.append((self.PITCH_Y - self.pose.y) / dir[1])
        distances.append((-self.PITCH_Y - self.pose.y) / dir[1])
        distances = list(filter(lambda x: x > 0, distances))
        return min(distances)

    def loop(self):
        while not rospy.core.is_shutdown():

            # self.head_lift += 0.01 * self.head_left_delta
            # if self.head_lift > np.radians(60) or self.head_lift < np.radians(30):
            #     self.head_left_delta *= -1
            # self.kin_joints.position = [0, np.radians(45 + 15 * np.sin(time.time())), 0, 0]

            self.pub_kin.publish(self.kin_joints)
            speed_l = 0
            speed_r = 0

            if self.pose is not None and self.ball_pos is not None and self.ball_pos.confidence > 0.8:
                # get vector from robot to the ball
                ball_position = np.array(
                    [self.ball_pos.position.x, self.ball_pos.position.y])
                robot_position = np.array([self.pose.x, self.pose.y])
                ball_relative = ball_position - robot_position

                # vector directly to the right of where the robot is facing
                right = np.array(
                    [np.cos(self.pose.theta + np.pi/2), np.sin(self.pose.theta + np.pi/2)])
                left = np.array(
                    [np.cos(self.pose.theta - np.pi/2), np.sin(self.pose.theta - np.pi/2)])

                turn_speed = self.turn_speed * np.dot(right, ball_relative)
                speed_l = -turn_speed
                speed_r = turn_speed

                # l = self.get_nearest_wall(left)
                # r = self.get_nearest_wall(right)
                # speed_l += 0.1/l
                # speed_r += 0.1/r

                if self.CHASE != -1 and np.sqrt(ball_relative[0] ** 2 + ball_relative[1] ** 2) > self.CHASE:
                # and self.pose is not None and (
                #     -self.PITCH_X + self.PITCH_BORDER < self.pose.x < self.PITCH_X - self.PITCH_BORDER and
                #     -self.PITCH_Y + self.PITCH_BORDER < self.pose.y < self.PITCH_Y - self.PITCH_BORDER
                # ):
                    speed_l += 0.3
                    speed_r += 0.3

            else:
                speed_l = self.turn_speed
                speed_r = -self.turn_speed

            self.drive(speed_l, speed_r)

            self.rate.sleep()


if __name__ == '__main__':
    node = LookAtBall()
    node.loop()
