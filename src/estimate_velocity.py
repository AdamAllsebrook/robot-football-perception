#!/usr/bin/env python3

import rospy
import numpy as np
import os

from geometry_msgs.msg import Point

from miro_football_perception.msg import BallPos


class EstimateVelocity:
    TICK = 1/20
    # ewma smoothing factor
    ALPHA = 0.5
    def __init__(self):
        rospy.init_node('estimate_velocity')

        topic_base_name = rospy.get_namespace()

        self.sub_ball_pos = rospy.Subscriber(
            topic_base_name + 'perception/ball_position',
            BallPos,
            self.callback_ball_pos,
            queue_size=10
        )

        self.pub_ball_vel = rospy.Publisher(
            topic_base_name + 'perception/ball_velocity',
            Point,
            queue_size=10
        )

        self.ball_pos = np.empty((0, 2))
        self.max_len = 20

    def reset(self):
        self.ball_pos = np.empty((0, 2))

    def callback_ball_pos(self, msg):
        if msg.confidence > 0.95:

            if self.ball_pos.shape[0] > 0:
                diff = np.sqrt((self.ball_pos[-1][0] - msg.position.x) ** 2 + (self.ball_pos[-1][1] - msg.position.y) ** 2)
                if diff > 0.5:
                    self.reset()

            self.ball_pos = np.append(self.ball_pos, np.array([[msg.position.x, msg.position.y]]), axis=0)
            if self.ball_pos.shape[0] > self.max_len:
                self.ball_pos = np.delete(self.ball_pos, 0, axis=0)

        # if the ball hasnt been seen for too long then reset
        if msg.confidence < 0.2:
            self.reset()

    def estimate_velocity(self):
        if len(self.ball_pos) > 1:
            diff = np.diff(self.ball_pos, axis=0)
            
            vel = ewma(diff, diff.shape[0] - 1, self.ALPHA)

            # theta = np.arctan2(vel[1], vel[0])
            # r = np.sqrt(vel[0] ** 2 + vel[1] ** 2)
            return Point(vel[0] / self.TICK, vel[1] / self.TICK, 0)
        return Point()

    def loop(self):
        while not rospy.is_shutdown():
            self.pub_ball_vel.publish(self.estimate_velocity())
            rospy.sleep(self.TICK)


def ewma(Y, t, alpha):
    if t == 0:
        return Y[0]
    else:
        return alpha * Y[t] + (1 - alpha) * ewma(Y, t-1, alpha)


if __name__ == '__main__':
    main = EstimateVelocity()
    main.loop()