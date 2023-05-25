#!/usr/bin/env python3
"""

"""
import rospy
import os

from geometry_msgs.msg import Point

from miro_football_perception.srv import (BallTrajectoryPrediction, BallTrajectoryPredictionResponse, 
    BatchBallTrajectoryPrediction, BatchBallTrajectoryPredictionResponse)
from miro_football_perception.msg import BallPos


class BallTrajectory:

    def __init__(self):
        rospy.init_node('ball_trajectory')

        topic_base_name = rospy.get_namespace()

        self.sub_pos = rospy.Subscriber(
            topic_base_name + 'perception/ball_position',
            BallPos,
            self.callback_pos,
        )
        self.sub_vel = rospy.Subscriber(
            topic_base_name + 'perception/ball_velocity',
            Point,
            self.callback_vel,
        )

        self.service = rospy.Service(
            'get_ball_trajectory', BallTrajectoryPrediction, self.callback_trajectory)
        self.service = rospy.Service(
            'batch_get_ball_trajectory', BatchBallTrajectoryPrediction, self.callback_batch_trajectory)

        self.ball_pos = [0, 0]
        self.ball_pos_confidence = 0
        self.ball_vel = [0, 0]

    def callback_pos(self, msg):
        self.ball_pos = [msg.position.x, msg.position.y]
        self.ball_pos_confidence = msg.confidence

    def callback_vel(self, msg):
        self.ball_vel = [msg.x, msg.y]

    def callback_trajectory(self, req):
        res = BallTrajectoryPredictionResponse()
        res.x = self.ball_pos[0] + self.ball_vel[0] * req.t
        res.y = self.ball_pos[1] + self.ball_vel[1] * req.t
        res.confidence = self.ball_pos_confidence ** (req.t * 100)
        return res

    def callback_batch_trajectory(self, req):
        res = BatchBallTrajectoryPredictionResponse()
        res.x = []
        res.y = []
        res.confidence = []

        for t in req.t:
            res.x.append(self.ball_pos[0] + self.ball_vel[0] * t)
            res.y.append(self.ball_pos[1] + self.ball_vel[1] * t)
            res.confidence.append(self.ball_pos_confidence ** (t * 100))
        return res

    def loop(self):
        while not rospy.core.is_shutdown():
            rospy.sleep(1 / 20)


if __name__ == '__main__':
    main = BallTrajectory()
    main.loop()
