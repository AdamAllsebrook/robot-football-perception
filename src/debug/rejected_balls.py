#!/usr/bin/python3

"""
Node for displaying rejected balls
"""

import rospy
import actionlib

from debug_balls import DebugBalls
from miro_football_perception.msg import AddDebugBallAction

def execute(goal):
    debug_balls.add_image(goal.ros_image)
    server.set_succeeded()

if __name__ == '__main__':
    debug_balls = DebugBalls(cols=5, rows=8, title='reject')
    rospy.init_node("debug_rejected_balls_server", anonymous=True)
    server = actionlib.SimpleActionServer('add_debug_ball_rejected', AddDebugBallAction, execute, False)
    server.start()
    debug_balls.loop()