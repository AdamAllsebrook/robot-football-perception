#!/usr/bin/env python3

"""
Publish the position of the simulation ball to a topic
"""

import rospy
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import Point


def callback_link_states(msg):
    """
    publish the current position and velocity of the ball
    """
    link_name = 'ball::soccer_ball_link'
    if link_name in msg.name:
        i = msg.name.index(link_name)
        pos = msg.pose[i].position
        point = Point(pos.x, pos.y, pos.z)
        pub_ball.publish(point)
        vel = msg.twist[i].linear
        point = Point(vel.x, vel.y, vel.y)
        pub_ball_vel.publish(point)


if __name__ == '__main__':
    rospy.init_node('publish_sim_ball')

    # get real ball position
    sub_state = rospy.Subscriber(
        '/gazebo/link_states',
        LinkStates,
        callback_link_states,
        queue_size=1,
        tcp_nodelay=True
    )

    # publish real ball position
    pub_ball = rospy.Publisher(
        '/gazebo/ball_position',
        Point,
        queue_size=10
    )
    
    # publish real ball velocity
    pub_ball_vel = rospy.Publisher(
        '/gazebo/ball_velocity',
        Point,
        queue_size=10
    )

    while not rospy.is_shutdown():
        rospy.sleep(0.1)
