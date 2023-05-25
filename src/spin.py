#!/usr/bin/env python3

import os
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import TwistStamped  # ROS cmd_vel (velocity control) message
from miro2.lib import wheel_speed2cmd_vel  # Python 3
from sensor_msgs.msg import JointState  # ROS joints state message

class Publisher:
    
    def __init__(self):
        rospy.init_node('publisher_node', anonymous=True)
        self.rate = rospy.Rate(10) # hz
        # topic_base_name = rospy.get_namespace()
        topic_base_name = '/miro1/'
                
        self.ctrl_c = False
        rospy.on_shutdown(self.shutdownhook) 
        # Create a new publisher to send velocity commands to the robot
        self.vel_pub = rospy.Publisher(
            topic_base_name + "control/cmd_vel", TwistStamped, queue_size=0
        )

        # Create a new publisher to move the robot head
        self.pub_kin = rospy.Publisher(
            topic_base_name + "control/kinematic_joints", JointState, queue_size=0
        )
        self.kin_joints = JointState()  # Prepare the empty message
        self.kin_joints.name = ["tilt", "lift", "yaw", "pitch"]
        self.kin_joints.position = [0.0, 0.0, 0.0, 0.0]
        
        rospy.loginfo("publisher node is active...")

    def shutdownhook(self):
        self.shutdown_function()
        self.ctrl_c = True

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
        self.vel_pub.publish(msg_cmd_vel)

    def shutdown_function(self):
        print("stopping publisher node at: {}".format(rospy.get_time()))

    def main_loop(self):
        while not self.ctrl_c:
            publisher_message = "rospy time is: {}".format(rospy.get_time())
            self.drive(0, 0.1)
            # self.kin_joints.position[1] = (self.kin_joints.position[1] + 0.02 - 0.5) % 0.5 + 0.5
            # self.pub_kin.publish(self.kin_joints)
            self.rate.sleep()

if __name__ == '__main__':
    publisher_instance = Publisher()
    try:
        publisher_instance.main_loop()
    except rospy.ROSInterruptException:
        pass