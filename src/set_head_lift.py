#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
import numpy as np


rospy.init_node("set_head_lift", anonymous=True)
topic_base_name = 'miro1/'
# publish kinematics (head movement)
pub_kin = rospy.Publisher(
    topic_base_name + 'control/kinematic_joints',
    JointState,
    queue_size=0
)

kin_joints = JointState()
kin_joints.name = ["tilt", "lift", "yaw", "pitch"]
kin_joints.position = [0, np.radians(60), 0, 0]

while True:

    kin_joints.position[1] = np.radians(float(input('enter head lift (30-60):\n')))
    print('setting to ', kin_joints.position)
    pub_kin.publish(kin_joints)