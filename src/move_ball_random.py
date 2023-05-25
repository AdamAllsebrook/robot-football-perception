#!/usr/bin/env python3

import rospy
import os
import numpy as np

from gazebo_msgs.srv import SpawnModel, ApplyBodyWrench
from geometry_msgs.msg import *


dir_path = os.path.dirname(os.path.realpath(__file__))


def main():
    # TODO stop using globals
    global delete_model
    global pub_ball

    print("Waiting for gazebo services...")
    rospy.init_node("test_env")
    rospy.wait_for_service("gazebo/delete_model")
    rospy.wait_for_service("gazebo/spawn_sdf_model")
    rospy.wait_for_service('/gazebo/apply_body_wrench')
    print("initialised.")
    # delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
    spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
    force = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)

    model_path = os.path.join(os.path.expanduser(
        '~'), 'mdk', 'sim', 'models', 'robocup_3Dsim_ball', 'model.sdf')
    with open(model_path, "r") as f:
        ball_xml = f.read()

    # orient = Quaternion(tf.transformations.quaternion_from_euler(0,0,0))
    orient = Quaternion(0, 0, 0, 1)

    print("Spawning ball")
    ball_pose = Pose(Point(x=1, y=0.01, z=0),   orient)
    spawn_model("ball", ball_xml, "", ball_pose, "world")

    while not rospy.is_shutdown():
        wrench = Wrench()
        angle = np.random.random() * 2 * np.pi
        power = 0.2
        wrench.force.x = np.sin(angle) * power
        wrench.force.y = np.cos(angle) * power
        # print('applying force x %.3f y %.3f' %
        #       (wrench.force.x, wrench.force.y))
        response = force(body_name='ball::soccer_ball_link',
                         wrench=wrench, duration=rospy.Duration(0.1))
        rospy.sleep(5)


if __name__ == '__main__':
    # profiler = cProfile.Profile()
    # profiler.enable()
    # rospy.on_shutdown(end)
    main()
