#!/usr/bin/env python3
"""
Show the position of observed objects on the pitch
"""

import numpy as np
import cv2
import os

import rospy
import actionlib
from geometry_msgs.msg import Pose2D, Point

from miro_football_perception.msg import AddDebugPitchObjectAction, AddDebugPitchBoxAction, BallPos, BallObservation
from miro_football_perception.srv import BatchBallTrajectoryPrediction

BALL_L = 0
BALL_R = 1
MIRO = 2
REAL_BALL = 3
BALL_PERCEPT = 4
FUTURE_BALL = 5
COLOURS = [(255, 255, 0), (255, 0, 0),
           (255, 255, 255), (0, 255, 0), 
           (0, 0, 255), (255, 255, 255)]
RADII = [4, 4, 8, 4, 4, 4]


class DebugPitch:
    def __init__(self, pitch_shape=(2.6, 3.9), pitch_centre=(0, 0), image_scale=150):
        rospy.init_node('debug_pitch_server')

        self.pitch_shape = pitch_shape
        self.image_scale = image_scale
        self.image_shape = np.append(
            np.array(pitch_shape) * image_scale, 3).astype(int)
        self.pitch_translate = np.array(
            pitch_shape) / 2 - np.array(pitch_centre)
        self.objects = []
        self.boxes = []

        self.server = actionlib.SimpleActionServer(
            'add_debug_pitch_object', AddDebugPitchObjectAction, self.add_object_action, False)
        self.server.start()

        self.box_server = actionlib.SimpleActionServer(
            'add_debug_pitch_box', AddDebugPitchBoxAction, self.add_box_action, False
        )
        self.box_server.start()

        # Individual robot name acts as ROS topic prefix
        topic_base_name = rospy.get_namespace()

        rospy.wait_for_service('batch_get_ball_trajectory')
        self.predict_trajectory = rospy.ServiceProxy('batch_get_ball_trajectory', BatchBallTrajectoryPrediction)

        self.sub_percept_ball = rospy.Subscriber(
            topic_base_name + 'perception/ball_position',
            BallPos,
            self.callback_percept_ball_pos,
            queue_size=1
        )

        self.sub_observation = rospy.Subscriber(
            topic_base_name + 'perception/ball_observation',
            BallObservation,
            self.callback_observation,
            queue_size=2
        )

        self.sub_real_ball = rospy.Subscriber(
            '/gazebo/ball_position',
            Point,
            self.callback_real_ball_pos,
            queue_size=1
        )

        # get the current pose
        self.sub_pose = rospy.Subscriber(
            topic_base_name + "sensors/body_pose",
            Pose2D,
            self.callback_pose,
            queue_size=1
        )

        self.sub_vel = rospy.Subscriber(
            topic_base_name + 'perception/ball_velocity',
            Point,
            self.callback_ball_vel,
            queue_size=1
        )
        self.ball_vel = [0, 0]

    def callback_percept_ball_pos(self, msg):
        """
        Save the perceived ball position
        """
        self.add_object(msg.position.x, msg.position.y, BALL_PERCEPT,
                        {'theta': np.arctan2(self.ball_vel[1], self.ball_vel[0]),
                        'r': np.sqrt(self.ball_vel[0] ** 2 + self.ball_vel[1] ** 2)})

    def callback_observation(self, msg):
        if msg.stream_index == 0:
            ball = BALL_L
        else:
            ball = BALL_R
        self.add_object(msg.world_position.x, msg.world_position.y, ball)

    def callback_ball_vel(self, msg):
        self.ball_vel = [msg.x, msg.y]

    def callback_real_ball_pos(self, msg):
        """
        Save the real ball position (in simulation)
        """
        self.add_object(msg.x, msg.y, REAL_BALL)

    def callback_pose(self, msg):
        """
        Save the current MiRo position
        """
        self.add_object(msg.x, msg.y, MIRO, {'theta':msg.theta})

    def add_object_action(self, goal):
        """
        Add an object to be displayed on the pitch
        """
        self.add_object(goal.x, goal.y, goal.type)
        self.server.set_succeeded()

    def world_to_display_space(self, x, y):
        position = np.array([-y, x])
        position += self.pitch_translate
        position *= self.image_scale

        out_of_pitch = False

        if position[0] < 0 or position[0] > self.image_shape[0] or position[1] < 0 or position[1] > self.image_shape[1]:
            out_of_pitch = True
            position[0] = min(max(position[0], 0), self.image_shape[0])
            position[1] = min(max(position[1], 0), self.image_shape[1])

        return out_of_pitch, position

    def add_object(self, x, y, type, args={}):
        """
        Store an object in self.objects
        """
        err, position = self.world_to_display_space(x, y)
        if err:
            return

        if 'theta' in args:
            position = np.append(position, args['theta'])
        if 'r' in args:
            position = np.append(position, args['r'])
        n = 0
        if 'n' in args:
            n = args['n']

        self.objects.append((position, type, n))

    def add_box_action(self, msg):
        box = []
        _, position = self.world_to_display_space(msg.x1, msg.y1)
        box.append(position[0])
        box.append(position[1])
        _, position = self.world_to_display_space(msg.x2, msg.y2)
        box.append(position[0])
        box.append(position[1])

        # num ticks
        box.append(0)

        self.boxes.append(box)
        self.box_server.set_succeeded()

    def tick(self):
        """
        Darken each object, or remove after 5 ticks
        """

        objects = []

        t = [t/10 for t in range(1, 6)]
        future = self.predict_trajectory(t)
        for i, (x, y, confidence) in enumerate(zip(future.x, future.y, future.confidence)):
            err, pos = self.world_to_display_space(x, y)
            if not err:
                pos = [pos[0], pos[1], 0, 0, 28 * (1 - confidence) + 2]
                objects.append((pos, FUTURE_BALL, int(i * (1 - confidence))))

        for (pos, type, n) in self.objects:
            if n < 5:
                objects.append((pos, type, n+1))
        self.objects = objects
        
        boxes = []
        for box in self.boxes:
            if box[4] < 5:
                box[4] += 1
                boxes.append(box)
        self.boxes = boxes

    def draw(self):
        """
        Display the pitch overview
        """
        image = np.zeros(self.image_shape, dtype=np.uint8)
        for box in self.boxes:
            cv2.rectangle(image, (int(box[1]), int(box[2])), (int(box[3]), int(box[0])), (255 * 0.9 ** box[4], 0, 0), 1)
        for (pos, type, n) in self.objects:
            # get object colour from its type and number of ticks
            colour = []
            for channel in COLOURS[type]:
                colour.append(channel * 0.8 ** n)

            if len(pos) > 4:
                r = int(pos[4])
                cv2.circle(image, (int(pos[1]), int(pos[0])), r, tuple(colour), -1)
            else:
                r = RADII[type]
                cv2.circle(image, (int(pos[1]), int(pos[0])), r, tuple(colour), -1)
                if len(pos) > 3:
                    # set r to length of vector
                    r = int(pos[3] * 25)
                    cv2.line(image, 
                    (int(pos[1]), int(pos[0])), 
                    (int(pos[1] + np.cos(pos[2]) * r), int(pos[0] - np.sin(pos[2]) * r)), 
                    tuple(colour), 
                    1)
                elif len(pos) > 2:
                    cv2.circle(image, (int(pos[1] + np.cos(pos[2]) * r), int(
                        pos[0] - np.sin(pos[2]) * r)), int(r * 2/3), tuple(colour), -1)
        

        cv2.imshow('pitch overview', image)
        cv2.waitKey(1)

    def loop(self):
        while not rospy.core.is_shutdown():
            self.draw()
            self.tick()
            rospy.sleep(1 / 20)


if __name__ == '__main__':
    main = DebugPitch()
    main.loop()
