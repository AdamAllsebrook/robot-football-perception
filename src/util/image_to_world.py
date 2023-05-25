#!/usr/bin/python3
"""
Convert a position from image space to world space
"""
import numpy as np
import os

import miro2 as miro
from joblib import load


dir_path = os.path.dirname(os.path.realpath(__file__))


class ImageToWorld:

    def __init__(self):
        # state
        self.cam = miro.lib.camera_model.CameraModel()
        self.kc = miro.lib.kc_interf.kc_miro()
        # self.pose = np.array([0.0, 0.0, 0.0])

        # this needs to be set based on the actual frame size, which
        # can be obtained from the camera frame topic.
        self.cam.set_frame_size(640, 360)

        try:
            self.azimuth_correction = load(os.path.join(dir_path, '../azimuth_correction.gz'))
        except FileNotFoundError:
            self.azimuth_correction = None
        try:
            self.elevation_correction = load(os.path.join(dir_path, '../elevation_correction.gz'))
        except FileNotFoundError:
            self.elevation_correction = None


    # (Callbacks forwarded from ball_vision.py as this script is not a node)
    def callback_sensors(self, msg):
        """
        Get kinematic data
        """

        # get robot feedback
        kin = np.array(msg.position)

        # update kc
        self.kc.setConfig(kin)

    def callback_pose(self, msg):
        """
        Get pose data
        """
        self.kc.setPose([msg.x, msg.y, msg.theta])

    def convert(self, x, y, r, stream_index):
        """
        Convert from image to world space (p2ow)
        """
        p = [x, y]

        # map to a view line in CAM
        v = self.cam.p2v(p)

        # map to a location in HEAD
        oh = self.cam.v2oh(stream_index, v, r)

        # map the target into WORLD
        ow = self.kc.changeFrameAbs(
            miro.constants.LINK_HEAD, miro.constants.LINK_WORLD, oh)

        return [ow[0], ow[1], ow[2]]

    def correct(self, vw, image_pos):
        """
        Correct azimuth and elevation of vector in world space
        """
        # get azimuth and elevation
        azim = np.arctan2(vw[0], vw[1])
        elev = np.arctan2(vw[2], np.sqrt(vw[1] ** 2 + vw[0] ** 2))

        # correct azimuth and elevation
        if self.azimuth_correction is not None:
            theta = (self.azimuth_correction['A'] 
                * np.sin(image_pos[0] * self.azimuth_correction['omega']))
            azim -= theta
        if self.elevation_correction is not None:
            elev += self.elevation_correction(image_pos[1])

        # convert back into vector form
        return np.array([
            np.sin(azim) * np.cos(elev),
            np.cos(azim) * np.cos(elev),
            np.sin(elev)
        ])


