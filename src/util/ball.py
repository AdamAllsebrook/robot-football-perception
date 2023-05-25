"""
Class to store data about a ball observation
"""

import numpy as np

class Ball:
    def __init__(self, position, image_position, radius, stream_index, camera_position=[np.nan, np.nan, np.nan]):
        self.position = position
        self.image_position = image_position
        self.radius = radius
        self.stream_index = stream_index
        self.camera_position = camera_position