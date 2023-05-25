"""
Class to easily store information about a circle
"""
import numpy as np
import cv2

COLOURS = {
    'hog': (0, 0, 255),
    'colour': (0, 255, 255),
    'distance': (255, 0, 255),
    'best_per_frame': (0, 0, 128),
    'within_pitch': (128, 0, 128),
    None: (0, 255, 0)
}

class Circle:
    def __init__(self, x, y, r, index=None):
        self.x = x
        self.y = y
        self.r = r
        self.original_r = r
        self.images = []
        self.stream_index = index
        self.rejected_by = None
        self.confidence = 0

    # Normalise values to: x,y = [-0.5, 0.5], r = [0, 1]
    def normalised(self, size, centre):
        return Circle(
            (self.x - centre[0]) / size[0],
            (self.y - centre[1]) / -size[0],
            self.r / size[0]
        )

    def get_array(self):
        return np.array([self.x, self.y, self.r]).astype("float32")

    def is_in_frame(self, index):
        return self.stream_index == index

    def draw(self, frame, accepted=True):
        cv2.circle(frame, (self.x, self.y), int(self.r), COLOURS[self.rejected_by], 2)

    def bbox(self, max_w, max_h):
        x1, y1 = int(max(0, self.x - self.r)), int(max(0, self.y - self.r))
        x2, y2 = int(min(max_w, self.x + self.r)), int(min(max_h, self.y + self.r))
        return (x1, y1, x2, y2)

