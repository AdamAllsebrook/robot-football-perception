# values obtained from http://wiki.ros.org/camera_calibration/Tutorials/StereoCalibration

import numpy as np

"""


left

matrix
184.710605 0.000000 319.481160
0.000000 184.719345 179.499847
0.000000 0.000000 1.000000

distortion
-0.000034 0.000020 -0.000047 0.000020 0.000000

"""

camera_config = [
    {
        'mtx': np.array([
            [184.710605, 0., 319.48116],
            [0., 184.719345, 179.499847],
            [0., 0., 1.]
            ]),
        'dist': np.array([-0.000034, 0.00002, -0.000047, 0.00002, 0.])
    },
    {
        'mtx': np.array([
            [184.710605, 0., 319.48116],
            [0., 184.719345, 179.499847],
            [0., 0., 1.]
            ]),
        'dist': np.array([-0.000034, 0.00002, -0.000047, 0.00002, 0.])
    }
]