import numpy as np
from filterpy.kalman import KalmanFilter as KF
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag


class KalmanFilter:

    def __init__(self, R=1, Q=1):
        self.R = R
        self.Q = Q
        self.init_kf()

    def init_kf(self):
        pass

    def reset(self):
        self.init_kf()
    
    def get_update(self, x):
        return self.kf.get_update(x)

    def predict(self, dt):
        self.kf.predict()

    def update(self, position):
        self.kf.predict()
        self.kf.update(np.array(position))

    def get(self):
        return self.kf.x

    def get_mahalanobis(self):
        return self.kf.mahalanobis


class KalmanFilter1D(KalmanFilter):

    def init_kf(self):
        kf = KF(dim_x=2, dim_z=1)

        kf.x = np.array([[0., 0.]]).T
        kf.P *= np.diag((1., 1))
        kf.F = np.array([[1.,1.],
                         [0.,1.]])

        kf.H = np.array([[1., 0]])

        # measurement noise
        kf.R *= np.eye(1) * (self.R)
        # process noise
        kf.Q = Q_discrete_white_noise(dim=2, dt=1, var=self.Q)

        self.kf = kf


class KalmanFilter2D(KalmanFilter):

    def init_kf(self):
        kf = KF(dim_x=4, dim_z=2)
        dt = 1/10
        kf.x = np.array([[0., 0., 0., 0.]]).T
        kf.P *= np.diag((1., 1, 1, 1))

        kf.F = np.array([[1., 0., dt, 0.],
                        [0., 1., 0., dt],
                        [0., 0., 1., 0],
                        [0., 0., 0., 1.]])

        kf.H = np.array([[1., 0, 0, 0],
                        [0., 1., 0, 0]])

        # measurement noise
        kf.R *= np.eye(2) * (self.R)
        # process noise
        q = Q_discrete_white_noise(dim=2, dt=1, var=self.Q)
        kf.Q = block_diag(q, q)

        kf.alpha = 1.1
        # kf.alpha = 1

        self.kf = kf
