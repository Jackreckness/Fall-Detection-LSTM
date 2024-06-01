import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints


class UnscentedKalmanFilterPredictor:
    def __init__(self, fps):
        timestep = 1/fps  # Time between frames in the video.
        def fx(x, dt):
            # State transition function.
            F = np.array([[1,0,dt,0],
                        [0,1,0,dt],
                        [0,0,1,0],
                        [0,0,0,1]], np.float32)
            return np.dot(F, x)

        def hx(x):
            # Extract the measurement from the state.
            return np.array(x[:2])
        
        points = MerweScaledSigmaPoints(4, 0.1, 2.0, 1)

        self.kalman = UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=timestep, fx=fx, hx=hx, points=points)
        self.kalman.x = np.array([1,1,1,1])  # Initial State
        self.kalman.P = np.array([[2,0,0,0],
                            [0,2,0,0],
                            [0,0,2,0],
                            [0,0,0,2]], np.float32)  # Covariance Matrix
        self.kalman.R = np.array([[1,0],
                            [0,1]], np.float32)  # Measurement Noise
        self.kalman.Q = np.array([[1,0,0,0],
                            [0,1,0,0],
                            [0,0,100,0],
                            [0,0,0,100]], np.float32)  # Process Noise
        

    def predict(self):
        self.kalman.predict()
        return (self.kalman.x[0], self.kalman.x[1])

    def update(self, measured):
        self.kalman.update(measured)
