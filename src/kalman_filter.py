"""
Simple Kalman Filter for 6D state (position + velocity)
"""
import numpy as np

class SimpleKalmanFilter:
    """Constant velocity Kalman filter for 3D tracking"""
    
    def __init__(self, dt: float = 3.0):
        # State: [x, y, z, vx, vy, vz]
        self.x = np.zeros(6)
        
        # State transition (constant velocity)
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix (observe all states)
        self.H = np.eye(6)
        
        # Covariance matrix
        self.P = np.eye(6) * 5000.0**2
        
        # Process noise
        self.Q = np.eye(6) * 100.0**2
        
        # Measurement noise
        self.R = np.eye(6) * 2000.0**2
    
    def predict(self):
        """Predict next state"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, z):
        """Update with measurement"""
        # Innovation
        y = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P
