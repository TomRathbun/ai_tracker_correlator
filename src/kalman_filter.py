"""
Simple Kalman Filter for 6D state (position + velocity)
"""
import numpy as np

class SimpleKalmanFilter:
    """Constant velocity Kalman filter for 3D tracking"""
    
    def __init__(self, dt: float = 1.0):
        # State: [x, y, z, vx, vy, vz]
        self.x = np.zeros(6)
        self.dt = dt
        self._update_F(dt)
        
        # Measurement matrix (observe all states)
        self.H = np.eye(6)
        
        # Covariance matrix
        self.P = np.eye(6) * 5000.0**2
        
        # Process noise (how much state can change per step)
        self.Q = np.eye(6) * 20.0**2
        
        # Measurement noise (radar accuracy)
        self.R = np.eye(6) * 150.0**2
    
    def _update_F(self, dt):
        """Update state transition matrix for current dt."""
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

    def predict(self, dt=None):
        """Predict next state with optional variable dt."""
        if dt is not None and dt != self.dt:
            self.dt = dt
            self._update_F(dt)
            
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
