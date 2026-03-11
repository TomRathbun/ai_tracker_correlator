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
        self.H = np.eye(6, dtype=float)
        
        # Covariance matrix (Initial uncertainty)
        # Position 1km, Velocity 500m/s
        self.P = np.diag([1000.0**2, 1000.0**2, 1000.0**2, 500.0**2, 500.0**2, 500.0**2]).astype(float)
        
        # Process noise (how much state can change per step)
        # 50.0 m/s^2 corresponds to a maneuvering aircraft
        self.Q = np.eye(6, dtype=float) * 50.0**2
        
        # Measurement noise (radar accuracy)
        self.R = np.eye(6, dtype=float) * 150.0**2
    
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
        """Update with measurement. z can be [x,y,z] or [x,y,z,vx,vy,vz]"""
        # Create a dynamic H matrix based on measurement size
        # If velocity is missing or all zeros (from SSR), only update position
        z_val = np.array(z)
        if len(z_val) > 3 and not np.all(z_val[3:6] == 0):
            H = np.eye(6)
            R = self.R
        else:
            z_val = z_val[:3]
            H = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0]
            ], dtype=np.float64)
            R = self.R[:3, :3]

        # Innovation
        y = z_val - H @ self.x
        
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P
