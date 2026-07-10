"""
Simple Kalman Filter for 6D state (position + velocity).

Supports partial measurements:
  - [x, y, z]
  - [x, y, z, vx, vy]
  - [x, y, z, vx, vy, vz]
"""
import numpy as np


class SimpleKalmanFilter:
    """Constant velocity Kalman filter for 3D tracking."""

    def __init__(self, dt: float = 1.0):
        # State: [x, y, z, vx, vy, vz]
        self.x = np.zeros(6, dtype=float)
        self.dt = dt
        self._update_F(dt)

        # Covariance matrix (Initial uncertainty)
        # Position 1km, Velocity 500m/s
        self.P = np.diag(
            [1000.0**2, 1000.0**2, 1000.0**2, 500.0**2, 500.0**2, 500.0**2]
        ).astype(float)

        # Process noise (how much state can change per step)
        # 50.0 m/s^2 corresponds to a maneuvering aircraft
        self.Q = np.eye(6, dtype=float) * 50.0**2

        # Measurement noise (radar accuracy)
        self.R = np.eye(6, dtype=float) * 150.0**2
        # Slightly higher velocity measurement noise
        self.R[3:, 3:] = np.eye(3) * 20.0**2

    def _update_F(self, dt):
        """Update state transition matrix for current dt."""
        self.F = np.array(
            [
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        )

    def predict(self, dt=None):
        """Predict next state with optional variable dt."""
        if dt is not None and dt != self.dt:
            self.dt = float(dt)
            self._update_F(self.dt)

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """
        Update with measurement.

        z can be:
          - length 3: position only [x, y, z]  (SSR / no Doppler)
          - length 5: [x, y, z, vx, vy]       (PSR without vz)
          - length 6: full [x, y, z, vx, vy, vz]
        """
        z_val = np.asarray(z, dtype=float).ravel()
        if z_val.size == 0:
            return

        # Decide which state components are observed
        if z_val.size <= 3:
            obs_idx = np.array([0, 1, 2], dtype=int)
            z_obs = z_val[:3]
        elif z_val.size == 5:
            # Common PSR case: position + horizontal velocity
            obs_idx = np.array([0, 1, 2, 3, 4], dtype=int)
            z_obs = z_val[:5]
        elif z_val.size >= 6:
            z_obs = z_val[:6]
            # If caller padded missing velocity with NaN, drop those rows
            if np.any(np.isnan(z_obs[3:6])):
                valid = ~np.isnan(z_obs)
                # Always keep position if present
                obs_idx = np.where(valid)[0]
                z_obs = z_obs[obs_idx]
            else:
                obs_idx = np.arange(6, dtype=int)
        else:
            # length 1, 2, or 4 — use whatever leading components we have
            n = int(z_val.size)
            obs_idx = np.arange(n, dtype=int)
            z_obs = z_val[:n]

        n_obs = len(obs_idx)
        H = np.zeros((n_obs, 6), dtype=float)
        for row, col in enumerate(obs_idx):
            H[row, col] = 1.0

        R = self.R[np.ix_(obs_idx, obs_idx)]

        # Innovation
        y = z_obs - H @ self.x

        # Innovation covariance
        S = H @ self.P @ H.T + R

        # Kalman gain (solve instead of explicit inv for stability)
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)

        # Update state
        self.x = self.x + K @ y

        # Joseph form covariance update for numerical stability
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
