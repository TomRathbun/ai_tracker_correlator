import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Optional
import torch

class KalmanTracker:
    """
    A simple Kalman Filter for a 6D state representation: [x, y, z, vx, vy, vz].
    Assumes a constant velocity model.
    """
    count = 0

    def __init__(self, initial_state: np.ndarray, initial_timestamp: float):
        self.id = KalmanTracker.count
        KalmanTracker.count += 1
        
        # State: [x, y, z, vx, vy, vz]
        self.state = initial_state.flatten()
        self.timestamp = initial_timestamp
        
        # Initial covariance
        self.P = np.eye(6) * 100.0
        self.P[3:, 3:] *= 10.0  # Higher uncertainty for initial velocity
        
        # Process noise
        self.Q = np.eye(6) * 0.1
        
        # Measurement matrix (we measure x, y, z, vx, vy)
        # Note: vz is not typically measured in this dataset directly but we'll include it if available
        # Adjusting to 5D measurement for the radar logic [x, y, z, vx, vy]
        self.H = np.zeros((5, 6))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 3] = 1
        self.H[4, 4] = 1
        
        # Measurement noise
        self.R = np.eye(5) * 50.0  # Position noise
        self.R[3:, 3:] *= 5.0      # Velocity noise
        
        self.age = 0
        self.hits = 0
        self.time_since_update = 0
        self.history = []

    def predict(self, timestamp: float):
        """
        Predict state to the next time step.
        """
        dt = timestamp - self.timestamp
        if dt < 0:
            dt = 0
            
        # Transition matrix F
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        
        # Predict state
        self.state = F @ self.state
        
        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q
        
        self.timestamp = timestamp
        self.time_since_update += 1
        return self.state

    def update(self, measurement: np.ndarray):
        """
        Update the state with a new measurement [x, y, z, vx, vy].
        """
        self.time_since_update = 0
        self.hits += 1
        self.age += 1
        
        # Innovation / Measurement residual
        z = measurement.flatten()
        y = z - (self.H @ self.state)
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman Gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.state = self.state + (K @ y)
        self.P = (np.eye(6) - K @ self.H) @ self.P
        
        return self.state

class SortTracker:
    def __init__(self, init_thresh=0.3, max_age=10, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers: List[KalmanTracker] = []
        self.frame_count = 0

    def update(self, measurements: np.ndarray, timestamp: float):
        """
        Update the SORT tracker with new measurements.
        measurements: np.ndarray [N, 5] (x, y, z, vx, vy)
        """
        self.frame_count += 1
        
        # 1. Predict existing trackers
        for t in self.trackers:
            t.predict(timestamp)
            
        # 2. Data association
        if len(self.trackers) > 0 and len(measurements) > 0:
            tr_states = np.array([t.state[:3] for t in self.trackers])
            me_states = measurements[:, :3]
            
            # Distance cost matrix
            from scipy.spatial.distance import cdist
            cost_matrix = cdist(tr_states, me_states)
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Gating
            match_gate = 15000.0 # Match distance gate
            matched_indices = []
            unmatched_trackers = set(range(len(self.trackers)))
            unmatched_measurements = set(range(len(measurements)))
            
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < match_gate:
                    matched_indices.append((r, c))
                    unmatched_trackers.remove(r)
                    unmatched_measurements.remove(c)
            
            # 3. Update matched trackers
            for r, c in matched_indices:
                self.trackers[r].update(measurements[c])
        else:
            unmatched_measurements = set(range(len(measurements)))
            unmatched_trackers = set(range(len(self.trackers)))

        # 4. Initiate new trackers
        for i in unmatched_measurements:
            # Create measurement for Kalman [x, y, z, vx, vy]
            # Since vz is unknown, we initialize it to 0
            initial_state = np.zeros(6)
            initial_state[:2] = measurements[i, :2] # x, y
            initial_state[2] = measurements[i, 2]   # z
            initial_state[3:5] = measurements[i, 3:5] # vx, vy
            # vz (initial_state[5]) is 0
            
            new_tr = KalmanTracker(initial_state, timestamp)
            self.trackers.append(new_tr)

        # 5. Cull dead trackers
        ret = []
        i = len(self.trackers)
        for t in reversed(self.trackers):
            if t.time_since_update > self.max_age:
                self.trackers.pop(i - 1)
            elif t.hits >= self.min_hits or self.frame_count <= self.min_hits:
                # Active track
                ret.append({
                    'id': t.id,
                    'state': t.state,
                    'hits': t.hits,
                    'age': t.age
                })
            i -= 1
            
        return ret
