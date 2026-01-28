"""
Data Augmentation Module

Provides research-specific augmentations for simulating real-world
radar variability (SSR ID dropouts, noise injection, etc.).
"""
import random
import copy
from typing import List, Dict
import numpy as np


class DataAugmentor:
    """
    Handles data augmentation for radar measurements.
    
    Supports:
    - SSR ID dropouts (10-20% rate)
    - Gaussian noise injection
    - Measurement dropouts
    - Sensor bias simulation
    """
    
    def __init__(
        self,
        ssr_dropout_rate: float = 0.0,
        noise_std: float = 0.0,
        measurement_dropout_rate: float = 0.0,
        position_bias: Dict[str, float] = None
    ):
        """
        Initialize augmentor.
        
        Args:
            ssr_dropout_rate: Probability of dropping SSR identity codes (0-1)
            noise_std: Standard deviation of Gaussian noise to add to positions
            measurement_dropout_rate: Probability of dropping entire measurements
            position_bias: Dict of sensor-specific position biases (x, y, z)
        """
        self.ssr_dropout_rate = ssr_dropout_rate
        self.noise_std = noise_std
        self.measurement_dropout_rate = measurement_dropout_rate
        self.position_bias = position_bias or {}
    
    def augment_frame(self, frame: Dict) -> Dict:
        """
        Augment a single frame of measurements.
        
        Args:
            frame: Frame dictionary with 'measurements' key
            
        Returns:
            Augmented frame dictionary
        """
        augmented_frame = copy.deepcopy(frame)
        measurements = augmented_frame.get('measurements', [])
        
        augmented_measurements = []
        for m in measurements:
            # Measurement dropout
            if random.random() < self.measurement_dropout_rate:
                continue
            
            m_aug = copy.deepcopy(m)
            
            # SSR ID dropout
            if self._is_ssr(m_aug) and random.random() < self.ssr_dropout_rate:
                m_aug['mode_3a'] = None
                m_aug['mode_s'] = None
            
            # Position noise
            if self.noise_std > 0:
                m_aug['x'] += np.random.normal(0, self.noise_std)
                m_aug['y'] += np.random.normal(0, self.noise_std)
                m_aug['z'] += np.random.normal(0, self.noise_std)
            
            # Sensor bias
            sensor_id = m_aug.get('sensor_id', 0)
            if sensor_id in self.position_bias:
                bias = self.position_bias[sensor_id]
                m_aug['x'] += bias.get('x', 0)
                m_aug['y'] += bias.get('y', 0)
                m_aug['z'] += bias.get('z', 0)
            
            augmented_measurements.append(m_aug)
        
        augmented_frame['measurements'] = augmented_measurements
        return augmented_frame
    
    def augment_dataset(self, frames: List[Dict]) -> List[Dict]:
        """
        Augment an entire dataset.
        
        Args:
            frames: List of frame dictionaries
            
        Returns:
            List of augmented frame dictionaries
        """
        return [self.augment_frame(frame) for frame in frames]
    
    def _is_ssr(self, measurement: Dict) -> bool:
        """Check if measurement is from SSR sensor."""
        sensor_id = measurement.get('sensor_id', 0)
        return sensor_id >= 2 or measurement.get('sensor_type') == 'ssr'
    
    @classmethod
    def from_config(cls, config: Dict) -> 'DataAugmentor':
        """
        Create augmentor from configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            DataAugmentor instance
        """
        return cls(
            ssr_dropout_rate=config.get('ssr_dropout_rate', 0.0),
            noise_std=config.get('noise_std', 0.0),
            measurement_dropout_rate=config.get('measurement_dropout_rate', 0.0),
            position_bias=config.get('position_bias', {})
        )
