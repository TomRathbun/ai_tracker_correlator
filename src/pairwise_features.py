"""
Pairwise feature extraction for heterogeneous measurement association (PSR/SSR).
"""
import numpy as np
from typing import Dict, List, Tuple

def compute_psr_psr_features(m1: Dict, m2: Dict) -> np.ndarray:
    """Features for two PSR measurements (both have velocity)"""
    p1 = np.array([m1['x'], m1['y'], m1['z']])
    p2 = np.array([m2['x'], m2['y'], m2['z']])
    v1 = np.array([m1.get('vx', 0.0), m1.get('vy', 0.0), m1.get('vz', 0.0)])
    v2 = np.array([m2.get('vx', 0.0), m2.get('vy', 0.0), m2.get('vz', 0.0)])
    
    features = []
    
    # Position distance
    pos_dist = np.linalg.norm(p1 - p2)
    features.append(pos_dist / 100000.0)
    
    # Velocity cosine similarity
    v1_norm = np.linalg.norm(v1) + 1e-8
    v2_norm = np.linalg.norm(v2) + 1e-8
    features.append(np.dot(v1, v2) / (v1_norm * v2_norm))
    
    # Velocity magnitude diff
    features.append(abs(v1_norm - v2_norm) / 1000.0)
    
    # Angular separation
    az1, az2 = np.arctan2(p1[1], p1[0]), np.arctan2(p2[1], p2[0])
    az_diff = abs(az1 - az2)
    if az_diff > np.pi: az_diff = 2*np.pi - az_diff
    features.append(az_diff)
    
    el1 = np.arctan2(p1[2], np.sqrt(p1[0]**2 + p1[1]**2))
    el2 = np.arctan2(p2[2], np.sqrt(p2[0]**2 + p2[1]**2))
    features.append(abs(el1 - el2))
    
    # Amplitude similarity (PSR specific)
    amp1 = m1.get('amplitude', 50.0)
    amp2 = m2.get('amplitude', 50.0)
    features.append(abs(amp1 - amp2) / 100.0)
    
    return np.array(features, dtype=np.float32)

def compute_ssr_any_features(m1: Dict, m2: Dict) -> np.ndarray:
    """Features for PSR-SSR or SSR-SSR pairs (uses ID codes if available)"""
    p1 = np.array([m1['x'], m1['y'], m1['z']])
    p2 = np.array([m2['x'], m2['y'], m2['z']])
    
    features = []
    
    # Position distance
    pos_dist = np.linalg.norm(p1 - p2)
    features.append(pos_dist / 100000.0)
    
    # Angular separation (always useful)
    az1, az2 = np.arctan2(p1[1], p1[0]), np.arctan2(p2[1], p2[0])
    az_diff = abs(az1 - az2)
    if az_diff > np.pi: az_diff = 2*np.pi - az_diff
    features.append(az_diff)
    
    # Mode 3A (Squawk) match
    # 1.0 = match, -1.0 = mismatch, 0.0 = N/A (one side missing)
    m3a_1 = m1.get('mode_3a')
    m3a_2 = m2.get('mode_3a')
    if m3a_1 is not None and m3a_2 is not None:
        features.append(1.0 if m3a_1 == m3a_2 else -1.0)
    else:
        features.append(0.0)
        
    # Mode S ICAO match
    ms_1 = m1.get('mode_s')
    ms_2 = m2.get('mode_s')
    if ms_1 is not None and ms_2 is not None:
        features.append(1.0 if ms_1 == ms_2 else -1.0)
    else:
        features.append(0.0)
        
    return np.array(features, dtype=np.float32)

def get_psr_psr_dim(): return 6
def get_ssr_any_dim(): return 4

def extract_specialized_pairs(measurements: List[Dict], classifier_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract pairs specialized for a specific classifier type.
    classifier_type: 'PSR-PSR' or 'SSR-ANY'
    """
    all_features = []
    all_labels = []
    n = len(measurements)
    
    for i in range(n):
        for j in range(i+1, n):
            m1, m2 = measurements[i], measurements[j]
            t1, t2 = m1.get('type', 'PSR'), m2.get('type', 'PSR')
            
            # Determine if this pair belongs to this classifier
            if classifier_type == 'PSR-PSR':
                if t1 == 'PSR' and t2 == 'PSR':
                    feats = compute_psr_psr_features(m1, m2)
                else:
                    continue
            else: # SSR-ANY
                if t1 == 'SSR' or t2 == 'SSR':
                    feats = compute_ssr_any_features(m1, m2)
                else:
                    continue
            
            all_features.append(feats)
            tid1 = m1.get('track_id', -1)
            tid2 = m2.get('track_id', -1)
            label = 1.0 if (tid1 == tid2 and tid1 != -1) else 0.0
            all_labels.append(label)
            
    if not all_features:
        return np.zeros((0, get_psr_psr_dim() if classifier_type == 'PSR-PSR' else get_ssr_any_dim())), np.zeros(0)
        
    return np.array(all_features), np.array(all_labels)

# Maintain backward compatibility for now if needed, but discouraged
def compute_pairwise_features(m1: Dict, m2: Dict) -> np.ndarray:
    t1, t2 = m1.get('type', 'PSR'), m2.get('type', 'PSR')
    if t1 == 'PSR' and t2 == 'PSR':
        return compute_psr_psr_features(m1, m2)
    return compute_ssr_any_features(m1, m2)

def get_feature_dim():
    # This is ambiguous now, should use specialized functions
    return 6 
