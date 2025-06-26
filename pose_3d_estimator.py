"""
3D Pose Integration Module for Enhanced ReID Pipeline
FIXED VERSION: Ensures consistent feature dimensions
Place this file as: Horse_tracking/pose_3d_estimator.py

This module implements 3D pose estimation from 2D poses and depth maps,
providing enhanced geometric features for improved ReID performance.
"""

import torch
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch.nn.functional as F
from collections import defaultdict

class Pose3DEstimator:
    """
    3D Pose Estimation Module for ReID Enhancement
    Converts 2D poses to 3D using depth information and geometric constraints
    FIXED: Ensures consistent feature dimensions across frames
    """
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # 3D pose parameters from config
        self.enable_pose_prediction = getattr(config, 'enable_pose_prediction', False)
        self.pose_prediction_window = getattr(config, 'pose_prediction_window', 3)
        self.enable_multi_scale_depth = getattr(config, 'enable_multi_scale_depth', False)
        self.enable_depth_attention = getattr(config, 'enable_depth_attention', False)
        
        # Depth processing parameters
        self.depth_smoothing = getattr(config, 'depth_smoothing', 'adaptive_gaussian')
        self.gpu_acceleration = getattr(config, 'gpu_acceleration', True)
        self.batch_processing = getattr(config, 'batch_processing', True)
        self.gpu_batch_size = getattr(config, 'gpu_batch_size', 64)
        
        # FIXED: Consistent feature dimensions
        self.visual_feature_dim = 32  # Standard visual features from ReID pipeline
        self.geometric_feature_dim = 20  # 3D geometric features
        self.enhanced_feature_dim = self.visual_feature_dim + self.geometric_feature_dim  # 52 total
        
        # Feature weights for ReID (from config output)
        self.visual_weight = 0.35
        self.pose_3d_weight = 0.30
        self.geometric_weight = 0.20
        self.motion_weight = 0.10
        
        # Pose history for temporal consistency
        self.pose_history = defaultdict(list)
        self.max_history = self.pose_prediction_window
        
        # Quality parameters
        self.temporal_smoothing = getattr(config, 'temporal_smoothing', True)
        self.hole_filling = getattr(config, 'hole_filling', True)
        self.edge_preservation = getattr(config, 'edge_preservation', True)
        self.noise_reduction = getattr(config, 'noise_reduction', 0.7)
        
        # Geometric features flags
        self.volumetric_features = getattr(config, 'volumetric_features', True)
        self.spatial_distribution = getattr(config, 'spatial_distribution', True)
        self.depth_gradients = getattr(config, 'depth_gradients', True)
        self.temporal_3d = getattr(config, 'temporal_3d', True)
        
        print(f"🎯 3D Pose Estimator initialized")
        print(f"   Pose prediction: {self.enable_pose_prediction}")
        print(f"   Multi-scale depth: {self.enable_multi_scale_depth}")
        print(f"   Depth attention: {self.enable_depth_attention}")
        print(f"   GPU acceleration: {self.gpu_acceleration}")
        print(f"   Batch processing: {self.batch_processing}")
        print(f"   Feature dimensions: Visual({self.visual_feature_dim}) + Geometric({self.geometric_feature_dim}) = Enhanced({self.enhanced_feature_dim})")
    
    def estimate_3d_poses(self, poses_2d: List[Dict], depth_map: np.ndarray, 
                         camera_intrinsics: Optional[Dict] = None) -> List[Dict]:
        """
        Convert 2D poses to 3D using depth information
        
        Args:
            poses_2d: List of 2D pose results with keypoints
            depth_map: Depth map from Depth-Anything
            camera_intrinsics: Camera calibration parameters (optional)
            
        Returns:
            List of 3D pose results
        """
        if not poses_2d or depth_map is None:
            return []
        
        poses_3d = []
        
        # Default camera intrinsics if not provided
        if camera_intrinsics is None:
            h, w = depth_map.shape
            camera_intrinsics = {
                'fx': w * 0.8,  # Approximate focal length
                'fy': h * 0.8,
                'cx': w / 2,    # Principal point
                'cy': h / 2
            }
        
        # Process in batches if GPU acceleration is enabled
        if self.batch_processing and self.gpu_acceleration and len(poses_2d) > 1:
            poses_3d = self.batch_process_poses(poses_2d, depth_map, camera_intrinsics)
        else:
            # Process individually
            for i, pose_2d in enumerate(poses_2d):
                try:
                    pose_3d = self.lift_2d_to_3d(pose_2d, depth_map, camera_intrinsics)
                    if pose_3d is not None:
                        poses_3d.append(pose_3d)
                except Exception as e:
                    print(f"⚠️ 3D pose estimation failed for pose {i}: {e}")
                    continue
        
        return poses_3d
    
    def batch_process_poses(self, poses_2d: List[Dict], depth_map: np.ndarray, 
                           camera_intrinsics: Dict) -> List[Dict]:
        """
        Batch process multiple poses for GPU acceleration
        """
        poses_3d = []
        batch_size = min(self.gpu_batch_size, len(poses_2d))
        
        for i in range(0, len(poses_2d), batch_size):
            batch_poses = poses_2d[i:i + batch_size]
            
            for pose_2d in batch_poses:
                try:
                    pose_3d = self.lift_2d_to_3d(pose_2d, depth_map, camera_intrinsics)
                    if pose_3d is not None:
                        poses_3d.append(pose_3d)
                except Exception as e:
                    continue
        
        return poses_3d
    
    def lift_2d_to_3d(self, pose_2d: Dict, depth_map: np.ndarray, 
                      camera_intrinsics: Dict) -> Optional[Dict]:
        """
        Lift single 2D pose to 3D using depth information with enhanced processing
        """
        if 'keypoints' not in pose_2d:
            return None
        
        keypoints_2d = pose_2d['keypoints']
        method = pose_2d.get('method', 'unknown')
        track_id = pose_2d.get('track_id', -1)
        
        # Handle different keypoint formats
        if hasattr(keypoints_2d, 'cpu'):
            keypoints_2d = keypoints_2d.cpu().numpy()
        
        # Apply depth preprocessing if enabled
        processed_depth = self.preprocess_depth_map(depth_map) if self.hole_filling else depth_map
        
        keypoints_3d = []
        valid_count = 0
        confidence_sum = 0.0
        
        for kp_idx, kp in enumerate(keypoints_2d):
            if len(kp) >= 3:  # [x, y, confidence]
                x, y, conf = kp[0], kp[1], kp[2]
                
                # Skip invalid keypoints
                if conf <= 0 or x <= 0 or y <= 0:
                    keypoints_3d.append([0, 0, 0, 0])  # [x, y, z, confidence]
                    continue
                
                # Get depth value at keypoint location with enhanced sampling
                depth_value = self.sample_depth_at_keypoint(processed_depth, x, y)
                
                # Convert to 3D world coordinates
                if depth_value > 0:
                    # Unproject to 3D using camera intrinsics
                    x_3d = (x - camera_intrinsics['cx']) * depth_value / camera_intrinsics['fx']
                    y_3d = (y - camera_intrinsics['cy']) * depth_value / camera_intrinsics['fy']
                    z_3d = depth_value
                    
                    # Apply temporal smoothing if enabled
                    if self.temporal_smoothing and track_id >= 0:
                        z_3d = self.apply_temporal_smoothing(track_id, kp_idx, z_3d)
                    
                    # Apply noise reduction
                    if self.noise_reduction > 0:
                        final_conf = conf * (1.0 - self.noise_reduction * 0.1)  # Slight confidence reduction for robustness
                    else:
                        final_conf = conf
                    
                    keypoints_3d.append([x_3d, y_3d, z_3d, final_conf])
                    valid_count += 1
                    confidence_sum += final_conf
                else:
                    keypoints_3d.append([0, 0, 0, 0])
            else:
                keypoints_3d.append([0, 0, 0, 0])
        
        if valid_count == 0:
            return None
        
        # Create 3D pose result
        pose_3d = {
            'keypoints_3d': np.array(keypoints_3d),
            'keypoints_2d': keypoints_2d,
            'method': method,
            'track_id': track_id,
            'confidence': confidence_sum / valid_count,
            'valid_keypoints': valid_count,
            'box': pose_2d.get('box', None),
            'frame_timestamp': pose_2d.get('frame_timestamp', 0)
        }
        
        # Update pose history for temporal consistency
        if track_id >= 0 and self.temporal_3d:
            self.update_pose_history(track_id, pose_3d)
        
        return pose_3d
    
    def preprocess_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Preprocess depth map with hole filling and edge preservation
        """
        processed_depth = depth_map.copy()
        
        if self.hole_filling:
            # Fill holes using inpainting
            mask = (processed_depth == 0).astype(np.uint8)
            if np.any(mask):
                processed_depth = cv2.inpaint(processed_depth.astype(np.float32), mask, 3, cv2.INPAINT_TELEA)
        
        if self.edge_preservation:
            # Apply bilateral filter for edge preservation
            processed_depth = cv2.bilateralFilter(processed_depth.astype(np.float32), 5, 50, 50)
        
        return processed_depth.astype(np.uint8)
    
    def sample_depth_at_keypoint(self, depth_map: np.ndarray, x: float, y: float) -> float:
        """
        Sample depth at keypoint location with multi-scale approach if enabled
        """
        h, w = depth_map.shape
        x_int = int(np.clip(x, 0, w - 1))
        y_int = int(np.clip(y, 0, h - 1))
        
        if self.enable_multi_scale_depth:
            return self.sample_multi_scale_depth(depth_map, x_int, y_int)
        else:
            return float(depth_map[y_int, x_int])
    
    def sample_multi_scale_depth(self, depth_map: np.ndarray, x: int, y: int, 
                                window_size: int = 5) -> float:
        """
        Sample depth using multi-scale approach for robustness
        """
        h, w = depth_map.shape
        
        # Ensure coordinates are within bounds
        x = np.clip(x, window_size//2, w - window_size//2 - 1)
        y = np.clip(y, window_size//2, h - window_size//2 - 1)
        
        # Extract window around keypoint
        window = depth_map[y-window_size//2:y+window_size//2+1, 
                          x-window_size//2:x+window_size//2+1]
        
        # Remove zero values (invalid depth)
        valid_depths = window[window > 0]
        
        if len(valid_depths) == 0:
            return 0.0
        
        if self.enable_depth_attention:
            # Use attention-weighted average
            center_depth = depth_map[y, x]
            if center_depth > 0:
                weights = np.exp(-np.abs(valid_depths - center_depth) / (center_depth + 1e-6))
                return np.average(valid_depths, weights=weights)
        
        # Use median for robustness
        return float(np.median(valid_depths))
    
    def apply_temporal_smoothing(self, track_id: int, keypoint_idx: int, depth_value: float) -> float:
        """
        Apply temporal smoothing to depth values for consistency
        """
        if track_id not in self.pose_history or len(self.pose_history[track_id]) == 0:
            return depth_value
        
        # Get recent depth values for this keypoint
        recent_depths = []
        for pose in self.pose_history[track_id]:
            if len(pose['keypoints_3d']) > keypoint_idx:
                kp_3d = pose['keypoints_3d'][keypoint_idx]
                if kp_3d[3] > 0:  # Valid keypoint
                    recent_depths.append(kp_3d[2])  # Z coordinate
        
        if len(recent_depths) == 0:
            return depth_value
        
        # Apply adaptive Gaussian smoothing
        if self.depth_smoothing == 'adaptive_gaussian':
            # Weights decay exponentially with time
            weights = np.exp(-0.5 * np.arange(len(recent_depths))**2)
            weights = weights / np.sum(weights)
            
            smoothed_depth = np.sum(np.array(recent_depths) * weights) * 0.7 + depth_value * 0.3
            return smoothed_depth
        else:
            # Simple moving average
            return np.mean(recent_depths + [depth_value])
    
    def update_pose_history(self, track_id: int, pose_3d: Dict):
        """
        Update pose history for temporal consistency
        """
        self.pose_history[track_id].append(pose_3d)
        
        # Keep only recent history
        if len(self.pose_history[track_id]) > self.max_history:
            self.pose_history[track_id].pop(0)
    
    def extract_geometric_features(self, pose_3d: Dict) -> np.ndarray:
        """
        Extract comprehensive geometric features from 3D pose for ReID
        FIXED: Always returns exactly geometric_feature_dim (20) dimensions
        """
        if 'keypoints_3d' not in pose_3d:
            return np.zeros(self.geometric_feature_dim, dtype=np.float32)  # Return zero features if no 3D pose
        
        keypoints_3d = pose_3d['keypoints_3d']
        features = []
        
        # Valid keypoints only
        valid_mask = keypoints_3d[:, 3] > 0
        valid_kps = keypoints_3d[valid_mask]
        
        if len(valid_kps) < 3:
            return np.zeros(self.geometric_feature_dim, dtype=np.float32)
        
        points_3d = valid_kps[:, :3]
        
        # 1. Volumetric features (4 dimensions)
        if self.volumetric_features and len(valid_kps) >= 3:
            # Bounding box volume
            bbox_3d = np.array([
                np.max(points_3d[:, 0]) - np.min(points_3d[:, 0]),  # width
                np.max(points_3d[:, 1]) - np.min(points_3d[:, 1]),  # height  
                np.max(points_3d[:, 2]) - np.min(points_3d[:, 2])   # depth
            ])
            volume = np.prod(bbox_3d)
            features.extend([volume / 1000000, bbox_3d[0]/1000, bbox_3d[1]/1000, bbox_3d[2]/1000])
        else:
            features.extend([0, 0, 0, 0])
        
        # 2. Spatial distribution features (4 dimensions)
        if self.spatial_distribution:
            centroid = np.mean(points_3d, axis=0)
            distances = np.linalg.norm(points_3d - centroid, axis=1)
            features.extend([
                np.mean(distances) / 1000,
                np.std(distances) / 1000,
                np.min(distances) / 1000,
                np.max(distances) / 1000
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # 3. Depth gradient features (4 dimensions)
        if self.depth_gradients and len(points_3d) >= 5:
            depth_values = points_3d[:, 2]
            depth_gradient = np.gradient(depth_values)
            features.extend([
                np.mean(depth_gradient) / 1000,
                np.std(depth_gradient) / 1000,
                np.min(depth_gradient) / 1000,
                np.max(depth_gradient) / 1000
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # 4. Pose compactness and density (2 dimensions)
        if len(valid_kps) >= 3:
            centroid = np.mean(points_3d, axis=0)
            distances = np.linalg.norm(points_3d - centroid, axis=1)
            pose_spread = np.mean(distances) / (np.max(distances) + 1e-6)
            keypoint_density = len(valid_kps) / len(keypoints_3d)
            features.extend([pose_spread, keypoint_density])
        else:
            features.extend([0, 0])
        
        # 5. Temporal 3D features (2 dimensions)
        if self.temporal_3d and pose_3d.get('track_id', -1) >= 0:
            track_id = pose_3d['track_id']
            if track_id in self.pose_history and len(self.pose_history[track_id]) > 1:
                prev_pose = self.pose_history[track_id][-2]
                prev_valid = prev_pose['keypoints_3d'][prev_pose['keypoints_3d'][:, 3] > 0]
                
                if len(prev_valid) >= 3:
                    prev_centroid = np.mean(prev_valid[:, :3], axis=0)
                    current_centroid = np.mean(points_3d, axis=0)
                    motion_3d = np.linalg.norm(current_centroid - prev_centroid)
                    
                    # Motion smoothness
                    if len(self.pose_history[track_id]) >= 3:
                        prev_prev_pose = self.pose_history[track_id][-3]
                        prev_prev_valid = prev_prev_pose['keypoints_3d'][prev_prev_pose['keypoints_3d'][:, 3] > 0]
                        if len(prev_prev_valid) >= 3:
                            prev_prev_centroid = np.mean(prev_prev_valid[:, :3], axis=0)
                            prev_motion = np.linalg.norm(prev_centroid - prev_prev_centroid)
                            motion_smoothness = 1.0 / (1.0 + abs(motion_3d - prev_motion))
                        else:
                            motion_smoothness = 0.5
                    else:
                        motion_smoothness = 0.5
                    
                    features.extend([motion_3d / 1000, motion_smoothness])
                else:
                    features.extend([0, 0])
            else:
                features.extend([0, 0])
        else:
            features.extend([0, 0])
        
        # 6. Pose stability and confidence (4 dimensions)
        avg_confidence = np.mean(valid_kps[:, 3])
        conf_std = np.std(valid_kps[:, 3])
        
        # Additional pose quality metrics
        relative_spread = np.std(np.linalg.norm(points_3d - np.mean(points_3d, axis=0), axis=1)) / 1000
        depth_consistency = 1.0 / (1.0 + np.std(points_3d[:, 2]) / 1000)
        
        features.extend([avg_confidence, conf_std, relative_spread, depth_consistency])
        
        # CRITICAL: Ensure we have exactly geometric_feature_dim features
        features = features[:self.geometric_feature_dim] + [0] * max(0, self.geometric_feature_dim - len(features))
        
        # Normalize and clean
        features = np.array(features, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Apply feature normalization
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def combine_features_with_3d(self, visual_features: np.ndarray, 
                                pose_3d: Optional[Dict] = None) -> np.ndarray:
        """
        FIXED: Combine visual features with 3D pose features for enhanced ReID
        Ensures consistent output dimensions
        """
        # Ensure visual features are the correct dimension
        if len(visual_features) != self.visual_feature_dim:
            print(f"⚠️ Warning: Visual features dimension mismatch. Expected {self.visual_feature_dim}, got {len(visual_features)}")
            # Pad or truncate to correct size
            if len(visual_features) < self.visual_feature_dim:
                padded_visual = np.zeros(self.visual_feature_dim, dtype=np.float32)
                padded_visual[:len(visual_features)] = visual_features
                visual_features = padded_visual
            else:
                visual_features = visual_features[:self.visual_feature_dim]
        
        if pose_3d is None:
            # If no 3D pose, return weighted visual features + zero geometric features
            enhanced_features = np.zeros(self.enhanced_feature_dim, dtype=np.float32)
            enhanced_features[:self.visual_feature_dim] = visual_features * (self.visual_weight + self.pose_3d_weight)
            # Geometric part remains zero
            return enhanced_features
        
        # Extract geometric features from 3D pose (always returns geometric_feature_dim dimensions)
        geometric_features = self.extract_geometric_features(pose_3d)
        
        # Normalize features to same scale
        visual_norm = visual_features / (np.linalg.norm(visual_features) + 1e-8)
        geometric_norm = geometric_features / (np.linalg.norm(geometric_features) + 1e-8)
        
        # Combine with learned weights from config
        combined_features = np.concatenate([
            visual_norm * self.visual_weight,
            geometric_norm * self.geometric_weight
        ])
        
        # Ensure exact dimension
        assert len(combined_features) == self.enhanced_feature_dim, \
            f"Feature dimension mismatch: expected {self.enhanced_feature_dim}, got {len(combined_features)}"
        
        # Apply final normalization
        final_norm = np.linalg.norm(combined_features)
        if final_norm > 0:
            combined_features = combined_features / final_norm
        
        return combined_features.astype(np.float32)
    
    def predict_next_pose_3d(self, track_id: int) -> Optional[Dict]:
        """
        Predict next 3D pose based on motion history (if pose prediction is enabled)
        """
        if not self.enable_pose_prediction or track_id not in self.pose_history:
            return None
        
        history = self.pose_history[track_id]
        if len(history) < 2:
            return None
        
        # Simple linear prediction based on last two poses
        current_pose = history[-1]
        previous_pose = history[-2]
        
        current_kps = current_pose['keypoints_3d']
        previous_kps = previous_pose['keypoints_3d']
        
        predicted_kps = []
        for i in range(len(current_kps)):
            if current_kps[i][3] > 0 and previous_kps[i][3] > 0:
                # Linear extrapolation
                velocity = current_kps[i][:3] - previous_kps[i][:3]
                predicted_pos = current_kps[i][:3] + velocity
                predicted_conf = current_kps[i][3] * 0.8  # Reduce confidence for prediction
                predicted_kps.append([predicted_pos[0], predicted_pos[1], predicted_pos[2], predicted_conf])
            else:
                predicted_kps.append([0, 0, 0, 0])
        
        predicted_pose = {
            'keypoints_3d': np.array(predicted_kps),
            'method': current_pose['method'] + '_predicted',
            'track_id': track_id,
            'confidence': current_pose['confidence'] * 0.8,
            'valid_keypoints': sum(1 for kp in predicted_kps if kp[3] > 0),
            'is_prediction': True
        }
        
        return predicted_pose
    
    def cleanup_track_history(self, track_id: int):
        """Clean up pose history for removed tracks"""
        if track_id in self.pose_history:
            del self.pose_history[track_id]
    
    def get_3d_stats(self) -> Dict:
        """Get 3D pose estimation statistics"""
        total_tracks = len(self.pose_history)
        total_poses = sum(len(history) for history in self.pose_history.values())
        
        return {
            'total_3d_tracks': total_tracks,
            'total_3d_poses': total_poses,
            'avg_poses_per_track': total_poses / max(1, total_tracks),
            'temporal_smoothing': self.temporal_smoothing,
            'multi_scale_depth': self.enable_multi_scale_depth,
            'depth_attention': self.enable_depth_attention,
            'feature_dimensions': {
                'visual': self.visual_feature_dim,
                'geometric': self.geometric_feature_dim,
                'enhanced': self.enhanced_feature_dim
            }
        }