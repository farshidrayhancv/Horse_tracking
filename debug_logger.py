"""
Enhanced debug_logger.py with 3D Pose Metrics and Advanced Performance Analysis
Comprehensive tracking for 3D pose estimation, depth processing, and ReID performance
"""

import json
import csv
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

class TrackingDebugLogger:
    def __init__(self, config, log_dir: str = "debug_logs"):
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize data collectors
        self.frame_data = []
        self.detection_data = []
        self.tracking_data = []
        self.reid_data = []
        self.samurai_data = []
        self.pose_data = []
        self.pose_3d_data = []  # NEW: 3D pose specific data
        self.depth_data = []
        self.depth_processing_data = []  # NEW: Enhanced depth processing
        self.consolidation_data = []
        self.recovery_data = []
        self.performance_data = []  # NEW: Performance metrics
        
        # Session metadata
        self.session_start = datetime.now()
        self.total_frames = 0
        self.video_name = None
        self.current_frame = 0
        self.frame_start_time = 0
        
        # Enhanced statistics collectors
        self.stats = {
            'total_detections': 0,
            'total_tracks_created': 0,
            'total_recoveries': 0,
            'total_consolidations': 0,
            'confidence_distribution': {'human': [], 'horse': []},
            'similarity_scores': [],
            'motion_distances': [],
            'depth_stats': [],
            'track_lifecycle': {},
            # NEW: 3D pose statistics
            'pose_3d_quality_distribution': {'human': [], 'horse': []},
            'depth_accuracy_metrics': [],
            'pose_2d_to_3d_conversion_rate': 0,
            'depth_smoothing_effectiveness': [],
            'geometric_feature_quality': [],
            # NEW: Performance statistics
            'processing_times': {
                'detection': [], 'tracking': [], 'pose_estimation': [], 
                'pose_3d_conversion': [], 'reid': [], 'depth_estimation': []
            },
            'gpu_memory_usage': [],
            'frame_rates': []
        }
    
    def set_video_name(self, video_path: str):
        """Set video name for logging"""
        self.video_name = Path(video_path).stem
    
    def log_frame_start(self, frame_num: int, frame_shape: tuple):
        """Log frame start information"""
        self.current_frame = frame_num
        self.frame_start_time = time.time()
        
        frame_info = {
            'frame_num': frame_num,
            'timestamp': datetime.now().isoformat(),
            'frame_shape': frame_shape,
            'processing_start': self.frame_start_time
        }
        
        return frame_info
    
    def log_detections(self, human_detections, horse_detections, detection_method: str, processing_time: float = 0):
        """Log detection information with performance metrics"""
        try:
            import supervision as sv
            
            # Log processing time
            self.stats['processing_times']['detection'].append(processing_time)
            
            # Human detections
            if sv and hasattr(human_detections, 'xyxy') and len(human_detections) > 0:
                for i, (bbox, conf) in enumerate(zip(human_detections.xyxy, human_detections.confidence)):
                    detection_data = {
                        'frame': self.current_frame,
                        'type': 'human',
                        'detection_id': i,
                        'bbox': bbox.tolist(),
                        'confidence': float(conf),
                        'method': detection_method,
                        'area': float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
                        'processing_time_ms': processing_time * 1000
                    }
                    self.detection_data.append(detection_data)
                    self.stats['confidence_distribution']['human'].append(float(conf))
            
            # Horse detections
            if sv and hasattr(horse_detections, 'xyxy') and len(horse_detections) > 0:
                for i, (bbox, conf) in enumerate(zip(horse_detections.xyxy, horse_detections.confidence)):
                    detection_data = {
                        'frame': self.current_frame,
                        'type': 'horse',
                        'detection_id': i,
                        'bbox': bbox.tolist(),
                        'confidence': float(conf),
                        'method': detection_method,
                        'area': float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
                        'processing_time_ms': processing_time * 1000
                    }
                    self.detection_data.append(detection_data)
                    self.stats['confidence_distribution']['horse'].append(float(conf))
            
            human_count = len(human_detections) if hasattr(human_detections, '__len__') else 0
            horse_count = len(horse_detections) if hasattr(horse_detections, '__len__') else 0
            self.stats['total_detections'] += human_count + horse_count
            
        except Exception as e:
            # Fallback logging
            human_count = len(human_detections) if hasattr(human_detections, '__len__') else 0
            horse_count = len(horse_detections) if hasattr(horse_detections, '__len__') else 0
            
            detection_data = {
                'frame': self.current_frame,
                'error': f"Detection logging failed: {e}",
                'human_count': human_count,
                'horse_count': horse_count,
                'method': detection_method,
                'processing_time_ms': processing_time * 1000
            }
            self.detection_data.append(detection_data)
    
    def log_tracking_update(self, tracked_humans, tracked_horses, tracker_type: str = "ByteTrack", processing_time: float = 0):
        """Log tracking updates with performance metrics"""
        try:
            import supervision as sv
            
            # Log processing time
            self.stats['processing_times']['tracking'].append(processing_time)
            
            if not sv:
                return
            
            # Log human tracking
            if hasattr(tracked_humans, 'xyxy') and len(tracked_humans) > 0:
                if hasattr(tracked_humans, 'tracker_id'):
                    for i, (bbox, track_id, conf) in enumerate(zip(
                        tracked_humans.xyxy, tracked_humans.tracker_id, tracked_humans.confidence)):
                        
                        tracking_data = {
                            'frame': self.current_frame,
                            'type': 'human',
                            'track_id': int(track_id),
                            'bbox': bbox.tolist(),
                            'confidence': float(conf),
                            'tracker': tracker_type,
                            'is_new_track': track_id not in self.stats['track_lifecycle'],
                            'processing_time_ms': processing_time * 1000
                        }
                        
                        # Update track lifecycle
                        if track_id not in self.stats['track_lifecycle']:
                            self.stats['track_lifecycle'][track_id] = {
                                'created': self.current_frame,
                                'last_seen': self.current_frame,
                                'total_frames': 1,
                                'type': 'human'
                            }
                            self.stats['total_tracks_created'] += 1
                        else:
                            self.stats['track_lifecycle'][track_id]['last_seen'] = self.current_frame
                            self.stats['track_lifecycle'][track_id]['total_frames'] += 1
                        
                        self.tracking_data.append(tracking_data)
            
            # Log horse tracking
            if hasattr(tracked_horses, 'xyxy') and len(tracked_horses) > 0:
                if hasattr(tracked_horses, 'tracker_id'):
                    for i, (bbox, track_id, conf) in enumerate(zip(
                        tracked_horses.xyxy, tracked_horses.tracker_id, tracked_horses.confidence)):
                        
                        tracking_data = {
                            'frame': self.current_frame,
                            'type': 'horse',
                            'track_id': int(track_id),
                            'bbox': bbox.tolist(),
                            'confidence': float(conf),
                            'tracker': tracker_type,
                            'is_new_track': track_id not in self.stats['track_lifecycle'],
                            'processing_time_ms': processing_time * 1000
                        }
                        
                        # Update track lifecycle
                        if track_id not in self.stats['track_lifecycle']:
                            self.stats['track_lifecycle'][track_id] = {
                                'created': self.current_frame,
                                'last_seen': self.current_frame,
                                'total_frames': 1,
                                'type': 'horse'
                            }
                            self.stats['total_tracks_created'] += 1
                        else:
                            self.stats['track_lifecycle'][track_id]['last_seen'] = self.current_frame
                            self.stats['track_lifecycle'][track_id]['total_frames'] += 1
                        
                        self.tracking_data.append(tracking_data)
                        
        except Exception as e:
            tracking_data = {
                'frame': self.current_frame,
                'error': f"Tracking logging failed: {e}",
                'tracker': tracker_type,
                'processing_time_ms': processing_time * 1000
            }
            self.tracking_data.append(tracking_data)
    
    def log_pose_estimation(self, human_poses: List, horse_poses: List, processing_time: float = 0):
        """Enhanced pose estimation logging with 3D metrics"""
        # Log processing time
        self.stats['processing_times']['pose_estimation'].append(processing_time)
        
        # Log human poses (2D and 3D)
        for i, pose in enumerate(human_poses):
            pose_data = self._extract_pose_data(pose, 'human', i, processing_time)
            self.pose_data.append(pose_data)
            
            # Log 3D specific data if available
            if 'keypoints_3d' in pose:
                pose_3d_data = self._extract_3d_pose_data(pose, 'human', i)
                self.pose_3d_data.append(pose_3d_data)
                self.stats['pose_3d_quality_distribution']['human'].append(pose.get('confidence_3d', 0))
        
        # Log horse poses (2D and 3D)
        for i, pose in enumerate(horse_poses):
            pose_data = self._extract_pose_data(pose, 'horse', i, processing_time)
            self.pose_data.append(pose_data)
            
            # Log 3D specific data if available
            if 'keypoints_3d' in pose:
                pose_3d_data = self._extract_3d_pose_data(pose, 'horse', i)
                self.pose_3d_data.append(pose_3d_data)
                self.stats['pose_3d_quality_distribution']['horse'].append(pose.get('confidence_3d', 0))
    
    def _extract_pose_data(self, pose: Dict, pose_type: str, pose_id: int, processing_time: float) -> Dict:
        """Extract standard pose data"""
        pose_data = {
            'frame': self.current_frame,
            'type': pose_type,
            'pose_id': pose_id,
            'track_id': pose.get('track_id', -1),
            'method': pose.get('method', 'unknown'),
            'confidence': pose.get('confidence', 0.0),
            'num_keypoints': 0,
            'valid_keypoints': 0,
            'processing_time_ms': processing_time * 1000,
            'has_3d': 'keypoints_3d' in pose
        }
        
        # Count keypoints
        if 'keypoints' in pose:
            keypoints = pose['keypoints']
            if hasattr(keypoints, 'cpu'):
                keypoints = keypoints.cpu().numpy()
            pose_data['num_keypoints'] = len(keypoints)
            
            if 'scores' in pose:
                scores = pose['scores']
                if hasattr(scores, 'cpu'):
                    scores = scores.cpu().numpy()
                valid_count = np.sum(scores > 0)
                pose_data['valid_keypoints'] = int(valid_count)
            elif isinstance(keypoints, np.ndarray) and keypoints.shape[-1] >= 3:
                # SuperAnimal format: [x, y, confidence]
                valid_count = np.sum((keypoints[:, 0] != -1) & (keypoints[:, 2] > 0))
                pose_data['valid_keypoints'] = int(valid_count)
        
        return pose_data
    
    def _extract_3d_pose_data(self, pose: Dict, pose_type: str, pose_id: int) -> Dict:
        """Extract 3D pose specific data"""
        keypoints_3d = pose['keypoints_3d']
        
        pose_3d_data = {
            'frame': self.current_frame,
            'type': pose_type,
            'pose_id': pose_id,
            'track_id': pose.get('track_id', -1),
            'method': pose.get('method', 'unknown'),
            'confidence_3d': pose.get('confidence_3d', 0.0),
            'valid_keypoints_3d': pose.get('valid_keypoints_3d', 0),
            'total_keypoints_3d': len(keypoints_3d),
            'depth_statistics': self._calculate_depth_statistics(keypoints_3d),
            'geometric_features': self._calculate_geometric_features(keypoints_3d),
            'pose_quality_metrics': self._calculate_pose_quality_metrics(pose)
        }
        
        return pose_3d_data
    
    def _calculate_depth_statistics(self, keypoints_3d: np.ndarray) -> Dict:
        """Calculate depth-related statistics for 3D pose"""
        valid_keypoints = keypoints_3d[keypoints_3d[:, 3] > 0]
        
        if len(valid_keypoints) == 0:
            return {'mean_depth': 0, 'depth_variance': 0, 'depth_range': 0, 'depth_std': 0}
        
        depths = valid_keypoints[:, 2]
        
        return {
            'mean_depth': float(np.mean(depths)),
            'depth_variance': float(np.var(depths)),
            'depth_range': float(np.ptp(depths)),
            'depth_std': float(np.std(depths)),
            'min_depth': float(np.min(depths)),
            'max_depth': float(np.max(depths))
        }
    
    def _calculate_geometric_features(self, keypoints_3d: np.ndarray) -> Dict:
        """Calculate geometric features for 3D pose"""
        valid_keypoints = keypoints_3d[keypoints_3d[:, 3] > 0]
        
        if len(valid_keypoints) < 3:
            return {'volume_estimate': 0, 'surface_area_estimate': 0, 'compactness': 0}
        
        # 3D bounding box volume
        x_range = np.ptp(valid_keypoints[:, 0])
        y_range = np.ptp(valid_keypoints[:, 1])
        z_range = np.ptp(valid_keypoints[:, 2])
        volume = x_range * y_range * z_range
        
        # Surface area approximation
        surface_area = 2 * (x_range * y_range + y_range * z_range + z_range * x_range)
        
        # Compactness measure
        centroid = np.mean(valid_keypoints[:, :3], axis=0)
        distances = np.linalg.norm(valid_keypoints[:, :3] - centroid, axis=1)
        compactness = np.std(distances) / (np.mean(distances) + 1e-6)
        
        return {
            'volume_estimate': float(volume),
            'surface_area_estimate': float(surface_area),
            'compactness': float(compactness),
            'centroid': centroid.tolist(),
            'avg_distance_from_centroid': float(np.mean(distances)),
            'max_extent': float(np.max(distances))
        }
    
    def _calculate_pose_quality_metrics(self, pose: Dict) -> Dict:
        """Calculate comprehensive pose quality metrics"""
        metrics = {
            'conversion_success': 'keypoints_3d' in pose,
            'depth_consistency_score': 0.0,
            'spatial_coherence_score': 0.0,
            'confidence_distribution_quality': 0.0
        }
        
        if 'keypoints_3d' not in pose:
            return metrics
        
        keypoints_3d = pose['keypoints_3d']
        valid_keypoints = keypoints_3d[keypoints_3d[:, 3] > 0]
        
        if len(valid_keypoints) > 1:
            # Depth consistency (lower variance = better)
            depth_variance = np.var(valid_keypoints[:, 2])
            metrics['depth_consistency_score'] = 1.0 / (1.0 + depth_variance / 1000.0)
            
            # Spatial coherence (how well distributed the keypoints are)
            distances = np.linalg.norm(valid_keypoints[:, :2][1:] - valid_keypoints[:, :2][:-1], axis=1)
            spatial_variance = np.var(distances)
            metrics['spatial_coherence_score'] = 1.0 / (1.0 + spatial_variance / 100.0)
            
            # Confidence distribution quality
            confidences = valid_keypoints[:, 3]
            conf_mean = np.mean(confidences)
            conf_std = np.std(confidences)
            metrics['confidence_distribution_quality'] = conf_mean * (1.0 - conf_std)
        
        return metrics
    
    def log_depth_processing(self, depth_map: np.ndarray, depth_stats: List, processing_time: float = 0):
        """Enhanced depth processing logging"""
        # Log processing time
        self.stats['processing_times']['depth_estimation'].append(processing_time)
        
        depth_frame_data = {
            'frame': self.current_frame,
            'depth_map_shape': list(depth_map.shape) if depth_map is not None else [0, 0],
            'depth_range': {},
            'per_detection_stats': [],
            'processing_time_ms': processing_time * 1000,
            'depth_quality_metrics': {}
        }
        
        if depth_map is not None and depth_map.size > 0:
            # Enhanced depth map analysis
            depth_frame_data['depth_range'] = {
                'min': float(np.min(depth_map)),
                'max': float(np.max(depth_map)),
                'mean': float(np.mean(depth_map)),
                'std': float(np.std(depth_map)),
                'median': float(np.median(depth_map)),
                'percentile_25': float(np.percentile(depth_map, 25)),
                'percentile_75': float(np.percentile(depth_map, 75))
            }
            
            # Depth quality metrics
            depth_frame_data['depth_quality_metrics'] = {
                'uniformity': self._calculate_depth_uniformity(depth_map),
                'gradient_magnitude': self._calculate_average_gradient(depth_map),
                'noise_level': self._estimate_depth_noise(depth_map)
            }
        
        # Enhanced per-detection depth statistics
        for i, stats in enumerate(depth_stats):
            enhanced_stats = {
                'detection_id': i,
                'mask_area': stats.get('area', 0),
                'sam_confidence': stats.get('confidence', 0.0),
                'mask_quality': stats.get('mask_quality', 0.0),
                'depth_variance': stats.get('depth_variance', 0.0),
                'depth_smoothness': self._calculate_depth_smoothness(stats),
                'segmentation_quality': self._calculate_segmentation_quality(stats)
            }
            depth_frame_data['per_detection_stats'].append(enhanced_stats)
            self.stats['depth_stats'].append(stats.get('depth_variance', 0.0))
        
        self.depth_processing_data.append(depth_frame_data)
    
    def _calculate_depth_uniformity(self, depth_map: np.ndarray) -> float:
        """Calculate depth map uniformity score"""
        if depth_map.size == 0:
            return 0.0
        
        # Calculate coefficient of variation
        mean_depth = np.mean(depth_map)
        std_depth = np.std(depth_map)
        
        if mean_depth == 0:
            return 0.0
        
        cv = std_depth / mean_depth
        uniformity = 1.0 / (1.0 + cv)  # Higher uniformity = lower coefficient of variation
        
        return float(uniformity)
    
    def _calculate_average_gradient(self, depth_map: np.ndarray) -> float:
        """Calculate average gradient magnitude in depth map"""
        if depth_map.size == 0:
            return 0.0
        
        # Sobel gradients
        grad_x = np.gradient(depth_map, axis=1)
        grad_y = np.gradient(depth_map, axis=0)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        avg_gradient = np.mean(gradient_magnitude)
        
        return float(avg_gradient)
    
    def _estimate_depth_noise(self, depth_map: np.ndarray) -> float:
        """Estimate noise level in depth map"""
        if depth_map.size == 0:
            return 0.0
        
        # Use Laplacian to estimate noise
        laplacian = np.abs(np.gradient(np.gradient(depth_map, axis=0), axis=0) + 
                          np.gradient(np.gradient(depth_map, axis=1), axis=1))
        
        noise_estimate = np.mean(laplacian)
        return float(noise_estimate)
    
    def _calculate_depth_smoothness(self, stats: Dict) -> float:
        """Calculate depth smoothness for a detection"""
        depth_variance = stats.get('depth_variance', 0)
        area = stats.get('area', 1)
        
        # Normalize variance by area
        normalized_variance = depth_variance / (area + 1)
        smoothness = 1.0 / (1.0 + normalized_variance)
        
        return float(smoothness)
    
    def _calculate_segmentation_quality(self, stats: Dict) -> float:
        """Calculate segmentation quality score"""
        mask_quality = stats.get('mask_quality', 0)
        sam_confidence = stats.get('confidence', 0)
        area = stats.get('area', 0)
        
        # Combine metrics
        quality = (mask_quality * 0.4 + sam_confidence * 0.4 + 
                  min(1.0, area / 10000) * 0.2)  # Normalize area component
        
        return float(quality)
    
    def log_reid_process(self, reid_features, similarity_scores: Dict = None, assignments: Dict = None, 
                        untracked_count: int = 0, processing_time: float = 0, reid_3d_improvements: int = 0):
        """Enhanced ReID process logging with 3D integration metrics"""
        # Log processing time
        self.stats['processing_times']['reid'].append(processing_time)
        
        if similarity_scores is None:
            similarity_scores = {}
        if assignments is None:
            assignments = {}
            
        reid_frame_data = {
            'frame': self.current_frame,
            'num_features_extracted': len(reid_features) if hasattr(reid_features, '__len__') else 0,
            'feature_dimensions': reid_features[0].shape[0] if len(reid_features) > 0 and hasattr(reid_features[0], 'shape') else 0,
            'similarity_scores': {},
            'assignments': assignments,
            'untracked_detections': untracked_count,
            'processing_time_ms': processing_time * 1000,
            'reid_3d_improvements': reid_3d_improvements,
            'feature_quality_metrics': self._calculate_feature_quality_metrics(reid_features)
        }
        
        # Log similarity scores
        for track_id, score in similarity_scores.items():
            reid_frame_data['similarity_scores'][str(track_id)] = float(score)
            self.stats['similarity_scores'].append(float(score))
        
        self.reid_data.append(reid_frame_data)
    
    def _calculate_feature_quality_metrics(self, reid_features) -> Dict:
        """Calculate quality metrics for ReID features"""
        if not hasattr(reid_features, '__len__') or len(reid_features) == 0:
            return {'avg_feature_norm': 0, 'feature_diversity': 0, 'feature_stability': 0}
        
        try:
            features_array = np.array(reid_features)
            
            # Average feature norm
            norms = np.linalg.norm(features_array, axis=1)
            avg_norm = np.mean(norms)
            
            # Feature diversity (average pairwise distance)
            if len(features_array) > 1:
                pairwise_distances = []
                for i in range(len(features_array)):
                    for j in range(i+1, len(features_array)):
                        dist = np.linalg.norm(features_array[i] - features_array[j])
                        pairwise_distances.append(dist)
                diversity = np.mean(pairwise_distances) if pairwise_distances else 0
            else:
                diversity = 0
            
            # Feature stability (inverse of variance across dimensions)
            feature_variance = np.var(features_array, axis=0)
            stability = 1.0 / (1.0 + np.mean(feature_variance))
            
            return {
                'avg_feature_norm': float(avg_norm),
                'feature_diversity': float(diversity),
                'feature_stability': float(stability)
            }
        except Exception:
            return {'avg_feature_norm': 0, 'feature_diversity': 0, 'feature_stability': 0}
    
    def log_performance_metrics(self, gpu_memory_mb: float = 0, frame_rate: float = 0):
        """Log performance metrics"""
        performance_data = {
            'frame': self.current_frame,
            'timestamp': time.time(),
            'gpu_memory_mb': gpu_memory_mb,
            'frame_rate': frame_rate,
            'cpu_usage_percent': self._get_cpu_usage(),
            'memory_usage_mb': self._get_memory_usage()
        }
        
        self.performance_data.append(performance_data)
        self.stats['gpu_memory_usage'].append(gpu_memory_mb)
        self.stats['frame_rates'].append(frame_rate)
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            return psutil.virtual_memory().used / 1024 / 1024
        except ImportError:
            return 0.0
    
    def log_samurai_tracking(self, tracking_info: Dict, recovered_tracks: List = None):
        """Enhanced SAMURAI tracking logging"""
        if recovered_tracks is None:
            recovered_tracks = []
            
        samurai_data = {
            'frame': self.current_frame,
            'active_tracks': tracking_info.get('active_tracks', 0),
            'lost_tracks': tracking_info.get('lost_tracks', 0),
            'memory_efficiency': tracking_info.get('memory_efficiency', 0),
            'total_reassignments': tracking_info.get('total_reassignments', 0),
            'pose_3d_improvements': tracking_info.get('pose_3d_improvements', 0),
            'oscillations_prevented': tracking_info.get('oscillations_prevented', 0),
            'stability_locks': tracking_info.get('stability_locks', 0),
            'motion_predictions': {},
            'recovered_this_frame': len(recovered_tracks)
        }
        
        # Log motion predictions
        motion_predictions = tracking_info.get('motion_predictions', {})
        for track_id, prediction in motion_predictions.items():
            if prediction is not None:
                samurai_data['motion_predictions'][str(track_id)] = {
                    'predicted_x': float(prediction[0]),
                    'predicted_y': float(prediction[1])
                }
        
        # Log recoveries
        for recovery in recovered_tracks:
            recovery_data = {
                'frame': self.current_frame,
                'track_id': recovery.get('track_id', -1),
                'method': recovery.get('method', 'unknown'),
                'confidence': recovery.get('confidence', 0.0),
                'distance': recovery.get('distance', 0.0)
            }
            self.recovery_data.append(recovery_data)
            self.stats['total_recoveries'] += 1
        
        self.samurai_data.append(samurai_data)
    
    def log_track_consolidation(self, consolidations: List):
        """Log track consolidation events"""
        for consolidation in consolidations:
            consolidation_data = {
                'frame': self.current_frame,
                'source_track': consolidation.get('source_track'),
                'target_track': consolidation.get('target_track'),
                'similarity': consolidation.get('similarity', 0.0),
                'distance': consolidation.get('distance', 0.0),
                'reason': consolidation.get('reason', 'similarity')
            }
            self.consolidation_data.append(consolidation_data)
            self.stats['total_consolidations'] += 1
    
    def log_motion_analysis(self, track_id: int, current_pos: np.ndarray, 
                          predicted_pos: np.ndarray, actual_distance: float):
        """Log motion prediction accuracy"""
        if predicted_pos is not None:
            self.stats['motion_distances'].append(float(actual_distance))
    
    def log_frame_end(self, processing_time: float):
        """Log frame end and processing time"""
        frame_data = {
            'frame': self.current_frame,
            'processing_time_ms': processing_time * 1000,
            'timestamp_end': datetime.now().isoformat()
        }
        self.frame_data.append(frame_data)
        self.total_frames = self.current_frame + 1
        
        # Calculate and log frame rate
        if processing_time > 0:
            frame_rate = 1.0 / processing_time
            self.stats['frame_rates'].append(frame_rate)
    
    def calculate_final_statistics(self):
        """Calculate comprehensive statistics including 3D pose metrics"""
        final_stats = {
            'session_info': {
                'video_name': self.video_name,
                'start_time': self.session_start.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_frames': self.total_frames,
                'config_summary': {
                    'human_detector': self.config.human_detector,
                    'horse_detector': self.config.horse_detector,
                    'human_pose_estimator': self.config.human_pose_estimator,
                    'horse_pose_estimator': self.config.horse_pose_estimator,
                    'sam_model': getattr(self.config, 'sam_model', 'none'),
                    'reid_enabled': getattr(self.config, 'enable_reid_pipeline', False),
                    'reid_similarity_threshold': getattr(self.config, 'reid_similarity_threshold', 0.3),
                    'motion_distance_threshold': getattr(self.config, 'motion_distance_threshold', 200),
                    'enable_3d_poses': getattr(self.config, 'enable_3d_poses', True),
                    'depth_smoothing_factor': getattr(self.config, 'depth_smoothing_factor', 0.3)
                }
            },
            
            'detection_stats': {
                'total_detections': self.stats['total_detections'],
                'avg_detections_per_frame': self.stats['total_detections'] / max(1, self.total_frames),
                'confidence_stats': {
                    'human': self._calculate_confidence_stats(self.stats['confidence_distribution']['human']),
                    'horse': self._calculate_confidence_stats(self.stats['confidence_distribution']['horse'])
                }
            },
            
            'tracking_stats': {
                'total_tracks_created': self.stats['total_tracks_created'],
                'unique_human_tracks': len([t for t in self.stats['track_lifecycle'].values() if t['type'] == 'human']),
                'unique_horse_tracks': len([t for t in self.stats['track_lifecycle'].values() if t['type'] == 'horse']),
                'track_lifecycle_analysis': self._analyze_track_lifecycle(),
                'track_fragmentation_score': self._calculate_fragmentation_score()
            },
            
            'pose_3d_stats': {
                'total_3d_poses': len(self.pose_3d_data),
                'human_3d_quality': self._calculate_confidence_stats(self.stats['pose_3d_quality_distribution']['human']),
                'horse_3d_quality': self._calculate_confidence_stats(self.stats['pose_3d_quality_distribution']['horse']),
                'conversion_success_rate': self._calculate_3d_conversion_rate(),
                'depth_processing_quality': self._analyze_depth_processing_quality(),
                'geometric_feature_analysis': self._analyze_geometric_features()
            },
            
            'reid_stats': {
                'total_recoveries': self.stats['total_recoveries'],
                'recovery_rate': self.stats['total_recoveries'] / max(1, self.total_frames),
                'similarity_stats': self._calculate_similarity_stats(),
                'reid_3d_improvements': sum([d.get('reid_3d_improvements', 0) for d in self.reid_data])
            },
            
            'performance_stats': {
                'processing_times': self._analyze_processing_times(),
                'frame_rate_analysis': self._analyze_frame_rates(),
                'resource_usage': self._analyze_resource_usage(),
                'bottleneck_analysis': self._identify_bottlenecks()
            },
            
            'motion_stats': {
                'mean_motion_distance': float(np.mean(self.stats['motion_distances'])) if self.stats['motion_distances'] else 0,
                'motion_distance_std': float(np.std(self.stats['motion_distances'])) if self.stats['motion_distances'] else 0,
                'large_movements': len([d for d in self.stats['motion_distances'] if d > getattr(self.config, 'motion_distance_threshold', 200)])
            },
            
            'depth_stats': {
                'mean_depth_variance': float(np.mean(self.stats['depth_stats'])) if self.stats['depth_stats'] else 0,
                'depth_variance_std': float(np.std(self.stats['depth_stats'])) if self.stats['depth_stats'] else 0,
                'depth_processing_analysis': self._analyze_depth_processing_performance()
            },
            
            'performance_analysis': self._analyze_performance_issues()
        }
        
        return final_stats
    
    def _calculate_3d_conversion_rate(self) -> float:
        """Calculate 3D pose conversion success rate"""
        total_poses = len(self.pose_data)
        if total_poses == 0:
            return 0.0
        
        poses_with_3d = len([p for p in self.pose_data if p.get('has_3d', False)])
        return poses_with_3d / total_poses
    
    def _analyze_depth_processing_quality(self) -> Dict:
        """Analyze depth processing quality metrics"""
        if not self.depth_processing_data:
            return {'avg_uniformity': 0, 'avg_gradient': 0, 'avg_noise': 0}
        
        uniformities = []
        gradients = []
        noises = []
        
        for data in self.depth_processing_data:
            metrics = data.get('depth_quality_metrics', {})
            uniformities.append(metrics.get('uniformity', 0))
            gradients.append(metrics.get('gradient_magnitude', 0))
            noises.append(metrics.get('noise_level', 0))
        
        return {
            'avg_uniformity': float(np.mean(uniformities)) if uniformities else 0,
            'avg_gradient': float(np.mean(gradients)) if gradients else 0,
            'avg_noise': float(np.mean(noises)) if noises else 0,
            'uniformity_std': float(np.std(uniformities)) if uniformities else 0
        }
    
    def _analyze_geometric_features(self) -> Dict:
        """Analyze geometric features from 3D poses"""
        if not self.pose_3d_data:
            return {'avg_volume': 0, 'avg_compactness': 0, 'avg_surface_area': 0}
        
        volumes = []
        compactness_scores = []
        surface_areas = []
        
        for pose_data in self.pose_3d_data:
            geom_features = pose_data.get('geometric_features', {})
            volumes.append(geom_features.get('volume_estimate', 0))
            compactness_scores.append(geom_features.get('compactness', 0))
            surface_areas.append(geom_features.get('surface_area_estimate', 0))
        
        return {
            'avg_volume': float(np.mean(volumes)) if volumes else 0,
            'avg_compactness': float(np.mean(compactness_scores)) if compactness_scores else 0,
            'avg_surface_area': float(np.mean(surface_areas)) if surface_areas else 0,
            'volume_variance': float(np.var(volumes)) if volumes else 0
        }
    
    def _analyze_processing_times(self) -> Dict:
        """Analyze processing times for different components"""
        analysis = {}
        
        for component, times in self.stats['processing_times'].items():
            if times:
                analysis[component] = {
                    'mean_ms': float(np.mean(times) * 1000),
                    'std_ms': float(np.std(times) * 1000),
                    'min_ms': float(np.min(times) * 1000),
                    'max_ms': float(np.max(times) * 1000),
                    'median_ms': float(np.median(times) * 1000)
                }
            else:
                analysis[component] = {
                    'mean_ms': 0, 'std_ms': 0, 'min_ms': 0, 'max_ms': 0, 'median_ms': 0
                }
        
        return analysis
    
    def _analyze_frame_rates(self) -> Dict:
        """Analyze frame rate performance"""
        if not self.stats['frame_rates']:
            return {'avg_fps': 0, 'min_fps': 0, 'max_fps': 0}
        
        frame_rates = self.stats['frame_rates']
        return {
            'avg_fps': float(np.mean(frame_rates)),
            'min_fps': float(np.min(frame_rates)),
            'max_fps': float(np.max(frame_rates)),
            'std_fps': float(np.std(frame_rates)),
            'fps_stability': 1.0 - (np.std(frame_rates) / np.mean(frame_rates)) if np.mean(frame_rates) > 0 else 0
        }
    
    def _analyze_resource_usage(self) -> Dict:
        """Analyze resource usage patterns"""
        if not self.performance_data:
            return {'avg_gpu_memory': 0, 'avg_cpu_usage': 0, 'avg_memory_usage': 0}
        
        gpu_memory = [d['gpu_memory_mb'] for d in self.performance_data if d['gpu_memory_mb'] > 0]
        cpu_usage = [d['cpu_usage_percent'] for d in self.performance_data if d['cpu_usage_percent'] > 0]
        memory_usage = [d['memory_usage_mb'] for d in self.performance_data if d['memory_usage_mb'] > 0]
        
        return {
            'avg_gpu_memory_mb': float(np.mean(gpu_memory)) if gpu_memory else 0,
            'peak_gpu_memory_mb': float(np.max(gpu_memory)) if gpu_memory else 0,
            'avg_cpu_usage_percent': float(np.mean(cpu_usage)) if cpu_usage else 0,
            'avg_memory_usage_mb': float(np.mean(memory_usage)) if memory_usage else 0
        }
    
    def _identify_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        # Analyze processing times to find bottlenecks
        processing_times = self.stats['processing_times']
        total_times = {}
        
        for component, times in processing_times.items():
            if times:
                total_times[component] = np.sum(times)
        
        if total_times:
            max_component = max(total_times, key=total_times.get)
            max_time = total_times[max_component]
            total_time = sum(total_times.values())
            
            if max_time / total_time > 0.4:  # If one component takes >40% of time
                bottlenecks.append(f"Primary bottleneck: {max_component} ({max_time/total_time*100:.1f}% of processing time)")
        
        # Check frame rate stability
        if self.stats['frame_rates']:
            fps_std = np.std(self.stats['frame_rates'])
            fps_mean = np.mean(self.stats['frame_rates'])
            if fps_std / fps_mean > 0.3:  # High frame rate variance
                bottlenecks.append("Unstable frame rate detected - possible resource contention")
        
        return bottlenecks
    
    def _analyze_depth_processing_performance(self) -> Dict:
        """Analyze depth processing performance"""
        if not self.depth_processing_data:
            return {'avg_processing_time': 0, 'depth_map_quality': 0}
        
        processing_times = [d['processing_time_ms'] for d in self.depth_processing_data]
        depth_qualities = []
        
        for data in self.depth_processing_data:
            metrics = data.get('depth_quality_metrics', {})
            if metrics:
                quality = (metrics.get('uniformity', 0) + 
                          (1.0 / (1.0 + metrics.get('noise_level', 1)))) / 2
                depth_qualities.append(quality)
        
        return {
            'avg_processing_time_ms': float(np.mean(processing_times)) if processing_times else 0,
            'avg_depth_quality': float(np.mean(depth_qualities)) if depth_qualities else 0,
            'processing_time_std': float(np.std(processing_times)) if processing_times else 0
        }
    
    # Keep existing helper methods
    def _calculate_confidence_stats(self, confidence_list):
        """Calculate confidence statistics"""
        if not confidence_list:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}
        
        return {
            'mean': float(np.mean(confidence_list)),
            'std': float(np.std(confidence_list)),
            'min': float(np.min(confidence_list)),
            'max': float(np.max(confidence_list)),
            'count': len(confidence_list)
        }
    
    def _calculate_similarity_stats(self):
        """Calculate similarity statistics"""
        if not self.stats['similarity_scores']:
            return {'mean': 0, 'std': 0, 'above_threshold': 0}
        
        threshold = getattr(self.config, 'reid_similarity_threshold', 0.3)
        return {
            'mean': float(np.mean(self.stats['similarity_scores'])),
            'std': float(np.std(self.stats['similarity_scores'])),
            'above_threshold': len([s for s in self.stats['similarity_scores'] if s > threshold])
        }
    
    def _analyze_track_lifecycle(self):
        """Analyze track lifecycle patterns"""
        lifecycles = list(self.stats['track_lifecycle'].values())
        if not lifecycles:
            return {'avg_track_duration': 0, 'median_track_duration': 0, 'short_tracks_count': 0, 'long_tracks_count': 0}
        
        durations = [t['total_frames'] for t in lifecycles]
        
        return {
            'avg_track_duration': float(np.mean(durations)),
            'median_track_duration': float(np.median(durations)),
            'short_tracks_count': len([d for d in durations if d < 10]),
            'long_tracks_count': len([d for d in durations if d > 100]),
            'tracks_created_per_frame': len(lifecycles) / max(1, self.total_frames)
        }
    
    def _calculate_fragmentation_score(self):
        """Calculate track fragmentation score"""
        expected_horses = 11  # Default expected
        expected_jockeys = 11
        
        actual_horse_tracks = len([t for t in self.stats['track_lifecycle'].values() if t['type'] == 'horse'])
        actual_human_tracks = len([t for t in self.stats['track_lifecycle'].values() if t['type'] == 'human'])
        
        horse_fragmentation = actual_horse_tracks / max(1, expected_horses)
        human_fragmentation = actual_human_tracks / max(1, expected_jockeys)
        
        return {
            'horse_fragmentation': float(horse_fragmentation),
            'human_fragmentation': float(human_fragmentation),
            'overall_fragmentation': float((horse_fragmentation + human_fragmentation) / 2)
        }
    
    def _analyze_performance_issues(self):
        """Enhanced performance issue analysis including 3D pose metrics"""
        issues = []
        recommendations = []
        
        # Check fragmentation
        frag_score = self._calculate_fragmentation_score()
        if frag_score['horse_fragmentation'] > 5:
            issues.append(f"High horse track fragmentation: {frag_score['horse_fragmentation']:.1f}x expected")
            recommendations.append("Lower reid_similarity_threshold to 0.15 and increase lost_track_buffer to 50")
        
        if frag_score['human_fragmentation'] > 5:
            issues.append(f"High human track fragmentation: {frag_score['human_fragmentation']:.1f}x expected")
            recommendations.append("Lower minimum_matching_threshold to 0.3 for ByteTrack")
        
        # Check 3D pose conversion rate
        conversion_rate = self._calculate_3d_conversion_rate()
        if conversion_rate < 0.5:
            issues.append(f"Low 3D pose conversion rate: {conversion_rate:.2f}")
            recommendations.append("Improve depth map quality or lower depth smoothing threshold")
        
        # Check confidence levels
        if self.stats['confidence_distribution']['horse']:
            avg_horse_conf = np.mean(self.stats['confidence_distribution']['horse'])
            if avg_horse_conf < 0.6:
                issues.append(f"Low average horse detection confidence: {avg_horse_conf:.3f}")
                recommendations.append("Lower confidence_horse_detection threshold to 0.3")
        
        # Check 3D pose quality
        if self.stats['pose_3d_quality_distribution']['horse']:
            avg_3d_quality = np.mean(self.stats['pose_3d_quality_distribution']['horse'])
            if avg_3d_quality < 0.4:
                issues.append(f"Low 3D pose quality: {avg_3d_quality:.3f}")
                recommendations.append("Increase depth_smoothing_factor or improve depth estimation")
        
        # Check processing time bottlenecks
        bottlenecks = self._identify_bottlenecks()
        issues.extend(bottlenecks)
        
        if any("depth" in b.lower() for b in bottlenecks):
            recommendations.append("Consider switching to MobileSAM or disabling depth processing")
        
        return {
            'issues_detected': issues,
            'recommendations': recommendations,
            'severity': 'HIGH' if len(issues) > 4 else 'MEDIUM' if len(issues) > 2 else 'LOW'
        }
    
    def save_logs(self, output_path: str):
        """Save all collected logs including 3D pose metrics"""
        base_path = Path(output_path).parent
        video_name = self.video_name or "unknown_video"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        log_prefix = f"{video_name}_{timestamp}"
        
        # Calculate final statistics
        final_stats = self.calculate_final_statistics()
        
        # Save summary statistics (JSON)
        summary_path = base_path / f"{log_prefix}_3d_enhanced_summary.json"
        with open(summary_path, 'w') as f:
            cleaned_stats = round_floats(convert_numpy_types(final_stats))
            json.dump(cleaned_stats, f, indent=2)
        
        # Save detailed frame-by-frame data (CSV)
        csv_path = base_path / f"{log_prefix}_3d_detailed.csv"
        self._save_detailed_csv(csv_path)
        
        # Save 3D pose analysis (JSON)
        pose_3d_path = base_path / f"{log_prefix}_3d_pose_analysis.json"
        pose_3d_analysis = {
            'pose_3d_data': self.pose_3d_data,
            'depth_processing_data': self.depth_processing_data,
            'performance_data': self.performance_data
        }
        with open(pose_3d_path, 'w') as f:
            cleaned_3d_data = round_floats(convert_numpy_types(pose_3d_analysis))
            json.dump(cleaned_3d_data, f, separators=(',', ':'))
        
        # Save tracking analysis (JSON with compression)
        tracking_path = base_path / f"{log_prefix}_tracking_analysis.json"
        tracking_analysis = {
            'tracking_data': self.tracking_data,
            'reid_data': self.reid_data,
            'samurai_data': self.samurai_data,
            'recovery_data': self.recovery_data,
            'consolidation_data': self.consolidation_data
        }
        with open(tracking_path, 'w') as f:
            cleaned_tracking = round_floats(convert_numpy_types(tracking_analysis))
            json.dump(cleaned_tracking, f, separators=(',', ':'))
        
        # Save pose and depth data (JSON with compression)
        pose_depth_path = base_path / f"{log_prefix}_pose_depth.json"
        pose_depth_data = {
            'pose_data': self.pose_data,
            'depth_data': self.depth_data
        }
        with open(pose_depth_path, 'w') as f:
            cleaned_pose_depth = round_floats(convert_numpy_types(pose_depth_data))
            json.dump(cleaned_pose_depth, f, separators=(',', ':'))
        
        print(f"📊 Enhanced 3D debug logs saved:")
        print(f"   Summary: {summary_path}")
        print(f"   Detailed: {csv_path}")
        print(f"   3D Analysis: {pose_3d_path}")
        print(f"   Tracking: {tracking_path}")
        print(f"   Pose/Depth: {pose_depth_path}")
        
        # Print key findings
        self._print_key_findings(final_stats)
        
        return {
            'summary': summary_path,
            'detailed': csv_path,
            'pose_3d_analysis': pose_3d_path,
            'tracking': tracking_path,
            'pose_depth': pose_depth_path
        }
    
    def _save_detailed_csv(self, csv_path: Path):
        """Save detailed frame-by-frame data including 3D metrics"""
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'frame', 'timestamp', 'processing_time_ms',
                'human_detections', 'horse_detections', 'human_tracks', 'horse_tracks',
                'avg_human_conf', 'avg_horse_conf', 'reid_recoveries', 'consolidations',
                'active_samurai_tracks', 'lost_tracks', 'motion_predictions',
                'poses_3d_count', 'avg_3d_quality', 'depth_processing_time_ms',
                'reid_3d_improvements', 'frame_rate'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for frame_num in range(self.total_frames):
                # Aggregate data for this frame
                frame_detections = [d for d in self.detection_data if d['frame'] == frame_num]
                frame_tracking = [t for t in self.tracking_data if t['frame'] == frame_num]
                frame_samurai = [s for s in self.samurai_data if s['frame'] == frame_num]
                frame_recovery = [r for r in self.recovery_data if r['frame'] == frame_num]
                frame_consolidation = [c for c in self.consolidation_data if c['frame'] == frame_num]
                frame_pose_3d = [p for p in self.pose_3d_data if p['frame'] == frame_num]
                frame_depth_processing = [d for d in self.depth_processing_data if d['frame'] == frame_num]
                frame_reid = [r for r in self.reid_data if r['frame'] == frame_num]
                
                # Calculate aggregated values
                human_dets = [d for d in frame_detections if d['type'] == 'human']
                horse_dets = [d for d in frame_detections if d['type'] == 'horse']
                human_tracks = [t for t in frame_tracking if t['type'] == 'human']
                horse_tracks = [t for t in frame_tracking if t['type'] == 'horse']
                
                processing_time = 0
                if frame_num < len(self.frame_data):
                    processing_time = self.frame_data[frame_num].get('processing_time_ms', 0)
                
                # 3D pose metrics
                poses_3d_count = len(frame_pose_3d)
                avg_3d_quality = np.mean([p['confidence_3d'] for p in frame_pose_3d]) if frame_pose_3d else 0
                
                # Depth processing time
                depth_processing_time = frame_depth_processing[0].get('processing_time_ms', 0) if frame_depth_processing else 0
                
                # ReID 3D improvements
                reid_3d_improvements = frame_reid[0].get('reid_3d_improvements', 0) if frame_reid else 0
                
                # Frame rate
                frame_rate = 1000 / processing_time if processing_time > 0 else 0
                
                row = {
                    'frame': frame_num,
                    'timestamp': datetime.now().isoformat(),
                    'processing_time_ms': processing_time,
                    'human_detections': len(human_dets),
                    'horse_detections': len(horse_dets),
                    'human_tracks': len(human_tracks),
                    'horse_tracks': len(horse_tracks),
                    'avg_human_conf': np.mean([d['confidence'] for d in human_dets]) if human_dets else 0,
                    'avg_horse_conf': np.mean([d['confidence'] for d in horse_dets]) if horse_dets else 0,
                    'reid_recoveries': len(frame_recovery),
                    'consolidations': len(frame_consolidation),
                    'active_samurai_tracks': frame_samurai[0].get('active_tracks', 0) if frame_samurai else 0,
                    'lost_tracks': frame_samurai[0].get('lost_tracks', 0) if frame_samurai else 0,
                    'motion_predictions': len(frame_samurai[0].get('motion_predictions', {})) if frame_samurai else 0,
                    'poses_3d_count': poses_3d_count,
                    'avg_3d_quality': avg_3d_quality,
                    'depth_processing_time_ms': depth_processing_time,
                    'reid_3d_improvements': reid_3d_improvements,
                    'frame_rate': frame_rate
                }
                writer.writerow(row)
    
    def _print_key_findings(self, final_stats: Dict):
        """Print key findings including 3D pose metrics"""
        print(f"\n🔍 KEY FINDINGS for {self.video_name}:")
        print(f"📊 Track Fragmentation:")
        print(f"   Expected: 11 horses, 11 jockeys")
        print(f"   Actual: {final_stats['tracking_stats']['unique_horse_tracks']} horses, {final_stats['tracking_stats']['unique_human_tracks']} humans")
        print(f"   Fragmentation Score: {final_stats['tracking_stats']['track_fragmentation_score']['overall_fragmentation']:.1f}x")
        
        print(f"\n📈 Detection Quality:")
        horse_conf = final_stats['detection_stats']['confidence_stats']['horse']
        human_conf = final_stats['detection_stats']['confidence_stats']['human']
        print(f"   Horse confidence: {horse_conf['mean']:.3f} ± {horse_conf['std']:.3f}")
        print(f"   Human confidence: {human_conf['mean']:.3f} ± {human_conf['std']:.3f}")
        
        print(f"\n🎯 3D Pose Integration:")
        pose_3d_stats = final_stats['pose_3d_stats']
        print(f"   Total 3D poses: {pose_3d_stats['total_3d_poses']}")
        print(f"   Conversion rate: {pose_3d_stats['conversion_success_rate']:.2f}")
        print(f"   Horse 3D quality: {pose_3d_stats['horse_3d_quality']['mean']:.3f}")
        print(f"   Human 3D quality: {pose_3d_stats['human_3d_quality']['mean']:.3f}")
        
        print(f"\n🔄 Enhanced ReID Performance:")
        print(f"   Total recoveries: {final_stats['reid_stats']['total_recoveries']}")
        print(f"   3D improvements: {final_stats['reid_stats']['reid_3d_improvements']}")
        print(f"   Recovery rate: {final_stats['reid_stats']['recovery_rate']:.3f} per frame")
        
        print(f"\n⚡ Performance Analysis:")
        perf_stats = final_stats['performance_stats']
        print(f"   Avg frame rate: {perf_stats['frame_rate_analysis']['avg_fps']:.1f} FPS")
        print(f"   Primary bottlenecks: {len(perf_stats['bottleneck_analysis'])}")
        
        issues = final_stats['performance_analysis']['issues_detected']
        recommendations = final_stats['performance_analysis']['recommendations']
        
        if issues:
            print(f"\n⚠️ ISSUES DETECTED ({final_stats['performance_analysis']['severity']} severity):")
            for issue in issues[:3]:  # Show top 3 issues
                print(f"   - {issue}")
        
        if recommendations:
            print(f"\n💡 TOP RECOMMENDATIONS:")
            for rec in recommendations[:3]:  # Show top 3 recommendations
                print(f"   - {rec}")


# Helper functions for JSON serialization
def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def round_floats(obj, decimals=3):
    """Round floats to reduce file size"""
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, list):
        return [round_floats(item, decimals) for item in obj]
    elif isinstance(obj, dict):
        return {key: round_floats(value, decimals) for key, value in obj.items()}
    return obj