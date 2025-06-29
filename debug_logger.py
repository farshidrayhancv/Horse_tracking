"""
debug_logger.py - Complete Debug Logger for Horse Tracking System
Save this as debug_logger.py in your project directory
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
        self.depth_data = []
        self.consolidation_data = []
        self.recovery_data = []
        
        # Session metadata
        self.session_start = datetime.now()
        self.total_frames = 0
        self.video_name = None
        self.current_frame = 0
        self.frame_start_time = 0
        
        # Statistics collectors
        self.stats = {
            'total_detections': 0,
            'total_tracks_created': 0,
            'total_recoveries': 0,
            'total_consolidations': 0,
            'confidence_distribution': {'human': [], 'horse': []},
            'similarity_scores': [],
            'motion_distances': [],
            'depth_stats': [],
            'track_lifecycle': {}
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
    
    def log_detections(self, human_detections, horse_detections, detection_method: str):
        """Log detection information"""
        try:
            import supervision as sv
            
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
                        'area': float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
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
                        'area': float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
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
                'method': detection_method
            }
            self.detection_data.append(detection_data)
    
    def log_tracking_update(self, tracked_humans, tracked_horses, tracker_type: str = "ByteTrack"):
        """Log tracking updates and new track assignments"""
        try:
            import supervision as sv
            
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
                            'is_new_track': track_id not in self.stats['track_lifecycle']
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
                            'is_new_track': track_id not in self.stats['track_lifecycle']
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
                'tracker': tracker_type
            }
            self.tracking_data.append(tracking_data)
    
    def log_reid_process(self, reid_features, similarity_scores: Dict = None, assignments: Dict = None, untracked_count: int = 0):
        """Log ReID pipeline processing"""
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
            'untracked_detections': untracked_count
        }
        
        # Log similarity scores
        for track_id, score in similarity_scores.items():
            reid_frame_data['similarity_scores'][str(track_id)] = float(score)
            self.stats['similarity_scores'].append(float(score))
        
        self.reid_data.append(reid_frame_data)
    
    def log_samurai_tracking(self, tracking_info: Dict, recovered_tracks: List = None):
        """Log SAMURAI tracking information"""
        if recovered_tracks is None:
            recovered_tracks = []
            
        samurai_data = {
            'frame': self.current_frame,
            'active_tracks': tracking_info.get('active_tracks', 0),
            'lost_tracks': tracking_info.get('lost_tracks', 0),
            'memory_efficiency': tracking_info.get('memory_efficiency', 0),
            'total_reassignments': tracking_info.get('total_reassignments', 0),
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
    
    def log_pose_estimation(self, human_poses: List, horse_poses: List):
        """Log pose estimation results"""
        # Log human poses
        for i, pose in enumerate(human_poses):
            pose_data = {
                'frame': self.current_frame,
                'type': 'human',
                'pose_id': i,
                'track_id': pose.get('track_id', -1),
                'method': 'ViTPose',
                'confidence': pose.get('confidence', 0.0),
                'num_keypoints': 0,
                'valid_keypoints': 0
            }
            
            # Count valid keypoints
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
            
            self.pose_data.append(pose_data)
        
        # Log horse poses
        for i, pose in enumerate(horse_poses):
            pose_data = {
                'frame': self.current_frame,
                'type': 'horse',
                'pose_id': i,
                'track_id': pose.get('track_id', -1),
                'method': pose.get('method', 'unknown'),
                'confidence': pose.get('confidence', 0.0),
                'num_keypoints': 0,
                'valid_keypoints': 0
            }
            
            # Count valid keypoints for horse poses
            if 'keypoints' in pose:
                keypoints = pose['keypoints']
                if isinstance(keypoints, np.ndarray):
                    pose_data['num_keypoints'] = len(keypoints)
                    if keypoints.shape[-1] >= 3:
                        # SuperAnimal format: [x, y, confidence]
                        valid_count = np.sum((keypoints[:, 0] != -1) & (keypoints[:, 2] > 0))
                        pose_data['valid_keypoints'] = int(valid_count)
            
            self.pose_data.append(pose_data)
    
    def log_depth_processing(self, depth_map: np.ndarray, depth_stats: List):
        """Log depth map processing information"""
        depth_frame_data = {
            'frame': self.current_frame,
            'depth_map_shape': list(depth_map.shape) if depth_map is not None else [0, 0],
            'depth_range': {},
            'per_detection_stats': []
        }
        
        if depth_map is not None and depth_map.size > 0:
            depth_frame_data['depth_range'] = {
                'min': float(np.min(depth_map)),
                'max': float(np.max(depth_map)),
                'mean': float(np.mean(depth_map)),
                'std': float(np.std(depth_map))
            }
        
        # Log per-detection depth statistics
        for i, stats in enumerate(depth_stats):
            detection_depth = {
                'detection_id': i,
                'mask_area': stats.get('area', 0),
                'sam_confidence': stats.get('confidence', 0.0),
                'mask_quality': stats.get('mask_quality', 0.0),
                'depth_variance': stats.get('depth_variance', 0.0)
            }
            depth_frame_data['per_detection_stats'].append(detection_depth)
            self.stats['depth_stats'].append(stats.get('depth_variance', 0.0))
        
        self.depth_data.append(depth_frame_data)
    
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
    
    def calculate_final_statistics(self):
        """Calculate comprehensive statistics"""
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
                    'motion_distance_threshold': getattr(self.config, 'motion_distance_threshold', 200)
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
            
            'reid_stats': {
                'total_recoveries': self.stats['total_recoveries'],
                'recovery_rate': self.stats['total_recoveries'] / max(1, self.total_frames),
                'similarity_stats': self._calculate_similarity_stats()
            },
            
            'motion_stats': {
                'mean_motion_distance': float(np.mean(self.stats['motion_distances'])) if self.stats['motion_distances'] else 0,
                'motion_distance_std': float(np.std(self.stats['motion_distances'])) if self.stats['motion_distances'] else 0,
                'large_movements': len([d for d in self.stats['motion_distances'] if d > getattr(self.config, 'motion_distance_threshold', 200)])
            },
            
            'depth_stats': {
                'mean_depth_variance': float(np.mean(self.stats['depth_stats'])) if self.stats['depth_stats'] else 0,
                'depth_variance_std': float(np.std(self.stats['depth_stats'])) if self.stats['depth_stats'] else 0
            },
            
            'performance_analysis': self._analyze_performance_issues()
        }
        
        return final_stats
    
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
        expected_horses = 9  # Default expected
        expected_jockeys = 9
        
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
        """Analyze potential performance issues"""
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
        
        # Check confidence levels
        if self.stats['confidence_distribution']['horse']:
            avg_horse_conf = np.mean(self.stats['confidence_distribution']['horse'])
            if avg_horse_conf < 0.6:
                issues.append(f"Low average horse detection confidence: {avg_horse_conf:.3f}")
                recommendations.append("Lower confidence_horse_detection threshold to 0.3")
        
        # Check similarity scores
        if self.stats['similarity_scores']:
            high_similarity_count = len([s for s in self.stats['similarity_scores'] if s > 0.5])
            if high_similarity_count < len(self.stats['similarity_scores']) * 0.3:
                issues.append("Few high similarity ReID matches")
                recommendations.append("Check feature extraction quality and lower similarity thresholds")
        
        # Check motion distances
        if self.stats['motion_distances']:
            large_motion_count = len([d for d in self.stats['motion_distances'] if d > 200])
            if large_motion_count > len(self.stats['motion_distances']) * 0.5:
                issues.append("Many large motion distances detected")
                recommendations.append("Increase motion_distance_threshold to 400 for broadcast footage")
        
        return {
            'issues_detected': issues,
            'recommendations': recommendations,
            'severity': 'HIGH' if len(issues) > 3 else 'MEDIUM' if len(issues) > 1 else 'LOW'
        }
    
    def save_logs(self, output_path: str):
        """Save all collected logs to files"""
        base_path = Path(output_path).parent
        video_name = self.video_name or "unknown_video"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        log_prefix = f"{video_name}_{timestamp}"
        
        # Calculate final statistics
        final_stats = self.calculate_final_statistics()
        
        # Save summary statistics (JSON)
        summary_path = base_path / f"{log_prefix}_summary.json"
        with open(summary_path, 'w') as f:
            cleaned_stats = round_floats(convert_numpy_types(final_stats))
            json.dump(cleaned_stats, f, indent=2)
        
        # Save detailed frame-by-frame data (CSV)
        csv_path = base_path / f"{log_prefix}_detailed.csv"
        self._save_detailed_csv(csv_path)
        
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
            json.dump(cleaned_tracking, f, separators=(',', ':'))  # No indentation for smaller size
        
        # Save pose and depth data (JSON with compression)
        pose_depth_path = base_path / f"{log_prefix}_pose_depth.json"
        pose_depth_data = {
            'pose_data': self.pose_data,
            'depth_data': self.depth_data
        }
        with open(pose_depth_path, 'w') as f:
            cleaned_pose_depth = round_floats(convert_numpy_types(pose_depth_data))
            json.dump(cleaned_pose_depth, f, separators=(',', ':'))  # No indentation for smaller size
        
        print(f"📊 Debug logs saved:")
        print(f"   Summary: {summary_path}")
        print(f"   Detailed: {csv_path}")
        print(f"   Tracking: {tracking_path}")
        print(f"   Pose/Depth: {pose_depth_path}")
        
        # Print key findings
        self._print_key_findings(final_stats)
        
        return {
            'summary': summary_path,
            'detailed': csv_path,
            'tracking': tracking_path,
            'pose_depth': pose_depth_path
        }
    
    def _save_detailed_csv(self, csv_path: Path):
        """Save detailed frame-by-frame data as CSV"""
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'frame', 'timestamp', 'processing_time_ms',
                'human_detections', 'horse_detections', 'human_tracks', 'horse_tracks',
                'avg_human_conf', 'avg_horse_conf', 'reid_recoveries', 'consolidations',
                'active_samurai_tracks', 'lost_tracks', 'motion_predictions'
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
                
                # Calculate aggregated values
                human_dets = [d for d in frame_detections if d['type'] == 'human']
                horse_dets = [d for d in frame_detections if d['type'] == 'horse']
                human_tracks = [t for t in frame_tracking if t['type'] == 'human']
                horse_tracks = [t for t in frame_tracking if t['type'] == 'horse']
                
                processing_time = 0
                if frame_num < len(self.frame_data):
                    processing_time = self.frame_data[frame_num].get('processing_time_ms', 0)
                
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
                    'motion_predictions': len(frame_samurai[0].get('motion_predictions', {})) if frame_samurai else 0
                }
                writer.writerow(row)
    
    def _print_key_findings(self, final_stats: Dict):
        """Print key findings and recommendations"""
        print(f"\n🔍 KEY FINDINGS for {self.video_name}:")
        print(f"📊 Track Fragmentation:")
        print(f"   Expected: 9 horses, 9 jockeys")
        print(f"   Actual: {final_stats['tracking_stats']['unique_horse_tracks']} horses, {final_stats['tracking_stats']['unique_human_tracks']} humans")
        print(f"   Fragmentation Score: {final_stats['tracking_stats']['track_fragmentation_score']['overall_fragmentation']:.1f}x")
        
        print(f"\n📈 Detection Quality:")
        horse_conf = final_stats['detection_stats']['confidence_stats']['horse']
        human_conf = final_stats['detection_stats']['confidence_stats']['human']
        print(f"   Horse confidence: {horse_conf['mean']:.3f} ± {horse_conf['std']:.3f}")
        print(f"   Human confidence: {human_conf['mean']:.3f} ± {human_conf['std']:.3f}")
        
        print(f"\n🔄 ReID Performance:")
        print(f"   Total recoveries: {final_stats['reid_stats']['total_recoveries']}")
        print(f"   Recovery rate: {final_stats['reid_stats']['recovery_rate']:.3f} per frame")
        print(f"   Avg similarity: {final_stats['reid_stats']['similarity_stats']['mean']:.3f}")
        
        print(f"\n🎯 Motion Analysis:")
        print(f"   Avg motion distance: {final_stats['motion_stats']['mean_motion_distance']:.1f} pixels")
        print(f"   Large movements: {final_stats['motion_stats']['large_movements']}")
        
        issues = final_stats['performance_analysis']['issues_detected']
        recommendations = final_stats['performance_analysis']['recommendations']
        
        if issues:
            print(f"\n⚠️ ISSUES DETECTED ({final_stats['performance_analysis']['severity']} severity):")
            for issue in issues:
                print(f"   - {issue}")
        
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"   - {rec}")


# Additional helper functions for integration
def log_untracked_detections(logger, detections):
    """Helper to count untracked detections"""
    untracked_count = 0
    if hasattr(detections, 'tracker_id'):
        untracked_count = sum(1 for tid in detections.tracker_id if tid < 0)
    return untracked_count

def extract_similarity_scores_from_reid(reid_pipeline):
    """Helper to extract similarity scores from ReID pipeline"""
    similarity_scores = {}
    try:
        if hasattr(reid_pipeline, 'memory') and hasattr(reid_pipeline.memory, 'feature_memory'):
            # Extract recent similarity calculations
            for track_id in reid_pipeline.memory.feature_memory.keys():
                # This would need to be implemented in reid_pipeline to expose scores
                similarity_scores[track_id] = 0.5  # Placeholder
    except:
        pass
    return similarity_scores

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