"""
Simplified Debug Logger for Racer Tracking System
Focused on compound racer entities with pose and ReID tracking
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
        self.pose_data = []
        self.reid_data = []
        
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
            'confidence_distribution': [],
            'track_lifecycle': {},
            'reid_reassignments': 0,
            'pose_estimations': 0
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
    
    def log_detections(self, human_detections, racer_detections, detection_method: str):
        """Log racer detection information (simplified for compound entities)"""
        try:
            import supervision as sv
            
            # Log racer detections
            if sv and hasattr(racer_detections, 'xyxy') and len(racer_detections) > 0:
                for i, (bbox, conf) in enumerate(zip(racer_detections.xyxy, racer_detections.confidence)):
                    detection_data = {
                        'frame': self.current_frame,
                        'type': 'racer',
                        'detection_id': i,
                        'bbox': bbox.tolist(),
                        'confidence': float(conf),
                        'method': detection_method,
                        'area': float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                    }
                    self.detection_data.append(detection_data)
                    self.stats['confidence_distribution'].append(float(conf))
            
            racer_count = len(racer_detections) if hasattr(racer_detections, '__len__') else 0
            self.stats['total_detections'] += racer_count
            
        except Exception as e:
            # Fallback logging
            racer_count = len(racer_detections) if hasattr(racer_detections, '__len__') else 0
            
            detection_data = {
                'frame': self.current_frame,
                'error': f"Detection logging failed: {e}",
                'racer_count': racer_count,
                'method': detection_method
            }
            self.detection_data.append(detection_data)
    
    def log_tracking_update(self, human_tracks, racer_tracks, tracker_type: str = "DeepOCSORT"):
        """Log tracking updates for compound racers"""
        try:
            import supervision as sv
            
            if not sv:
                return
            
            # Log racer tracking
            if hasattr(racer_tracks, 'xyxy') and len(racer_tracks) > 0:
                if hasattr(racer_tracks, 'tracker_id'):
                    for i, (bbox, track_id, conf) in enumerate(zip(
                        racer_tracks.xyxy, racer_tracks.tracker_id, racer_tracks.confidence)):
                        
                        tracking_data = {
                            'frame': self.current_frame,
                            'type': 'racer',
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
                                'type': 'racer'
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
    
    def log_pose_estimation(self, human_poses: List, racer_poses: List):
        """Log pose estimation results for compound racers"""
        
        # Log compound racer poses
        for i, pose in enumerate(racer_poses):
            pose_data = {
                'frame': self.current_frame,
                'type': 'racer',
                'pose_id': i,
                'compound_box_index': pose.get('compound_box_index', -1),
                'method': pose.get('method', 'unknown'),
                'confidence': pose.get('confidence', 0.0),
                'num_keypoints': 0,
                'valid_keypoints': 0
            }
            
            # Count valid keypoints
            if 'keypoints' in pose:
                keypoints = pose['keypoints']
                if isinstance(keypoints, np.ndarray):
                    pose_data['num_keypoints'] = len(keypoints)
                    if keypoints.shape[-1] >= 3:
                        # Format: [x, y, confidence]
                        valid_count = np.sum((keypoints[:, 0] != -1) & (keypoints[:, 2] > 0))
                        pose_data['valid_keypoints'] = int(valid_count)
                    elif keypoints.shape[-1] == 2:
                        # Format: [x, y]
                        valid_count = np.sum((keypoints[:, 0] != -1) & (keypoints[:, 1] != -1))
                        pose_data['valid_keypoints'] = int(valid_count)
            
            self.pose_data.append(pose_data)
            self.stats['pose_estimations'] += 1
    
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
        """Calculate comprehensive statistics for simplified system"""
        final_stats = {
            'session_info': {
                'video_name': self.video_name,
                'start_time': self.session_start.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_frames': self.total_frames,
                'system_type': 'Simplified Racer Tracking',
                'config_summary': {
                    'roboflow_model': getattr(self.config, 'roboflow_model_id', 'not_set'),
                    'pose_estimator': getattr(self.config, 'horse_pose_estimator', 'not_set'),
                    'tracker_type': getattr(self.config, 'tracker_type', 'deepocsort'),
                    'reid_enabled': getattr(self.config, 'enable_reid', False),
                    'roboflow_confidence': getattr(self.config, 'roboflow_confidence', 0.5)
                }
            },
            
            'detection_stats': {
                'total_racer_detections': self.stats['total_detections'],
                'avg_detections_per_frame': self.stats['total_detections'] / max(1, self.total_frames),
                'confidence_stats': self._calculate_confidence_stats(self.stats['confidence_distribution'])
            },
            
            'tracking_stats': {
                'total_tracks_created': self.stats['total_tracks_created'],
                'unique_racer_tracks': len([t for t in self.stats['track_lifecycle'].values() if t['type'] == 'racer']),
                'track_lifecycle_analysis': self._analyze_track_lifecycle()
            },
            
            'pose_estimation_stats': {
                'total_pose_estimations': self.stats['pose_estimations'],
                'avg_poses_per_frame': self.stats['pose_estimations'] / max(1, self.total_frames),
                'method_distribution': self._analyze_pose_methods()
            },
            
            'system_performance': self._analyze_system_performance()
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
    
    def _analyze_track_lifecycle(self):
        """Analyze track lifecycle patterns"""
        lifecycles = list(self.stats['track_lifecycle'].values())
        if not lifecycles:
            return {
                'avg_track_duration': 0, 
                'median_track_duration': 0, 
                'short_tracks_count': 0, 
                'stable_tracks_count': 0,
                'tracks_per_frame': 0
            }
        
        durations = [t['total_frames'] for t in lifecycles]
        
        return {
            'avg_track_duration': float(np.mean(durations)),
            'median_track_duration': float(np.median(durations)),
            'short_tracks_count': len([d for d in durations if d < 10]),
            'stable_tracks_count': len([d for d in durations if d > 50]),
            'tracks_per_frame': len(lifecycles) / max(1, self.total_frames)
        }
    
    def _analyze_pose_methods(self):
        """Analyze pose estimation method distribution"""
        racer_poses = [p for p in self.pose_data if p['type'] == 'racer']
        if not racer_poses:
            return {'superanimal': 0, 'vitpose': 0, 'unknown': 0}
        
        method_counts = {}
        for pose in racer_poses:
            method = pose.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        return method_counts
    
    def _analyze_system_performance(self):
        """Analyze system performance and potential issues"""
        issues = []
        recommendations = []
        
        # Check detection rate
        avg_detections = self.stats['total_detections'] / max(1, self.total_frames)
        expected_racers = getattr(self.config, 'max_racers', 10)
        
        if avg_detections < expected_racers * 0.5:
            issues.append(f"Low detection rate: {avg_detections:.1f} avg vs {expected_racers} expected")
            recommendations.append("Lower roboflow_confidence threshold or check video quality")
        
        # Check tracking stability
        if self.stats['total_tracks_created'] > 0:
            avg_duration = np.mean([t['total_frames'] for t in self.stats['track_lifecycle'].values()])
            if avg_duration < 30:
                issues.append(f"Short track duration: {avg_duration:.1f} frames")
                recommendations.append("Increase tracker max_age or check detection consistency")
        
        # Check pose estimation rate
        pose_rate = self.stats['pose_estimations'] / max(1, self.total_frames)
        if pose_rate < avg_detections * 0.8:
            issues.append(f"Low pose estimation rate: {pose_rate:.1f} vs {avg_detections:.1f} detections")
            recommendations.append("Lower pose confidence thresholds")
        
        return {
            'issues_detected': issues,
            'recommendations': recommendations,
            'severity': 'HIGH' if len(issues) > 3 else 'MEDIUM' if len(issues) > 1 else 'LOW'
        }
    
    def save_logs(self, output_path: str):
        """Save all logs to files"""
        base_path = Path(output_path).parent
        video_name = self.video_name or "unknown_video"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        log_prefix = f"{video_name}_{timestamp}"
        
        # Calculate final statistics
        final_stats = self.calculate_final_statistics()
        
        # Save summary statistics
        summary_path = base_path / f"{log_prefix}_summary.json"
        with open(summary_path, 'w') as f:
            cleaned_stats = self._clean_for_json(final_stats)
            json.dump(cleaned_stats, f, indent=2)
        
        # Save detailed CSV
        csv_path = base_path / f"{log_prefix}_detailed.csv"
        self._save_detailed_csv(csv_path)
        
        print(f"📊 Debug logs saved:")
        print(f"   Summary: {summary_path}")
        print(f"   Detailed: {csv_path}")
        
        # Print key findings
        self._print_key_findings(final_stats)
        
        return {
            'summary': summary_path,
            'detailed': csv_path
        }
    
    def _clean_for_json(self, obj):
        """Clean object for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._clean_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        return obj
    
    def _save_detailed_csv(self, csv_path: Path):
        """Save detailed frame-by-frame data"""
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'frame', 'processing_time_ms', 'racer_detections',
                'racer_tracks', 'avg_confidence', 'pose_estimations'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for frame_num in range(self.total_frames):
                frame_detections = [d for d in self.detection_data if d['frame'] == frame_num]
                frame_tracking = [t for t in self.tracking_data if t['frame'] == frame_num]
                frame_poses = [p for p in self.pose_data if p['frame'] == frame_num]
                
                racer_dets = [d for d in frame_detections if d['type'] == 'racer']
                racer_tracks = [t for t in frame_tracking if t['type'] == 'racer']
                racer_poses = len([p for p in frame_poses if p['type'] == 'racer'])
                
                processing_time = 0
                if frame_num < len(self.frame_data):
                    processing_time = self.frame_data[frame_num].get('processing_time_ms', 0)
                
                row = {
                    'frame': frame_num,
                    'processing_time_ms': processing_time,
                    'racer_detections': len(racer_dets),
                    'racer_tracks': len(racer_tracks),
                    'avg_confidence': np.mean([d['confidence'] for d in racer_dets]) if racer_dets else 0,
                    'pose_estimations': racer_poses
                }
                writer.writerow(row)
    
    def _print_key_findings(self, final_stats: Dict):
        """Print key findings"""
        print(f"\n🔍 KEY FINDINGS for {self.video_name}:")
        
        detection_stats = final_stats['detection_stats']
        tracking_stats = final_stats['tracking_stats']
        pose_stats = final_stats['pose_estimation_stats']
        
        print(f"📊 Detection Performance:")
        print(f"   Total detections: {detection_stats['total_racer_detections']}")
        print(f"   Avg per frame: {detection_stats['avg_detections_per_frame']:.1f}")
        print(f"   Confidence: {detection_stats['confidence_stats']['mean']:.3f}")
        
        print(f"\n🎯 Tracking Performance:")
        print(f"   Unique tracks: {tracking_stats['unique_racer_tracks']}")
        print(f"   Avg duration: {tracking_stats['track_lifecycle_analysis']['avg_track_duration']:.1f} frames")
        
        print(f"\n🦴 Pose Estimation:")
        print(f"   Total poses: {pose_stats['total_pose_estimations']}")
        print(f"   Methods: {pose_stats['method_distribution']}")
        
        performance = final_stats['system_performance']
        if performance['issues_detected']:
            print(f"\n⚠️ ISSUES ({performance['severity']}):")
            for issue in performance['issues_detected']:
                print(f"   - {issue}")
            
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in performance['recommendations']:
                print(f"   - {rec}")