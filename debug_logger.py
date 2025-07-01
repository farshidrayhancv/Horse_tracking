"""
debug_logger.py - Deep OC-SORT Horse Tracking Debug Logger
Focused logging for horse-only tracking with OCR classification
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
        self.ocr_classification_data = []
        
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
            'confidence_distribution': {'horse': []},
            'track_lifecycle': {},
            'ocr_classifications': 0,
            'successful_ocr_identifications': 0
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
        """Log horse detection information only"""
        try:
            import supervision as sv
            
            # Horse detections only (humans not used)
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
            
            horse_count = len(horse_detections) if hasattr(horse_detections, '__len__') else 0
            self.stats['total_detections'] += horse_count
            
        except Exception as e:
            # Fallback logging
            horse_count = len(horse_detections) if hasattr(horse_detections, '__len__') else 0
            
            detection_data = {
                'frame': self.current_frame,
                'error': f"Detection logging failed: {e}",
                'horse_count': horse_count,
                'method': detection_method
            }
            self.detection_data.append(detection_data)
    
    def log_tracking_update(self, human_tracks, horse_tracks, tracker_type: str = "DeepOCSORT"):
        """Log Deep OC-SORT tracking updates for horses only"""
        try:
            import supervision as sv
            
            if not sv:
                return
            
            # Log horse tracking only
            if hasattr(horse_tracks, 'xyxy') and len(horse_tracks) > 0:
                if hasattr(horse_tracks, 'tracker_id'):
                    for i, (bbox, track_id, conf) in enumerate(zip(
                        horse_tracks.xyxy, horse_tracks.tracker_id, horse_tracks.confidence)):
                        
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
    
    def log_ocr_classification(self, classifications: List, successful_identifications: int):
        """Log SigLIP OCR classification results"""
        ocr_frame_data = {
            'frame': self.current_frame,
            'total_classifications': len(classifications),
            'successful_identifications': successful_identifications,
            'classification_results': []
        }
        
        for i, class_id in enumerate(classifications):
            classification_result = {
                'detection_index': i,
                'detected_number': int(class_id) if class_id >= 0 else None,
                'success': class_id >= 0
            }
            ocr_frame_data['classification_results'].append(classification_result)
        
        self.ocr_classification_data.append(ocr_frame_data)
        self.stats['ocr_classifications'] += len(classifications)
        self.stats['successful_ocr_identifications'] += successful_identifications
    
    def log_pose_estimation(self, human_poses: List, horse_poses: List):
        """Log horse pose estimation results only"""
        
        # Log horse poses only
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
                    elif keypoints.shape[-1] == 2:
                        # ViTPose format: [x, y]
                        valid_count = np.sum((keypoints[:, 0] != -1) & (keypoints[:, 1] != -1))
                        pose_data['valid_keypoints'] = int(valid_count)
            
            self.pose_data.append(pose_data)
    
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
        """Calculate comprehensive statistics for horse-only tracking"""
        final_stats = {
            'session_info': {
                'video_name': self.video_name,
                'start_time': self.session_start.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_frames': self.total_frames,
                'config_summary': {
                    'horse_detector': self.config.horse_detector,
                    'horse_pose_estimator': self.config.horse_pose_estimator,
                    'tracker_type': getattr(self.config, 'tracker_type', 'deepocsort'),
                    'siglip_ocr_enabled': getattr(self.config, 'enable_siglip_classification', False),
                    'siglip_confidence_threshold': getattr(self.config, 'siglip_confidence_threshold', 0.8),
                    'confidence_horse_detection': getattr(self.config, 'confidence_horse_detection', 0.7)
                }
            },
            
            'detection_stats': {
                'total_horse_detections': self.stats['total_detections'],
                'avg_detections_per_frame': self.stats['total_detections'] / max(1, self.total_frames),
                'horse_confidence_stats': self._calculate_confidence_stats(self.stats['confidence_distribution']['horse'])
            },
            
            'tracking_stats': {
                'total_tracks_created': self.stats['total_tracks_created'],
                'unique_horse_tracks': len([t for t in self.stats['track_lifecycle'].values() if t['type'] == 'horse']),
                'track_lifecycle_analysis': self._analyze_track_lifecycle(),
                'track_fragmentation_score': self._calculate_fragmentation_score()
            },
            
            'ocr_classification_stats': {
                'total_classifications': self.stats['ocr_classifications'],
                'successful_identifications': self.stats['successful_ocr_identifications'],
                'ocr_success_rate': self.stats['successful_ocr_identifications'] / max(1, self.stats['ocr_classifications']),
                'classification_rate_per_frame': self.stats['ocr_classifications'] / max(1, self.total_frames)
            },
            
            'pose_estimation_stats': {
                'total_horse_poses': len([p for p in self.pose_data if p['type'] == 'horse']),
                'avg_poses_per_frame': len([p for p in self.pose_data if p['type'] == 'horse']) / max(1, self.total_frames),
                'method_distribution': self._analyze_pose_methods()
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
        """Calculate track fragmentation score for horses"""
        expected_horses = getattr(self.config, 'max_horses', 9)  # Default expected
        
        actual_horse_tracks = len([t for t in self.stats['track_lifecycle'].values() if t['type'] == 'horse'])
        horse_fragmentation = actual_horse_tracks / max(1, expected_horses)
        
        return {
            'horse_fragmentation': float(horse_fragmentation),
            'expected_horses': expected_horses,
            'actual_horse_tracks': actual_horse_tracks
        }
    
    def _analyze_pose_methods(self):
        """Analyze pose estimation method distribution"""
        horse_poses = [p for p in self.pose_data if p['type'] == 'horse']
        if not horse_poses:
            return {'superanimal': 0, 'vitpose': 0, 'unknown': 0}
        
        method_counts = {}
        for pose in horse_poses:
            method = pose.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        return method_counts
    
    def _analyze_performance_issues(self):
        """Analyze potential performance issues for horse tracking"""
        issues = []
        recommendations = []
        
        # Check fragmentation
        frag_score = self._calculate_fragmentation_score()
        if frag_score['horse_fragmentation'] > 3:
            issues.append(f"High horse track fragmentation: {frag_score['horse_fragmentation']:.1f}x expected")
            recommendations.append("Lower detection threshold or increase Deep OC-SORT max_age parameter")
        
        # Check confidence levels
        if self.stats['confidence_distribution']['horse']:
            avg_horse_conf = np.mean(self.stats['confidence_distribution']['horse'])
            if avg_horse_conf < 0.6:
                issues.append(f"Low average horse detection confidence: {avg_horse_conf:.3f}")
                recommendations.append("Lower confidence_horse_detection threshold to improve detection rate")
        
        # Check OCR success rate
        if self.stats['ocr_classifications'] > 0:
            ocr_success_rate = self.stats['successful_ocr_identifications'] / self.stats['ocr_classifications']
            if ocr_success_rate < 0.5:
                issues.append(f"Low OCR classification success rate: {ocr_success_rate:.3f}")
                recommendations.append("Check video quality or lower siglip_confidence_threshold")
        
        # Check track stability
        if self.stats['total_tracks_created'] > 0:
            avg_track_duration = np.mean([t['total_frames'] for t in self.stats['track_lifecycle'].values()])
            if avg_track_duration < 20:
                issues.append(f"Short average track duration: {avg_track_duration:.1f} frames")
                recommendations.append("Increase Deep OC-SORT max_age and min_hits parameters")
        
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
        
        # Save tracking analysis (JSON)
        tracking_path = base_path / f"{log_prefix}_tracking_analysis.json"
        tracking_analysis = {
            'tracking_data': self.tracking_data,
            'ocr_classification_data': self.ocr_classification_data,
            'pose_data': self.pose_data
        }
        with open(tracking_path, 'w') as f:
            cleaned_tracking = round_floats(convert_numpy_types(tracking_analysis))
            json.dump(cleaned_tracking, f, separators=(',', ':'))
        
        print(f"📊 Debug logs saved:")
        print(f"   Summary: {summary_path}")
        print(f"   Detailed: {csv_path}")
        print(f"   Tracking: {tracking_path}")
        
        # Print key findings
        self._print_key_findings(final_stats)
        
        return {
            'summary': summary_path,
            'detailed': csv_path,
            'tracking': tracking_path
        }
    
    def _save_detailed_csv(self, csv_path: Path):
        """Save detailed frame-by-frame data as CSV"""
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'frame', 'timestamp', 'processing_time_ms',
                'horse_detections', 'horse_tracks', 'avg_horse_conf', 
                'ocr_classifications', 'successful_ocr_ids', 'horse_poses',
                'pose_methods'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for frame_num in range(self.total_frames):
                # Aggregate data for this frame
                frame_detections = [d for d in self.detection_data if d['frame'] == frame_num]
                frame_tracking = [t for t in self.tracking_data if t['frame'] == frame_num]
                frame_ocr = [o for o in self.ocr_classification_data if o['frame'] == frame_num]
                frame_poses = [p for p in self.pose_data if p['frame'] == frame_num]
                
                # Calculate aggregated values
                horse_dets = [d for d in frame_detections if d['type'] == 'horse']
                horse_tracks = [t for t in frame_tracking if t['type'] == 'horse']
                horse_poses_count = len([p for p in frame_poses if p['type'] == 'horse'])
                
                processing_time = 0
                if frame_num < len(self.frame_data):
                    processing_time = self.frame_data[frame_num].get('processing_time_ms', 0)
                
                # OCR data
                ocr_classifications = frame_ocr[0].get('total_classifications', 0) if frame_ocr else 0
                successful_ocr = frame_ocr[0].get('successful_identifications', 0) if frame_ocr else 0
                
                # Pose methods
                pose_methods = {}
                for pose in frame_poses:
                    method = pose.get('method', 'unknown')
                    pose_methods[method] = pose_methods.get(method, 0) + 1
                
                row = {
                    'frame': frame_num,
                    'timestamp': datetime.now().isoformat(),
                    'processing_time_ms': processing_time,
                    'horse_detections': len(horse_dets),
                    'horse_tracks': len(horse_tracks),
                    'avg_horse_conf': np.mean([d['confidence'] for d in horse_dets]) if horse_dets else 0,
                    'ocr_classifications': ocr_classifications,
                    'successful_ocr_ids': successful_ocr,
                    'horse_poses': horse_poses_count,
                    'pose_methods': str(pose_methods) if pose_methods else '{}'
                }
                writer.writerow(row)
    
    def _print_key_findings(self, final_stats: Dict):
        """Print key findings and recommendations"""
        print(f"\n🔍 KEY FINDINGS for {self.video_name}:")
        print(f"📊 Track Analysis:")
        expected_horses = final_stats['tracking_stats']['track_fragmentation_score']['expected_horses']
        actual_tracks = final_stats['tracking_stats']['unique_horse_tracks']
        fragmentation = final_stats['tracking_stats']['track_fragmentation_score']['horse_fragmentation']
        print(f"   Expected: {expected_horses} horses")
        print(f"   Actual tracks: {actual_tracks}")
        print(f"   Fragmentation: {fragmentation:.1f}x")
        
        print(f"\n📈 Detection Quality:")
        horse_conf = final_stats['detection_stats']['horse_confidence_stats']
        print(f"   Horse confidence: {horse_conf['mean']:.3f} ± {horse_conf['std']:.3f}")
        print(f"   Total detections: {horse_conf['count']}")
        
        print(f"\n🔢 OCR Classification:")
        ocr_stats = final_stats['ocr_classification_stats']
        print(f"   Total classifications: {ocr_stats['total_classifications']}")
        print(f"   Successful IDs: {ocr_stats['successful_identifications']}")
        print(f"   Success rate: {ocr_stats['ocr_success_rate']:.3f}")
        
        print(f"\n🎯 Pose Estimation:")
        pose_stats = final_stats['pose_estimation_stats']
        print(f"   Total horse poses: {pose_stats['total_horse_poses']}")
        print(f"   Method distribution: {pose_stats['method_distribution']}")
        
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