#!/usr/bin/env python3
"""
FIXED Multi-Tracker Racing ReID Testing System - Python 3.9 Compatible

Changes:
- Fixed BoostTrack proximity_thresh parameter error
- Fixed ByteTrack device parameter error
- Hardcoded racing-optimized configurations
- Removed dynamic config loading

Quick Start:
1. pip install boxmot supervision
2. python fixed_racing_reid.py config.yaml [frames]
3. Results saved to multi_tracker_racing_results/

Key Benefits:
- Working BoostTrack and ByteTrack initialization
- Simplified parameter management
- Focused racing persistence metrics
"""

import cv2
import numpy as np
import torch
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

# Import checks
try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    print("❌ Supervision not available")

try:
    from boxmot import DeepOcSort, BoostTrack, ByteTrack
    BOXMOT_AVAILABLE = True
    AVAILABLE_TRACKERS = ['bytetrack', 'deepocsort', 'boosttrack']
except ImportError:
    BOXMOT_AVAILABLE = False
    AVAILABLE_TRACKERS = []
    print("❌ BoxMOT trackers not available")

try:
    from inference import get_model
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False
    print("❌ Roboflow not available")

from config import Config

class SystematicCacheManager:
    """Cache management with proper empty detection handling"""
    
    def __init__(self, cache_dir: str = "detection_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        print(f"🔥 Cache manager initialized: {self.cache_dir}")
    
    def _generate_cache_key(self, video_path: str, model_id: str, confidence: float, max_frames: int) -> str:
        video_name = Path(video_path).stem
        clean_model_id = model_id.replace('/', '_').replace('-', '_').replace(':', '_')
        return f"{video_name}_{clean_model_id}_{confidence}_{max_frames}"
    
    def get_cache_path(self, video_path: str, model_id: str, confidence: float, max_frames: int) -> Path:
        cache_key = self._generate_cache_key(video_path, model_id, confidence, max_frames)
        return self.cache_dir / f"{cache_key}.json"
    
    def cache_exists(self, video_path: str, model_id: str, confidence: float, max_frames: int) -> bool:
        cache_path = self.get_cache_path(video_path, model_id, confidence, max_frames)
        if not cache_path.exists():
            return False
        try:
            cache_size = cache_path.stat().st_size
            return cache_size > 1000
        except:
            return False
    
    def load_detections(self, video_path: str, model_id: str, confidence: float, max_frames: int) -> Optional[List[Dict]]:
        """Load detections with proper empty handling"""
        cache_path = self.get_cache_path(video_path, model_id, confidence, max_frames)
        
        if not cache_path.exists():
            print(f"⚪ No cache found: {cache_path.name}")
            return None
        
        try:
            print(f"🔄 Loading cache: {cache_path.name}")
            
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            if 'detections' not in cache_data:
                print(f"❌ Invalid cache structure")
                return None
            
            detections_data = []
            empty_frames = 0
            
            for frame_data in cache_data['detections']:
                try:
                    frame_num = frame_data['frame_num']
                    detection_info = frame_data['detections']
                    
                    # Handle empty detections properly
                    if (not detection_info.get('xyxy') or 
                        len(detection_info['xyxy']) == 0 or
                        not detection_info.get('confidence')):
                        
                        detections = sv.Detections.empty()
                        empty_frames += 1
                        
                    else:
                        xyxy_data = detection_info['xyxy']
                        conf_data = detection_info['confidence']
                        
                        if len(xyxy_data) != len(conf_data):
                            print(f"⚠️ Data mismatch in frame {frame_num}")
                            detections = sv.Detections.empty()
                            continue
                        
                        xyxy = np.array(xyxy_data, dtype=np.float32)
                        confidence_arr = np.array(conf_data, dtype=np.float32)
                        
                        if (detection_info.get('class_id') and 
                            len(detection_info['class_id']) == len(xyxy_data)):
                            class_id = np.array(detection_info['class_id'], dtype=np.int32)
                        else:
                            class_id = np.ones(len(xyxy_data), dtype=np.int32)
                        
                        detections = sv.Detections(
                            xyxy=xyxy,
                            confidence=confidence_arr,
                            class_id=class_id
                        )
                    
                    detections_data.append({
                        'frame_num': frame_num,
                        'frame': None,
                        'detections': detections
                    })
                    
                except Exception as e:
                    print(f"⚠️ Error processing frame {frame_data.get('frame_num', '?')}: {e}")
                    detections_data.append({
                        'frame_num': frame_data.get('frame_num', len(detections_data)),
                        'frame': None,
                        'detections': sv.Detections.empty()
                    })
                    continue
            
            if not detections_data:
                print(f"❌ No valid detection data in cache")
                return None
            
            cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
            print(f"✅ Cache loaded successfully:")
            print(f"   Total frames: {len(detections_data)}")
            print(f"   Empty frames: {empty_frames}")
            print(f"   Cache size: {cache_size_mb:.1f} MB")
            
            return detections_data
            
        except Exception as e:
            print(f"❌ Cache loading failed: {e}")
            return None
    
    def save_detections(self, detections_data: List[Dict], video_path: str, model_id: str, 
                       confidence: float, max_frames: int) -> bool:
        """Save detections with robust error handling"""
        cache_path = self.get_cache_path(video_path, model_id, confidence, max_frames)
        
        try:
            serializable_data = []
            for data in detections_data:
                frame_data = {
                    'frame_num': data['frame_num'],
                    'detections': {
                        'xyxy': data['detections'].xyxy.tolist(),
                        'confidence': data['detections'].confidence.tolist(),
                        'class_id': data['detections'].class_id.tolist() if hasattr(data['detections'], 'class_id') and data['detections'].class_id is not None else []
                    }
                }
                serializable_data.append(frame_data)
            
            cache_data = {
                'metadata': {
                    'video_path': str(video_path),
                    'video_name': Path(video_path).name,
                    'model_id': model_id,
                    'confidence': confidence,
                    'max_frames': max_frames,
                    'cached_at': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'total_frames': len(serializable_data),
                    'device_used': 'cuda' if torch.cuda.is_available() else 'cpu'
                },
                'detections': serializable_data
            }
            
            temp_path = cache_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            temp_path.rename(cache_path)
            
            cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
            print(f"✅ Cache saved: {cache_path.name} ({cache_size_mb:.1f} MB)")
            return True
            
        except Exception as e:
            print(f"❌ Cache save failed: {e}")
            return False

class SystematicWeightManager:
    """Weight management without arbitrary size filtering"""
    
    def __init__(self, weights_dir: str = "reid_test_weights"):
        self.weights_dir = Path(weights_dir)
        
        # FIXED: Updated for .pt extensions
        self.weight_info = {
            'osnet_x0_25_msmt17': {
                'name': 'OSNet-x0.25-MSMT17 (Baseline)', 
                'filename': 'osnet_x0_25_msmt17.pt',
                'description': 'Lightweight baseline',
                'horse_suitability': 'BASELINE',
                'category': 'OSNet-Lightweight'
            },
            'osnet_ibn_x1_0_imagenet': {
                'name': 'OSNet-IBN-x1.0-ImageNet (TOP)',
                'filename': 'osnet_ibn_x1_0_imagenet.pt', 
                'description': 'ImageNet + IBN viewpoint robustness',
                'horse_suitability': 'EXCELLENT',
                'category': 'OSNet-IBN-ImageNet'
            },
            'osnet_ibn_x1_0_msmt17': {
                'name': 'OSNet-IBN-x1.0-MSMT17',
                'filename': 'osnet_ibn_x1_0_msmt17.pt',
                'description': 'IBN normalization + diversity',
                'horse_suitability': 'VERY GOOD', 
                'category': 'OSNet-IBN'
            },
            'osnet_x1_0_msmt17': {
                'name': 'OSNet-x1.0-MSMT17',
                'filename': 'osnet_x1_0_msmt17.pt',
                'description': 'Full capacity OSNet',
                'horse_suitability': 'GOOD',
                'category': 'OSNet-Full'
            }
        }
    
    def get_available_weights(self) -> List[str]:
        """Test ALL existing weight files - no size filtering"""
        available = []
        print(f"\n🔍 Systematic weight scanning in {self.weights_dir}:")
        
        for weight_key, weight_info in self.weight_info.items():
            weight_path = self.weights_dir / weight_info['filename']
            
            if weight_path.exists():
                try:
                    size_mb = weight_path.stat().st_size / (1024 * 1024)
                    is_valid = self._validate_pytorch_model(weight_path)
                    
                    if is_valid:
                        available.append(weight_key)
                        print(f"   ✅ {weight_key}: {size_mb:.1f} MB - {weight_info['horse_suitability']}")
                    else:
                        print(f"   ❌ {weight_key}: {size_mb:.1f} MB - CORRUPTED/INVALID")
                        
                except Exception as e:
                    print(f"   ❌ {weight_key}: Error - {e}")
            else:
                print(f"   ⚪ {weight_key}: Not found - {weight_info['filename']}")
        
        print(f"\n📊 Weight Summary: {len(available)} valid weights found")
        return available
    
    def _validate_pytorch_model(self, weight_path: Path) -> bool:
        """Validate PyTorch model file integrity"""
        try:
            size_bytes = weight_path.stat().st_size
            if size_bytes < 1024 * 1024:  # Less than 1MB
                return False
            
            try:
                checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
                
                if not isinstance(checkpoint, dict):
                    return False
                
                if len(checkpoint) < 5:
                    return False
                
                key_strings = str(list(checkpoint.keys())[:20]).lower()
                model_indicators = ['weight', 'bias', 'running', 'num_batches', 'backbone', 'classifier']
                
                has_model_structure = any(indicator in key_strings for indicator in model_indicators)
                
                if not has_model_structure:
                    for key, value in checkpoint.items():
                        if key in ['state_dict', 'model', 'net'] and isinstance(value, dict):
                            nested_keys = str(list(value.keys())[:10]).lower()
                            has_model_structure = any(indicator in nested_keys for indicator in model_indicators)
                            if has_model_structure:
                                break
                
                return has_model_structure
                
            except Exception:
                return False
                
        except Exception:
            return False
    
    def get_weight_path(self, weight_key: str) -> Path:
        return self.weights_dir / self.weight_info[weight_key]['filename']
    
    def get_weight_info(self, weight_key: str) -> Dict:
        return self.weight_info[weight_key]

class MultiTrackerFactory:
    """FIXED multi-tracker factory with correct BoxMOT parameters"""
    
    @staticmethod
    def create_tracker(tracker_type: str, weight_path: Path, device: str) -> Optional[object]:
        """Create tracker with FIXED parameters"""
        if not BOXMOT_AVAILABLE or not weight_path.exists():
            return None
        
        try:
            if device == 'cuda' and torch.cuda.is_available():
                device_str = 'cuda:0'
            else:
                device_str = 'cpu'
            
            print(f"🔥 Creating {tracker_type} on {device_str}")
            
            if tracker_type == 'deepocsort':
                tracker = DeepOcSort(
                    reid_weights=weight_path,  # REVERT: Use Path object directly
                    device=device_str,
                    half=True,
                    max_age=180,
                    min_hits=2,
                    det_thresh=0.5,
                    iou_threshold=0.15,
                    w_association_emb=0.98,
                    embedding_off=False,
                )
            elif tracker_type == 'boosttrack':
                tracker = BoostTrack(
                    reid_weights=weight_path,  # REVERT: Use Path object directly
                    device=device_str,
                    half=True,
                    max_age=180,
                    min_hits=2,
                    det_thresh=0.5,
                    iou_threshold=0.15,
                )
            elif tracker_type == 'bytetrack':
                tracker = ByteTrack(
                    track_thresh=0.6,
                    track_buffer=180,
                    match_thresh=0.8,
                    frame_rate=30
                )
            else:
                print(f"❌ Unknown tracker type: {tracker_type}")
                return None
            
            print(f"✅ {tracker_type} tracker created successfully")
            return tracker
            
        except Exception as e:
            print(f"❌ {tracker_type} tracker creation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

class RacingPersistenceMetrics:
    """Racing metrics focused on 1500+ frame persistence"""
    
    def __init__(self, expected_horses: int = 8, race_frames: int = 2400):
        self.expected_horses = expected_horses
        self.race_frames = race_frames
        self.persistence_threshold = 1500
        self.reset()
    
    def reset(self):
        self.tracks_seen = set()
        self.tracks_first_frame = {}
        self.tracks_last_frame = {}
        self.tracks_frame_count = defaultdict(int)
        self.tracks_active_frames = defaultdict(set)
        self.id_switch_count = 0
        self.total_detections = 0
        self.total_frames = 0
        self.last_positions = {}
        self.position_jumps = []
        
        self.persistent_tracks = set()
        self.complete_race_tracks = set()
        self.stable_tracks = set()
        
    def update_frame(self, frame_num: int, tracked_detections):
        self.total_frames = frame_num + 1
        
        if not SUPERVISION_AVAILABLE or len(tracked_detections) == 0:
            return
        
        self.total_detections += len(tracked_detections)
        current_positions = {}
        
        for bbox, track_id in zip(tracked_detections.xyxy, tracked_detections.tracker_id):
            if track_id < 0:
                continue
            
            self.tracks_seen.add(track_id)
            self.tracks_frame_count[track_id] += 1
            self.tracks_active_frames[track_id].add(frame_num)
            
            if track_id not in self.tracks_first_frame:
                self.tracks_first_frame[track_id] = frame_num
            self.tracks_last_frame[track_id] = frame_num
            
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            current_positions[track_id] = (center_x, center_y)
            
            if track_id in self.last_positions:
                last_x, last_y = self.last_positions[track_id]
                distance = np.sqrt((center_x - last_x)**2 + (center_y - last_y)**2)
                
                if distance > 250:
                    self.id_switch_count += 1
                    self.position_jumps.append({
                        'frame': frame_num,
                        'track_id': track_id,
                        'distance': distance
                    })
        
        self.last_positions = current_positions
    
    def calculate_racing_metrics(self) -> Dict:
        if not self.tracks_seen:
            return {'error': 'No tracks detected during testing'}
        
        early_phase = int(self.race_frames * 0.1)
        late_phase = int(self.race_frames * 0.9)
        
        track_lifespans = []
        persistent_count = 0
        complete_race_count = 0
        stable_count = 0
        
        for track_id in self.tracks_seen:
            first_frame = self.tracks_first_frame[track_id]
            last_frame = self.tracks_last_frame[track_id]
            frame_count = self.tracks_frame_count[track_id]
            active_frames = self.tracks_active_frames[track_id]
            
            track_lifespans.append(frame_count)
            
            if frame_count >= self.persistence_threshold:
                persistent_count += 1
                self.persistent_tracks.add(track_id)
            
            if first_frame <= early_phase and last_frame >= late_phase:
                complete_race_count += 1
                self.complete_race_tracks.add(track_id)
            
            if len(active_frames) > 0:
                sorted_frames = sorted(active_frames)
                expected_frames = sorted_frames[-1] - sorted_frames[0] + 1
                continuity_ratio = len(active_frames) / expected_frames
                
                if continuity_ratio > 0.90:
                    stable_count += 1
                    self.stable_tracks.add(track_id)
        
        racing_metrics = {
            'total_frames_processed': self.total_frames,
            'expected_race_frames': self.race_frames,
            'total_detections': self.total_detections,
            'unique_track_ids': len(self.tracks_seen),
            'expected_horses': self.expected_horses,
            
            'persistent_tracks_1500plus': persistent_count,
            'complete_race_tracks': complete_race_count,
            'stable_continuous_tracks': stable_count,
            
            'racing_persistence_rate': persistent_count / len(self.tracks_seen),
            'complete_race_rate': complete_race_count / len(self.tracks_seen),
            'horse_tracking_efficiency': persistent_count / min(self.expected_horses, len(self.tracks_seen)),
            
            'max_track_lifespan': max(track_lifespans) if track_lifespans else 0,
            'avg_track_lifespan': np.mean(track_lifespans) if track_lifespans else 0,
            'median_track_lifespan': np.median(track_lifespans) if track_lifespans else 0,
            
            'id_switches_detected': self.id_switch_count,
            'position_jumps_count': len(self.position_jumps),
            'id_consistency_score': max(0, 1.0 - (self.id_switch_count / len(self.tracks_seen))),
            'switches_per_track': self.id_switch_count / len(self.tracks_seen) if self.tracks_seen else 0,
            
            'avg_detections_per_frame': self.total_detections / max(1, self.total_frames),
            'track_density': len(self.tracks_seen) / max(1, self.total_frames)
        }
        
        return racing_metrics

def process_video_with_systematic_pipeline(config: Config, max_frames: int, cache_manager: SystematicCacheManager) -> List[Dict]:
    """Systematic video processing with GPU utilization"""
    
    cached_detections = cache_manager.load_detections(
        config.video_path, config.roboflow_model_id, config.roboflow_confidence, max_frames
    )
    
    if cached_detections:
        print(f"✅ Using cached detections")
        return cached_detections
    
    print(f"🔥 Running GPU detection pipeline...")
    
    if not ROBOFLOW_AVAILABLE:
        print(f"❌ Roboflow not available")
        return []
    
    try:
        model = get_model(model_id=config.roboflow_model_id, api_key=config.roboflow_api_key)
        print(f"✅ Roboflow model loaded: {config.roboflow_model_id}")
        
        if torch.cuda.is_available():
            print(f"🔥 GPU available: {torch.cuda.get_device_name()}")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return []
    
    cap = cv2.VideoCapture(config.video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {config.video_path}")
        return []
    
    detection_data = []
    start_time = time.time()
    
    print(f"📹 Processing {max_frames} frames...")
    
    for frame_count in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"   End of video at frame {frame_count}")
            break
        
        try:
            results = model.infer(frame, confidence=config.roboflow_confidence)[0]
            detections = sv.Detections.from_inference(results)
            
            detection_data.append({
                'frame_num': frame_count,
                'frame': None,
                'detections': detections
            })
            
            if frame_count % 250 == 0 and frame_count > 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"   Frame {frame_count}: {len(detections)} detections ({fps:.1f} FPS)")
                
        except Exception as e:
            print(f"❌ Detection failed at frame {frame_count}: {e}")
            break
    
    cap.release()
    
    if detection_data:
        processing_time = time.time() - start_time
        avg_fps = len(detection_data) / processing_time
        
        print(f"✅ Detection complete: {len(detection_data)} frames, {avg_fps:.1f} FPS")
        
        cache_manager.save_detections(
            detection_data, config.video_path, config.roboflow_model_id,
            config.roboflow_confidence, max_frames
        )
    
    return detection_data

def test_tracker_weight_combination(tracker_type: str, weight_key: str, weight_path: Path, 
                                  detection_data: List[Dict], config: Config) -> Dict:
    """Test tracker-weight combination with racing metrics"""
    
    print(f"\n🏇 Testing: {tracker_type} + {weight_key}")
    
    tracker = MultiTrackerFactory.create_tracker(tracker_type, weight_path, config.device)
    if not tracker:
        return {'error': f'Failed to create {tracker_type} with {weight_key}'}
    
    metrics = RacingPersistenceMetrics(expected_horses=8, race_frames=2400)
    test_start_time = time.time()
    
    cap = cv2.VideoCapture(config.video_path)
    if not cap.isOpened():
        return {'error': 'Cannot access video'}
    
    print(f"   ⚡ Processing {len(detection_data)} frames...")
    
    for i, data in enumerate(detection_data):
        frame_num = data['frame_num']
        detections = data['detections']
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue
        
        try:
            if len(detections) > 0:
                xyxy = np.array(detections.xyxy, dtype=np.float64)
                confidence = np.array([float(x) for x in detections.confidence], dtype=np.float64)
                class_ids = np.zeros(len(detections), dtype=np.float64)
                
                dets_np = np.column_stack((xyxy, confidence, class_ids)).astype(np.float64)
                
                tracks = tracker.update(dets_np, frame)
                
                if tracks is not None and len(tracks) > 0:
                    tracked_detections = sv.Detections(
                        xyxy=tracks[:, :4],
                        confidence=tracks[:, 5] if tracks.shape[1] > 5 else confidence[:len(tracks)],
                        class_id=tracks[:, 6].astype(np.int32) if tracks.shape[1] > 6 else class_ids[:len(tracks)].astype(np.int32),
                        tracker_id=tracks[:, 4].astype(np.int32)
                    )
                else:
                    tracked_detections = sv.Detections.empty()
            else:
                tracked_detections = sv.Detections.empty()
                
        except Exception as e:
            print(f"   ⚠️ Tracking error at frame {frame_num}: {e}")
            tracked_detections = sv.Detections.empty()
        
        metrics.update_frame(frame_num, tracked_detections)
        
        if i % 400 == 0 and i > 0:
            active_tracks = len(set(tracked_detections.tracker_id[tracked_detections.tracker_id >= 0])) if len(tracked_detections) > 0 else 0
            print(f"   Frame {frame_num}: {len(tracked_detections)} tracked, {active_tracks} active IDs")
    
    cap.release()
    
    test_duration = time.time() - test_start_time
    racing_metrics = metrics.calculate_racing_metrics()
    
    if 'error' not in racing_metrics:
        racing_metrics.update({
            'tracker_type': tracker_type,
            'weight_key': weight_key,
            'test_duration_seconds': test_duration,
            'frames_tested': len(detection_data),
            'processing_fps': len(detection_data) / test_duration if test_duration > 0 else 0
        })
    
    return racing_metrics

def print_combination_results(tracker_type: str, weight_key: str, results: Dict, weight_info: Dict):
    """Print tracker-weight combination results"""
    if 'error' in results:
        print(f"   ❌ FAILED: {results['error']}")
        return
    
    print(f"   🏇 {tracker_type.upper()} + {weight_info['name']} - Racing Results:")
    print(f"      Racing Persistent (1500+ frames): {results['persistent_tracks_1500plus']}")
    print(f"      Complete Race Tracks: {results['complete_race_tracks']}")
    print(f"      Stable Continuous Tracks: {results['stable_continuous_tracks']}")
    print(f"      Horse Tracking Efficiency: {results['horse_tracking_efficiency']:.3f}")
    print(f"      Max Track Lifespan: {results['max_track_lifespan']:.0f} frames")
    print(f"      ID Consistency Score: {results['id_consistency_score']:.3f}")
    print(f"      Processing FPS: {results['processing_fps']:.1f}")

def generate_racing_analysis(test_results: Dict, weight_manager: SystematicWeightManager) -> Dict:
    """Generate comprehensive racing analysis"""
    valid_results = {k: v for k, v in test_results.items() if 'error' not in v}
    
    if not valid_results:
        print(f"\n❌ No valid test results")
        return {}
    
    print(f"\n🏆 MULTI-TRACKER RACING ANALYSIS")
    print("=" * 80)
    
    print(f"\n🏇 Performance Rankings:")
    print(f"{'Tracker + Weight':<40} {'1500+ Persist':<12} {'Efficiency':<10} {'Max Life':<8}")
    print("-" * 75)
    
    sorted_results = sorted(valid_results.items(), 
                           key=lambda x: x[1]['persistent_tracks_1500plus'], reverse=True)
    
    for combo_key, metrics in sorted_results:
        tracker_type = metrics['tracker_type']
        weight_key = metrics['weight_key']
        combo_name = f"{tracker_type.upper()} + {weight_key}"
        short_combo = combo_name[:39]
        print(f"{short_combo:<40} {metrics['persistent_tracks_1500plus']:<12} "
              f"{metrics['horse_tracking_efficiency']:<10.3f} "
              f"{metrics['max_track_lifespan']:<8.0f}")
    
    print(f"\n🥇 Performance Champions:")
    
    best_persistence = max(valid_results.items(), key=lambda x: x[1]['persistent_tracks_1500plus'])
    print(f"   🎯 Most Persistent: {best_persistence[1]['tracker_type'].upper()} + {best_persistence[1]['weight_key']} ({best_persistence[1]['persistent_tracks_1500plus']} tracks)")
    
    best_efficiency = max(valid_results.items(), key=lambda x: x[1]['horse_tracking_efficiency'])
    print(f"   🏇 Best Efficiency: {best_efficiency[1]['tracker_type'].upper()} + {best_efficiency[1]['weight_key']} ({best_efficiency[1]['horse_tracking_efficiency']:.3f})")
    
    best_consistency = max(valid_results.items(), key=lambda x: x[1]['id_consistency_score'])
    print(f"   ✅ Best Consistency: {best_consistency[1]['tracker_type'].upper()} + {best_consistency[1]['weight_key']} ({best_consistency[1]['id_consistency_score']:.3f})")
    
    # Overall recommendation
    racing_scores = {}
    for combo_key, metrics in valid_results.items():
        score = (
            metrics['persistent_tracks_1500plus'] * 0.4 +
            metrics['horse_tracking_efficiency'] * 0.25 +
            metrics['id_consistency_score'] * 0.2 +
            (metrics['max_track_lifespan'] / 2400) * 0.15
        )
        racing_scores[combo_key] = score
    
    best_overall = max(racing_scores.items(), key=lambda x: x[1])
    best_metrics = valid_results[best_overall[0]]
    
    print(f"\n🏆 CHAMPION RECOMMENDATION:")
    print(f"   🥇 Best Combination: {best_metrics['tracker_type'].upper()} + {best_metrics['weight_key']}")
    print(f"   📊 Racing Score: {best_overall[1]:.3f}")
    print(f"   🐴 Persistent Tracks: {best_metrics['persistent_tracks_1500plus']}/8 horses")
    print(f"   ⚡ Processing Speed: {best_metrics['processing_fps']:.1f} FPS")
    
    return {
        'champion_combination': {
            'tracker': best_metrics['tracker_type'],
            'weight': best_metrics['weight_key'],
            'combo_key': best_overall[0]
        },
        'racing_scores': racing_scores
    }

def main():
    """Main execution"""
    import sys
    
    if len(sys.argv) not in [2, 3]:
        print("🏇 FIXED MULTI-TRACKER RACING REID SYSTEM")
        print("Usage: python fixed_racing_reid.py config.yaml [test_frames]")
        print("\nFixes Applied:")
        print("- BoostTrack proximity_thresh parameter removed")
        print("- ByteTrack device parameter removed")
        print("- Hardcoded racing configurations")
        print("- Python 3.9 compatible")
        print("\nTest Coverage: DeepOCSORT, BoostTrack, ByteTrack")
        sys.exit(1)
    
    config_file = sys.argv[1]
    test_frames = int(sys.argv[2]) if len(sys.argv) == 3 else 1800
    
    if not Path(config_file).exists():
        print(f"❌ Config file not found: {config_file}")
        sys.exit(1)
    
    config = Config(config_file)
    
    print(f"🏇 FIXED MULTI-TRACKER RACING REID SYSTEM")
    print(f"   Video: {config.video_path}")
    print(f"   Test Frames: {test_frames}")
    print(f"   Device: {config.device}")
    
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name()}")
    
    if not all([SUPERVISION_AVAILABLE, BOXMOT_AVAILABLE]):
        print(f"❌ Missing dependencies")
        sys.exit(1)
    
    print(f"✅ Available trackers: {', '.join(AVAILABLE_TRACKERS)}")
    
    cache_manager = SystematicCacheManager()
    weight_manager = SystematicWeightManager()
    
    available_weights = weight_manager.get_available_weights()
    
    if not available_weights:
        print(f"\n❌ NO VALID WEIGHTS FOUND")
        sys.exit(1)
    
    print(f"\n🎯 Testing {len(AVAILABLE_TRACKERS)} trackers with {len(available_weights)} weights")
    
    detection_data = process_video_with_systematic_pipeline(config, test_frames, cache_manager)
    
    if not detection_data:
        print(f"❌ No detection data")
        sys.exit(1)
    
    print(f"\n🏇 SYSTEMATIC TESTING")
    print("=" * 60)
    
    test_results = {}
    combination_count = 0
    total_combinations = len(AVAILABLE_TRACKERS) * len(available_weights)
    
    for tracker_type in AVAILABLE_TRACKERS:
        print(f"\n🔥 TESTING: {tracker_type.upper()}")
        
        for weight_key in available_weights:
            combination_count += 1
            combo_key = f"{tracker_type}+{weight_key}"
            
            print(f"\n🧪 Combination {combination_count}/{total_combinations}: {tracker_type.upper()} + {weight_key}")
            
            weight_path = weight_manager.get_weight_path(weight_key)
            weight_info = weight_manager.get_weight_info(weight_key)
            
            racing_results = test_tracker_weight_combination(
                tracker_type, weight_key, weight_path, detection_data, config
            )
            test_results[combo_key] = racing_results
            
            print_combination_results(tracker_type, weight_key, racing_results, weight_info)
    
    analysis = generate_racing_analysis(test_results, weight_manager)
    
    # Save results
    output_dir = Path("multi_tracker_racing_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_name = Path(config.video_path).stem
    results_file = output_dir / f"{video_name}_fixed_multi_tracker_{timestamp}.json"
    
    complete_results = {
        'test_configuration': {
            'video_path': config.video_path,
            'test_frames': test_frames,
            'trackers_tested': AVAILABLE_TRACKERS,
            'weights_tested': available_weights,
            'timestamp': timestamp,
            'fixes_applied': [
                'removed_boosttrack_proximity_thresh',
                'removed_bytetrack_device_param',
                'hardcoded_racing_configs'
            ]
        },
        'combination_test_results': test_results,
        'racing_analysis': analysis
    }
    
    with open(results_file, 'w') as f:
        json.dump(complete_results, f, indent=2, default=str)
    
    print(f"\n📊 Results saved: {results_file}")
    
    if 'champion_combination' in analysis:
        champion = analysis['champion_combination']
        print(f"\n✅ TESTING COMPLETE!")
        print(f"🏆 CHAMPION: {champion['tracker'].upper()} + {champion['weight']}")
        print(f"🔧 Use this combination for horse racing tracking")

if __name__ == "__main__":
    main()