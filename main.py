#!/usr/bin/env python3
"""
Enhanced Horse Tracking System with BoostTrack + SigLIP Classification
Simplified ReID using reference image approach
"""

import cv2
import numpy as np
import torch
import warnings
import re
import time
from pathlib import Path

# Local imports
from config import Config
from models import SuperAnimalQuadruped
from detectors import DetectionManager
from pose_estimators import PoseEstimationManager
from visualizers import Visualizer
from siglip_classifier import SigLIPClassifier
from debug_logger import TrackingDebugLogger

# Suppress warnings but keep errors and status prints
warnings.filterwarnings("ignore")

# Check dependencies
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Install tqdm for progress bars: pip install tqdm")

try:
    from transformers import AutoProcessor, VitPoseForPoseEstimation, RTDetrForObjectDetection
    VITPOSE_AVAILABLE = True
    print("✓ ViTPose available")
except ImportError:
    VITPOSE_AVAILABLE = False
    print("⚠️ ViTPose not available - install with: pip install transformers torch")

try:
    from dlclibrary import download_huggingface_model
    SUPERANIMAL_AVAILABLE = True
    print("✓ SuperAnimal available")
except ImportError:
    SUPERANIMAL_AVAILABLE = False
    print("⚠️ DLClibrary not available - install with: pip install dlclibrary")

try:
    import supervision as sv
    from supervision import ByteTrack
    print("✓ Supervision available")
except ImportError:
    print("❌ Supervision not available - install with: pip install supervision")
    sv = None

try:
    from boxmot import BoostTrack, DeepOcSort
    BOOSTTRACK_AVAILABLE = True
    DEEPOCSORT_AVAILABLE = True
    print("✓ BoostTrack available")
    print("✓ DeepOcSort available")
except ImportError:
    BOOSTTRACK_AVAILABLE = False
    DEEPOCSORT_AVAILABLE = False
    print("⚠️ BoxMOT trackers not available - install with: pip install boxmot")


class HybridPoseSystem:
    def __init__(self, video_path: str, config: Config):
        self.video_path = Path(video_path)
        self.config = config
        
        # Initialize debug logger
        self.debug_logger = TrackingDebugLogger(self.config)
        self.debug_logger.set_video_name(str(self.video_path))
        
        # Parse expected counts from filename
        self.expected_horses, self.expected_jockeys = self.parse_filename_counts()
        
        # Setup video
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        
        if self.total_frames <= 0:
            print("⚠️ Warning: Frame count not available, will process until end of video")
            self.total_frames = float('inf')
        
        # Setup tracking
        self.setup_trackers()
        
        # Setup models and components
        self.setup_models()
        
        # Print configuration
        self.config.print_config()
        print(f"🎯 Quality Focus: {self.expected_horses} horses, {self.expected_jockeys} jockeys (high-confidence tracking)")
        print(f"🐴🏇 Enhanced Horse Tracking System ready: {self.total_frames} frames @ {self.fps} FPS")
        print(f"📊 Debug logging enabled - logs will be saved at end of inference")
        print(f"🏆 Strategy: Perfect tracking of main contenders rather than fragmenting all horses")
    
    def parse_filename_counts(self):
        """Parse filename to extract expected horse/jockey counts - Quality-focused approach"""
        filename = self.video_path.stem
        
        # Look for pattern like horse_11, horse_22, etc.
        match = re.search(r'horse_(\d+)', filename, re.IGNORECASE)
        if match:
            count = int(match.group(1))
            # For quality-focused approach, limit expectations to main contenders
            quality_count = min(count, 6)  # Max 6 for quality tracking
            if quality_count < count:
                print(f"🎯 Quality-focused: Targeting {quality_count} main contenders from {count} total horses")
            return quality_count, quality_count
        
        # Default fallback - quality-focused
        print(f"⚠️ Could not parse count from filename '{filename}', using quality-focused defaults")
        return 4, 4  # Default to 4 horses, 4 jockeys for quality tracking
    
    def setup_trackers(self):
        """Initialize Deep OC-SORT tracker"""
        if not sv:
            raise RuntimeError("❌ Supervision not available - install with: pip install supervision")
        
        tracker_type = getattr(self.config, 'tracker_type', 'deepocsort')
        
        if tracker_type == 'deepocsort' and DEEPOCSORT_AVAILABLE:
            self.setup_deepocsort()
        elif tracker_type == 'boosttrack' and BOOSTTRACK_AVAILABLE:
            self.setup_boosttrack()
        else:
            raise RuntimeError(f"❌ {tracker_type} not available - install with: pip install boxmot")
    
    def setup_deepocsort(self):
        """Initialize Deep OC-SORT with all configurable parameters"""
        try:
            deepocsort_config = getattr(self.config, 'deepocsort_config', {})
            
            # Quality-focused defaults for Deep OC-SORT
            quality_defaults = {
                # Core tracking parameters
                'max_age': 200,              # Keep tracks alive much longer for racing
                'min_hits': 5,               # Require stable detections before tracking
                'det_thresh': 0.65,          # High confidence threshold
                'iou_threshold': 0.3,        # OC-SORT optimal IoU threshold
                'per_class': False,          # Single class tracking mode
                
                # Motion and Kalman filter parameters  
                'delta_t': 3,                # Velocity estimation window (frames)
                'inertia': 0.3,              # Higher inertia for smooth racing motion
                'Q_xy_scaling': 0.01,        # Position noise scaling
                'Q_s_scaling': 0.0001,       # Scale noise scaling
                
                # Association and ReID parameters
                'asso_func': 'iou',          # Association function (iou/giou)
                'w_association_emb': 0.6,    # Higher ReID weight for horse identification
                'alpha_fixed_emb': 0.9,      # Embedding update rate
                'aw_param': 0.5,             # Adaptive weighting parameter
                
                # Feature control flags
                'embedding_off': False,      # Keep ReID enabled
                'cmc_off': False,            # Keep camera motion compensation
                'aw_off': False,             # Keep adaptive weighting
            }
            
            # Merge config with defaults (config overrides)
            final_config = {**quality_defaults, **deepocsort_config}
            
            # Type validation and conversion
            typed_config = {}
            for key, value in final_config.items():
                try:
                    if key in ['max_age', 'min_hits', 'delta_t']:
                        typed_config[key] = int(value)
                    elif key in ['det_thresh', 'iou_threshold', 'inertia', 'w_association_emb', 
                               'alpha_fixed_emb', 'aw_param', 'Q_xy_scaling', 'Q_s_scaling']:
                        typed_config[key] = float(value)
                    elif key in ['per_class', 'embedding_off', 'cmc_off', 'aw_off']:
                        typed_config[key] = bool(value)
                    elif key in ['asso_func']:
                        typed_config[key] = str(value)
                    else:
                        typed_config[key] = value
                except (ValueError, TypeError) as e:
                    print(f"⚠️ Invalid config value for {key}={value}, using default")
                    typed_config[key] = quality_defaults.get(key, value)
            
            device_id = 0 if self.config.device == 'cuda' else 'cpu'
            reid_weights_path = Path('osnet_x0_25_msmt17.pt')
            
            print(f"🔧 Deep OC-SORT Quality Configuration:")
            key_params = ['max_age', 'min_hits', 'det_thresh', 'iou_threshold', 
                         'inertia', 'w_association_emb', 'alpha_fixed_emb']
            for param in key_params:
                if param in typed_config:
                    print(f"   {param}: {typed_config[param]}")
            
            # Show feature flags
            feature_flags = ['embedding_off', 'cmc_off', 'aw_off']
            enabled_features = [f for f in feature_flags if not typed_config.get(f, True)]
            disabled_features = [f for f in feature_flags if typed_config.get(f, False)]
            if enabled_features:
                print(f"   ✅ Features enabled: {', '.join(enabled_features)}")
            if disabled_features:
                print(f"   ❌ Features disabled: {', '.join(disabled_features)}")
            
            self.horse_tracker = DeepOcSort(
                reid_weights=reid_weights_path,
                device=device_id,
                half=True,
                **typed_config
            )
            
            self.human_tracker = DeepOcSort(
                reid_weights=reid_weights_path,
                device=device_id,
                half=True,
                **typed_config
            )
            
            self.tracker_type = 'deepocsort'
            print(f"✅ Deep OC-SORT + ReID initialized with {len(typed_config)} parameters")
            
        except Exception as e:
            print(f"❌ Deep OC-SORT initialization failed: {e}")
            print(f"📋 Config received: {deepocsort_config}")
            raise RuntimeError(f"Failed to initialize Deep OC-SORT: {e}")
    
    def setup_boosttrack(self):
        """Initialize BoostTrack with ReID"""
        try:
            boosttrack_config = getattr(self.config, 'boosttrack_config', {})
            
            # Force all config parameters to correct types
            typed_config = {}
            for key, value in boosttrack_config.items():
                if key in ['max_age', 'min_hits']:
                    typed_config[key] = int(value)
                elif key in ['det_thresh', 'iou_threshold', 'lambda_iou', 'lambda_mhd', 'lambda_shape', 'dlo_boost_coef', 'aspect_ratio_thresh']:
                    typed_config[key] = float(value)
                elif key in ['use_ecc', 'use_dlo_boost', 'use_duo_boost', 's_sim_corr', 'use_rich_s', 'use_sb', 'use_vt', 'with_reid']:
                    typed_config[key] = bool(value)
                elif key in ['min_box_area']:
                    typed_config[key] = int(value)
                else:
                    typed_config[key] = value
            
            # Enable ReID functionality
            typed_config['with_reid'] = True
            
            device_id = 0 if self.config.device == 'cuda' else 'cpu'
            
            # Fix: Convert reid_weights string to Path object
            reid_weights_path = Path('osnet_x0_25_msmt17.pt')
            
            self.horse_tracker = BoostTrack(
                reid_weights=reid_weights_path,
                device=device_id,
                half=True,
                with_reid=True,
                **typed_config
            )
            
            self.human_tracker = BoostTrack(
                reid_weights=reid_weights_path,
                device=device_id,
                half=True,
                with_reid=True,
                **typed_config
            )
            
            self.tracker_type = 'boosttrack'
            print(f"✅ BoostTrack + ReID initialized with {len(typed_config)} parameters")
            
        except Exception as e:
            print(f"❌ BoostTrack initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize BoostTrack: {e}")
    
    def setup_models(self):
        # Setup SuperAnimal model if needed
        self.superanimal = None
        if self.config.horse_detector in ['superanimal', 'both'] or self.config.horse_pose_estimator in ['superanimal', 'dual']:
            self.superanimal = SuperAnimalQuadruped(device=self.config.device, config=self.config)
        
        # Setup detection manager
        self.detection_manager = DetectionManager(self.config, self.superanimal)
        
        # Setup pose estimation manager
        self.pose_manager = PoseEstimationManager(self.config, self.superanimal)
        
        # Setup visualizer
        self.visualizer = Visualizer(self.config, self.superanimal)
        
        # Setup SigLIP classifier (REPLACES complex ReID)
        if getattr(self.config, 'enable_siglip_classification', False):
            self.siglip_classifier = SigLIPClassifier(self.config)
        else:
            self.siglip_classifier = None
            print("🔍 SigLIP Classification: DISABLED")
    
    def limit_detections(self, detections, max_count, detection_type="object"):
        """Limit detections to top-quality ones, keeping highest confidence detections"""
        if not sv or len(detections) == 0:
            return detections
        
        if len(detections) <= max_count:
            return detections
        
        # Sort by confidence (descending) and take top N highest quality detections
        sorted_indices = np.argsort(detections.confidence)[::-1]
        top_indices = sorted_indices[:max_count]
        
        # Log what we're filtering for quality analysis
        if len(detections) > max_count:
            kept_conf = detections.confidence[top_indices].min()
            dropped_conf = detections.confidence[sorted_indices[max_count:]].max()
            print(f"🎯 Quality filter: Kept {max_count} {detection_type} (conf≥{kept_conf:.3f}), dropped {len(detections)-max_count} (conf≤{dropped_conf:.3f})")
        
        limited_detections = sv.Detections(
            xyxy=detections.xyxy[top_indices],
            confidence=detections.confidence[top_indices],
            class_id=detections.class_id[top_indices] if hasattr(detections, 'class_id') and detections.class_id is not None else None
        )
        
        return limited_detections
    
    def update_tracker(self, tracker, detections, frame):
        """Update Deep OC-SORT with robust error handling"""
        if not sv or len(detections) == 0:
            return sv.Detections.empty()
        
        try:
            xyxy = np.array(detections.xyxy, dtype=np.float64)
            confidence = np.array([float(x) for x in detections.confidence], dtype=np.float64)
            
            if hasattr(detections, 'class_id') and detections.class_id is not None:
                class_ids = np.array([float(x) for x in detections.class_id], dtype=np.float64)
            else:
                class_ids = np.zeros(len(detections), dtype=np.float64)
            
            # Ensure minimum detection array size
            if len(xyxy) == 0:
                return sv.Detections.empty()
                
            dets_np = np.column_stack((xyxy, confidence, class_ids)).astype(np.float64)
            
            tracks = tracker.update(dets_np, frame)
            
            if tracks is None or len(tracks) == 0:
                return sv.Detections.empty()
            
            return sv.Detections(
                xyxy=tracks[:, :4],
                confidence=tracks[:, 5] if tracks.shape[1] > 5 else confidence[:len(tracks)],
                class_id=tracks[:, 6].astype(np.int32) if tracks.shape[1] > 6 else class_ids[:len(tracks)].astype(np.int32),
                tracker_id=tracks[:, 4].astype(np.int32)
            )
            
        except (IndexError, RuntimeError) as e:
            print(f"🔧 Deep OC-SORT error bypassed: {str(e)[:50]}...")
            # Continue processing without crashing
            return sv.Detections(
                xyxy=detections.xyxy,
                confidence=detections.confidence,
                class_id=detections.class_id if hasattr(detections, 'class_id') else None,
                tracker_id=np.arange(len(detections)) + 1000  # High IDs to avoid conflicts
            )
    
    def visualize_motion_predictions(self, frame, tracking_info):
        """Visualize motion predictions (placeholder for future enhancement)"""
        return frame
    
    def associate_poses_with_tracks(self, poses, tracked_detections):
        """Associate pose results with tracked detection IDs"""
        if not sv or not poses or len(tracked_detections) == 0:
            return poses
        
        # Add track IDs to pose results
        for i, pose in enumerate(poses):
            if hasattr(tracked_detections, 'tracker_id') and tracked_detections.tracker_id is not None and i < len(tracked_detections.tracker_id):
                pose['track_id'] = tracked_detections.tracker_id[i]
            else:
                pose['track_id'] = -1  # Untracked
        
        return poses
    
    def process_video(self):
        # Determine output path
        if self.config.output_path:
            output_path = self.config.output_path
        else:
            # Create safe output filename
            input_stem = self.video_path.stem if self.video_path.suffix else self.video_path.name
            output_path = str(self.video_path.parent / f"{input_stem}_boosttrack_output.mp4")
        
        print(f"🎬 Processing: {self.video_path}")
        print(f"📤 Output: {output_path}")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        frame_count = 0
        max_frames = self.config.max_frames or (self.total_frames if self.total_frames != float('inf') else 10000)
        paused = False
        
        stats = {
            'humans_detected': 0, 'horses_detected': 0,
            'human_poses': 0, 'horse_poses': 0,
            'superanimal_wins': 0, 'vitpose_wins': 0,
            'tracked_horses': 0, 'tracked_humans': 0,
            'active_horse_tracks': set(), 'active_human_tracks': set(),
            'siglip_classifications': 0, 'individual_identifications': 0
        }
        
        # Initialize progress bar (only if not displaying)
        if TQDM_AVAILABLE and not self.config.display and self.total_frames != float('inf'):
            pbar = tqdm(total=max_frames, desc="Processing with BoostTrack + SigLIP", 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
        else:
            pbar = None
        
        # Setup display window
        if self.config.display:
            window_name = "Enhanced Horse Racing System - BoostTrack + SigLIP"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            display_width = min(1200, self.width)
            display_height = int(self.height * (display_width / self.width))
            cv2.resizeWindow(window_name, display_width, display_height)

        while frame_count < max_frames:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # LOG: Frame start
            frame_start_time = time.time()
            self.debug_logger.log_frame_start(frame_count, frame.shape)
            
            # STEP 1: Detect objects
            human_detections = self.detection_manager.detect_humans(frame)
            horse_detections = self.detection_manager.detect_horses(frame)
            
            # LOG: Detections
            detection_method = f"H:{self.config.human_detector}/Ho:{self.config.horse_detector}"
            self.debug_logger.log_detections(human_detections, horse_detections, detection_method)
            
            # STEP 2: Limit detections to expected counts
            human_detections = self.limit_detections(human_detections, self.expected_jockeys, "jockeys")
            horse_detections = self.limit_detections(horse_detections, self.expected_horses, "horses")
            
            # STEP 3: Track detections with configured tracker
            tracked_horses = self.update_tracker(self.horse_tracker, horse_detections, frame)
            tracked_humans = self.update_tracker(self.human_tracker, human_detections, frame)
            
            # LOG: Tracking updates
            self.debug_logger.log_tracking_update(tracked_humans, tracked_horses, 'BoostTrack')
            
            # Update tracking stats - Handle None tracker_id
            if len(tracked_horses) > 0 and hasattr(tracked_horses, 'tracker_id') and tracked_horses.tracker_id is not None:
                stats['active_horse_tracks'].update(tracked_horses.tracker_id)
            if len(tracked_humans) > 0 and hasattr(tracked_humans, 'tracker_id') and tracked_humans.tracker_id is not None:
                stats['active_human_tracks'].update(tracked_humans.tracker_id)
            
            # Filter humans to jockeys only (using tracked detections)
            jockey_detections = self.detection_manager.filter_jockeys(tracked_humans, tracked_horses)
            
            # STEP 4: Apply SigLIP Classification (REPLACES complex ReID)
            if self.siglip_classifier:
                # Classify horses to specific individuals (1-9)
                horse_class_ids = self.siglip_classifier.classify_detections(
                    frame, tracked_horses, 'horse'
                )
                tracked_horses = self.siglip_classifier.update_tracker_ids(
                    tracked_horses, horse_class_ids, 'horse'
                )
                
                # Classify jockeys to specific individuals (1-9)
                jockey_class_ids = self.siglip_classifier.classify_detections(
                    frame, jockey_detections, 'jockey'
                )
                jockey_detections = self.siglip_classifier.update_tracker_ids(
                    jockey_detections, jockey_class_ids, 'jockey'
                )
                
                # Update stats
                stats['siglip_classifications'] = len(horse_class_ids) + len(jockey_class_ids)
                stats['individual_identifications'] = np.sum(horse_class_ids >= 0) + np.sum(jockey_class_ids >= 0)
            
            stats['humans_detected'] += len(jockey_detections) if sv else len(jockey_detections)
            stats['horses_detected'] += len(tracked_horses) if sv else len(tracked_horses)
            stats['tracked_horses'] = len(stats['active_horse_tracks'])
            stats['tracked_humans'] = len(stats['active_human_tracks'])
            
            # STEP 5: Estimate poses
            human_poses = self.pose_manager.estimate_human_poses(frame, jockey_detections)
            horse_poses = self.pose_manager.estimate_horse_poses(frame, tracked_horses)
            
            # LOG: Pose estimation
            self.debug_logger.log_pose_estimation(human_poses, horse_poses)
            
            # STEP 6: Associate poses with track IDs
            human_poses = self.associate_poses_with_tracks(human_poses, jockey_detections)
            horse_poses = self.associate_poses_with_tracks(horse_poses, tracked_horses)
            
            stats['human_poses'] += len(human_poses)
            stats['horse_poses'] += len(horse_poses)
            
            # Count method wins for dual mode
            if self.config.horse_pose_estimator == 'dual':
                superanimal_count = sum(1 for pose in horse_poses if pose.get('method') == 'SuperAnimal')
                vitpose_count = sum(1 for pose in horse_poses if pose.get('method') == 'ViTPose')
                stats['superanimal_wins'] += superanimal_count
                stats['vitpose_wins'] += vitpose_count
            
            # STEP 7: Visualize everything
            frame = self.visualizer.annotate_detections_with_tracking(frame, jockey_detections, tracked_horses)
            
            # Draw poses with track IDs
            for pose_result in human_poses:
                frame = self.visualizer.draw_human_pose_with_tracking(frame, pose_result)
            
            for pose_result in horse_poses:
                frame = self.visualizer.draw_horse_pose_with_tracking(frame, pose_result)
            
            # Add pose method labels
            frame = self.visualizer.draw_pose_labels(frame, horse_poses)
            
            # Add info overlay
            human_count = len(jockey_detections) if sv else len(jockey_detections)
            horse_count = len(tracked_horses) if sv else len(tracked_horses)
            
            frame = self.draw_enhanced_info_overlay(
                frame, frame_count, max_frames, human_count, horse_count,
                len(human_poses), len(horse_poses), stats,
                self.expected_horses, self.expected_jockeys
            )
            
            # Write frame to output video
            out.write(frame)
            
            # Display frame if requested
            if self.config.display:
                cv2.imshow(window_name, frame)
                
                # Handle key presses
                key = cv2.waitKey(1 if not paused else 0) & 0xFF
                
                if key == ord('q') or key == 27:  # Q or ESC
                    print("\n🛑 User requested quit")
                    break
                elif key == ord(' '):  # SPACE
                    paused = not paused
                    print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")
                elif paused and key != 255:  # Any other key when paused
                    pass  # Continue to next frame
            
            frame_count += 1
            
            # LOG: Frame end
            frame_end_time = time.time()
            self.debug_logger.log_frame_end(frame_end_time - frame_start_time)
            
            # Update progress bar
            if pbar:
                tracker_status = f"BoostTrack:{len(stats['active_horse_tracks'])}H/{len(stats['active_human_tracks'])}J"
                siglip_status = f"SigLIP:{stats['individual_identifications']}" if self.siglip_classifier else "SigLIP:OFF"
                pbar.set_postfix_str(f"H:{human_count}/{self.expected_jockeys} Ho:{horse_count}/{self.expected_horses} {tracker_status} {siglip_status}")
                pbar.update(1)
        
        self.cap.release()
        out.release()
        
        if self.config.display:
            cv2.destroyAllWindows()
        
        if pbar:
            pbar.close()
        
        # SAVE ALL DEBUG LOGS
        print(f"📊 Saving comprehensive debug logs...")
        log_files = self.debug_logger.save_logs(output_path)
        
        print(f"✅ Quality-focused processing complete!")
        print(f"📊 Final Stats:")
        print(f"   Target: {self.expected_horses} horses, {self.expected_jockeys} jockeys (main contenders)")
        print(f"   Humans detected: {stats['humans_detected']}")
        print(f"   Horses detected: {stats['horses_detected']}")
        print(f"   Human poses: {stats['human_poses']}")
        print(f"   Horse poses: {stats['horse_poses']}")
        print(f"   🔄 Unique tracks: {stats['tracked_humans']} humans, {stats['tracked_horses']} horses")
        
        # Quality assessment
        horse_quality = "EXCELLENT" if stats['tracked_horses'] <= self.expected_horses * 2 else "POOR"
        human_quality = "EXCELLENT" if stats['tracked_humans'] <= self.expected_jockeys * 2 else "POOR"
        print(f"   🏆 Tracking Quality: Horses {horse_quality}, Humans {human_quality}")
        
        # Tracking method info
        print(f"🎯 Tracking Method: BOOSTTRACK (Quality-Focused)")
        
        # SigLIP classification info
        if self.siglip_classifier:
            print(f"🔍 SigLIP Individual Identification:")
            print(f"   - Total classifications: {stats['siglip_classifications']}")
            print(f"   - Individual identifications: {stats['individual_identifications']}")
            siglip_stats = self.siglip_classifier.get_classification_stats()
            # print(f"   - Horse templates: {siglip_stats['horse_templates']}")
            print(f"   - siglip_stats: {siglip_stats}")
            print(f"   - Features: Reference image based identification")
        
        if self.config.horse_pose_estimator == 'dual':
            print(f"🥊 Competition Results:")
            print(f"   SuperAnimal wins: {stats['superanimal_wins']} (39 keypoints)")
            print(f"   ViTPose wins: {stats['vitpose_wins']} (17 keypoints)")
        
        print(f"🎯 Output: {output_path}")
        
        return output_path
    
    def draw_enhanced_info_overlay(self, frame: np.ndarray, frame_count: int, max_frames: int, 
                                 human_count: int, horse_count: int, human_poses: int, 
                                 horse_poses: int, stats: dict = None,
                                 expected_horses: int = 9, expected_jockeys: int = 9):
        """Draw enhanced info overlay with BoostTrack + SigLIP statistics"""
        total_display = str(max_frames) if max_frames != float('inf') else "∞"
        
        info_lines = [
            f"Frame: {frame_count+1}/{total_display}",
            f"Quality Focus: {expected_horses} horses, {expected_jockeys} jockeys (main contenders)",
            f"Config: H-Det:{self.config.human_detector} H-Pose:{self.config.human_pose_estimator}",
            f"        Horse-Det:{self.config.horse_detector} Horse-Pose:{self.config.horse_pose_estimator}",
            f"Tracked - Jockeys:{human_count}/{expected_jockeys} Horses:{horse_count}/{expected_horses}",
            f"Poses - Humans:{human_poses} Horses:{horse_poses}",
        ]
        
        # Add tracking statistics
        if stats:
            tracked_humans = stats.get('tracked_humans', 0)
            tracked_horses = stats.get('tracked_horses', 0)
            info_lines.append(f"🔄 Unique Tracks - Humans:{tracked_humans} Horses:{tracked_horses}")
        
        # Add tracking method info
        info_lines.append(f"🎯 Tracking Method: BOOSTTRACK")
        
        # Add SigLIP classification info
        if self.siglip_classifier and stats:
            individual_ids = stats.get('individual_identifications', 0)
            total_classifications = stats.get('siglip_classifications', 0)
            
            info_lines.append(f"🔍 SigLIP Individual ID: {individual_ids}/{total_classifications} identified")
            info_lines.append(f"   Features: Reference image based | Horse IDs: 100-108, Jockey IDs: 200-208")
        else:
            info_lines.append(f"🔍 SigLIP Individual ID: DISABLED")
        
        info_lines.append(f"📊 Debug logging: ENABLED (logs saved at end)")
        
        if self.config.horse_pose_estimator == 'dual' and stats:
            info_lines.append(f"Competition - SuperAnimal:{stats.get('superanimal_wins', 0)} ViTPose:{stats.get('vitpose_wins', 0)}")
        
        if self.config.display:
            info_lines.append(f"Controls: SPACE=Pause Q=Quit")
        
        # Semi-transparent background
        overlay = frame.copy()
        overlay_height = 25 + len(info_lines) * 18
        cv2.rectangle(overlay, (5, 5), (1000, overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        for i, line in enumerate(info_lines):
            y_pos = 25 + i * 18
            cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame


def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python main.py config.yaml")
        sys.exit(1)
    
    # Load config from file
    config_file = sys.argv[1]
    config = Config(config_file)
    
    # Get video path from config
    video_path = getattr(config, 'video_path', None)
    if not video_path:
        print("❌ Error: Config file must specify 'video_path'")
        print("Available config attributes:", [attr for attr in dir(config) if not attr.startswith('_')])
        sys.exit(1)
    
    # Auto-detect device if not specified
    if config.device == "cpu" and VITPOSE_AVAILABLE and torch.cuda.is_available():
        config.device = "cuda"
    
    # Check if video file exists
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"❌ Error: Video file '{video_path}' does not exist")
        sys.exit(1)
    
    try:
        system = HybridPoseSystem(video_path, config)
        system.process_video()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()