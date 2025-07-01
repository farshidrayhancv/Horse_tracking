#!/usr/bin/env python3
"""
Enhanced Horse Tracking System with DeepOCSORT + SigLIP OCR Classification
OCR-based horse number detection (0-9) for tracking
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
    print("✓ Supervision available")
except ImportError:
    print("❌ Supervision not available - install with: pip install supervision")
    sv = None

try:
    from boxmot import DeepOcSort
    DEEPOCSORT_AVAILABLE = True
    print("✓ DeepOcSort available")
except ImportError:
    DEEPOCSORT_AVAILABLE = False
    print("⚠️ BoxMOT trackers not available - install with: pip install boxmot")


class HorseTrackingSystem:
    def __init__(self, video_path: str, config: Config):
        self.video_path = Path(video_path)
        self.config = config
        
        # Initialize debug logger
        self.debug_logger = TrackingDebugLogger(self.config)
        self.debug_logger.set_video_name(str(self.video_path))
        
        # Parse expected counts from filename
        self.expected_horses = self.parse_filename_counts()
        
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
        self.setup_tracker()
        
        # Setup models and components
        self.setup_models()
        
        # Print configuration
        self.config.print_config()
        print(f"🎯 Horse-Only Focus: {self.expected_horses} horses with OCR number detection")
        print(f"🐴 Enhanced Horse Tracking System ready: {self.total_frames} frames @ {self.fps} FPS")
        print(f"📊 Debug logging enabled - logs will be saved at end of inference")
        print(f"🔢 Strategy: OCR-based horse number detection (0-9) for consistent tracking")
    
    def parse_filename_counts(self):
        """Parse filename to extract expected horse count"""
        filename = self.video_path.stem
        
        # Look for pattern like horse_11, horse_22, etc.
        match = re.search(r'horse_(\d+)', filename, re.IGNORECASE)
        if match:
            count = int(match.group(1))
            # For quality-focused approach, limit expectations to main contenders
            quality_count = min(count, 10)  # Max 10 horses for quality tracking
            if quality_count < count:
                print(f"🎯 Quality-focused: Targeting {quality_count} main horses from {count} total")
            return quality_count
        
        # Default fallback - quality-focused
        print(f"⚠️ Could not parse count from filename '{filename}', using quality-focused defaults")
        return 9  # Default to 9 horses for quality tracking
    
    def setup_tracker(self):
        """Initialize Deep OC-SORT tracker for horses only"""
        if not sv:
            raise RuntimeError("❌ Supervision not available - install with: pip install supervision")
        
        if not DEEPOCSORT_AVAILABLE:
            raise RuntimeError("❌ DeepOCSORT not available - install with: pip install boxmot")
        
        try:
            deepocsort_config = getattr(self.config, 'deepocsort_config', {})
            
            # Quality-focused defaults for Deep OC-SORT
            quality_defaults = {
                'max_age': 100,
                'min_hits': 5,
                'det_thresh': 0.65,
                'iou_threshold': 0.3,
                'per_class': False,
                'delta_t': 3,
                'inertia': 0.3,
                'Q_xy_scaling': 0.01,
                'Q_s_scaling': 0.0001,
                'asso_func': 'iou',
                'w_association_emb': 0.6,
                'alpha_fixed_emb': 0.9,
                'aw_param': 0.5,
                'embedding_off': True,
                'cmc_off': False,
                'aw_off': False,
            }
            
            # Merge config with defaults
            final_config = {**quality_defaults, **deepocsort_config}
            
            # Type validation
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
            
            print(f"🔧 Deep OC-SORT Horse Configuration:")
            key_params = ['max_age', 'min_hits', 'det_thresh', 'iou_threshold', 
                         'inertia', 'w_association_emb', 'alpha_fixed_emb']
            for param in key_params:
                if param in typed_config:
                    print(f"   {param}: {typed_config[param]}")
            
            self.horse_tracker = DeepOcSort(
                reid_weights=reid_weights_path,
                device=device_id,
                half=True,
                **typed_config
            )
            
            self.tracker_type = 'deepocsort'
            print(f"✅ Deep OC-SORT + ReID initialized for horses")
            
        except Exception as e:
            print(f"❌ Deep OC-SORT initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize Deep OC-SORT: {e}")
    
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
        
        # Setup SigLIP classifier with OCR
        if getattr(self.config, 'enable_siglip_classification', False):
            self.siglip_classifier = SigLIPClassifier(self.config)
        else:
            self.siglip_classifier = None
            print("🔍 SigLIP OCR Classification: DISABLED")
    
    def limit_detections(self, detections, max_count, detection_type="object"):
        """Limit detections to top-quality ones"""
        if not sv or len(detections) == 0:
            return detections
        
        if len(detections) <= max_count:
            return detections
        
        # Sort by confidence and take top N
        sorted_indices = np.argsort(detections.confidence)[::-1]
        top_indices = sorted_indices[:max_count]
        
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
            return sv.Detections(
                xyxy=detections.xyxy,
                confidence=detections.confidence,
                class_id=detections.class_id if hasattr(detections, 'class_id') else None,
                tracker_id=np.arange(len(detections)) + 1000
            )
    
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
            input_stem = self.video_path.stem if self.video_path.suffix else self.video_path.name
            output_path = str(self.video_path.parent / f"{input_stem}_horse_ocr_output.mp4")
        
        print(f"🎬 Processing: {self.video_path}")
        print(f"📤 Output: {output_path}")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        frame_count = 0
        max_frames = self.config.max_frames or (self.total_frames if self.total_frames != float('inf') else 10000)
        paused = False
        
        stats = {
            'horses_detected': 0,
            'horse_poses': 0,
            'superanimal_wins': 0, 
            'vitpose_wins': 0,
            'tracked_horses': 0,
            'active_horse_tracks': set(),
            'siglip_classifications': 0, 
            'horse_identifications': 0
        }
        
        # Initialize progress bar
        if TQDM_AVAILABLE and not self.config.display and self.total_frames != float('inf'):
            pbar = tqdm(total=max_frames, desc="Processing with OCR Horse Tracking", 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
        else:
            pbar = None
        
        # Setup display window
        if self.config.display:
            window_name = "Enhanced Horse Racing System - OCR Number Detection"
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
            
            # STEP 1: Detect horses only
            horse_detections = self.detection_manager.detect_horses(frame)
            
            # LOG: Detections
            detection_method = f"Horse:{self.config.horse_detector}"
            self.debug_logger.log_detections(sv.Detections.empty(), horse_detections, detection_method)
            
            # STEP 2: Limit detections to expected count
            horse_detections = self.limit_detections(horse_detections, self.expected_horses, "horses")
            
            # STEP 3: Track detections
            tracked_horses = self.update_tracker(self.horse_tracker, horse_detections, frame)
            
            # LOG: Tracking updates
            self.debug_logger.log_tracking_update(sv.Detections.empty(), tracked_horses, self.tracker_type)
            
            # Update tracking stats
            if len(tracked_horses) > 0 and hasattr(tracked_horses, 'tracker_id') and tracked_horses.tracker_id is not None:
                stats['active_horse_tracks'].update(tracked_horses.tracker_id)
            
            # STEP 4: Apply SigLIP OCR Classification
            if self.siglip_classifier:
                # Classify horses to specific numbers (0-9)
                horse_class_ids = self.siglip_classifier.classify_detections(
                    frame, tracked_horses, 'horse'
                )
                tracked_horses = self.siglip_classifier.update_tracker_ids(
                    tracked_horses, horse_class_ids, 'horse'
                )
                
                # Update stats
                stats['siglip_classifications'] = len(horse_class_ids)
                stats['horse_identifications'] = np.sum(horse_class_ids >= 0)
            
            stats['horses_detected'] += len(tracked_horses) if sv else len(tracked_horses)
            stats['tracked_horses'] = len(stats['active_horse_tracks'])
            
            # STEP 5: Estimate horse poses
            horse_poses = self.pose_manager.estimate_horse_poses(frame, tracked_horses)
            
            # LOG: Pose estimation
            self.debug_logger.log_pose_estimation([], horse_poses)
            
            # STEP 6: Associate poses with track IDs
            horse_poses = self.associate_poses_with_tracks(horse_poses, tracked_horses)
            
            stats['horse_poses'] += len(horse_poses)
            
            # Count method wins for dual mode
            if self.config.horse_pose_estimator == 'dual':
                superanimal_count = sum(1 for pose in horse_poses if pose.get('method') == 'SuperAnimal')
                vitpose_count = sum(1 for pose in horse_poses if pose.get('method') == 'ViTPose')
                stats['superanimal_wins'] += superanimal_count
                stats['vitpose_wins'] += vitpose_count
            
            # STEP 7: Visualize horses only
            frame = self.visualizer.annotate_detections_with_tracking(frame, sv.Detections.empty(), tracked_horses)
            
            # Draw poses with track IDs
            for pose_result in horse_poses:
                frame = self.visualizer.draw_horse_pose_with_tracking(frame, pose_result)
            
            # Add pose method labels
            frame = self.visualizer.draw_pose_labels(frame, horse_poses)
            
            # Add info overlay
            horse_count = len(tracked_horses) if sv else len(tracked_horses)
            
            frame = self.draw_horse_info_overlay(
                frame, frame_count, max_frames, horse_count,
                len(horse_poses), stats, self.expected_horses
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
                tracker_status = f"{self.tracker_type}:{len(stats['active_horse_tracks'])}H"
                ocr_status = f"OCR:{stats['horse_identifications']}" if self.siglip_classifier else "OCR:OFF"
                pbar.set_postfix_str(f"Horses:{horse_count}/{self.expected_horses} {tracker_status} {ocr_status}")
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
        
        print(f"✅ Horse-only OCR processing complete!")
        print(f"📊 Final Stats:")
        print(f"   Target: {self.expected_horses} horses")
        print(f"   Horses detected: {stats['horses_detected']}")
        print(f"   Horse poses: {stats['horse_poses']}")
        print(f"   🔄 Unique tracks: {stats['tracked_horses']} horses")
        
        # Quality assessment
        horse_quality = "EXCELLENT" if stats['tracked_horses'] <= self.expected_horses * 2 else "POOR"
        print(f"   🏆 Tracking Quality: Horses {horse_quality}")
        
        # Tracking method info
        print(f"🎯 Tracking Method: {self.tracker_type.upper()}")
        
        # SigLIP OCR info
        if self.siglip_classifier:
            print(f"🔢 SigLIP OCR Number Detection:")
            print(f"   - Total classifications: {stats['siglip_classifications']}")
            print(f"   - Horse identifications: {stats['horse_identifications']}")
            ocr_stats = self.siglip_classifier.get_classification_stats()
            print(f"   - Trackable horses: {ocr_stats['trackable_horses']}")
            print(f"   - Classification accuracy: {ocr_stats['accuracy']:.3f}" if ocr_stats['accuracy'] else "   - Classification accuracy: Not available")
        
        if self.config.horse_pose_estimator == 'dual':
            print(f"🥊 Competition Results:")
            print(f"   SuperAnimal wins: {stats['superanimal_wins']} (39 keypoints)")
            print(f"   ViTPose wins: {stats['vitpose_wins']} (17 keypoints)")
        
        print(f"🎯 Output: {output_path}")
        
        return output_path
    
    def draw_horse_info_overlay(self, frame: np.ndarray, frame_count: int, max_frames: int, 
                               horse_count: int, horse_poses: int, stats: dict = None,
                               expected_horses: int = 9):
        """Draw horse-only info overlay with dynamic OCR statistics"""
        total_display = str(max_frames) if max_frames != float('inf') else "∞"
        
        # Get dynamic trackable horses
        trackable_horses = []
        if self.siglip_classifier and hasattr(self.siglip_classifier, 'valid_classes'):
            trackable_horses = sorted(list(self.siglip_classifier.valid_classes))
        
        info_lines = [
            f"Frame: {frame_count+1}/{total_display}",
            f"Horse Focus: {len(trackable_horses)} trackable horses {trackable_horses}" if trackable_horses else f"Horse Focus: {expected_horses} horses with OCR detection",
            f"Config: Horse-Det:{self.config.horse_detector} Horse-Pose:{self.config.horse_pose_estimator}",
            f"Tracked Horses: {horse_count}",
            f"Horse Poses: {horse_poses}",
        ]
        
        # Add tracking statistics
        if stats:
            tracked_horses = stats.get('tracked_horses', 0)
            info_lines.append(f"🔄 Unique Horse Tracks: {tracked_horses}")
        
        # Add tracking method info
        info_lines.append(f"🎯 Tracking Method: {self.tracker_type.upper()}")
        
        # Add SigLIP OCR info
        if self.siglip_classifier and stats:
            horse_ids = stats.get('horse_identifications', 0)
            total_classifications = stats.get('siglip_classifications', 0)
            
            if trackable_horses:
                info_lines.append(f"🔢 SigLIP OCR: {horse_ids}/{total_classifications} identified | Horses: {trackable_horses}")
            else:
                info_lines.append(f"🔢 SigLIP OCR: {horse_ids}/{total_classifications} identified")
            info_lines.append(f"   Features: OCR training on detected numbers only")
        else:
            info_lines.append(f"🔢 SigLIP OCR Detection: DISABLED")
        
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
        system = HorseTrackingSystem(video_path, config)
        system.process_video()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()