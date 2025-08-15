#!/usr/bin/env python3
"""
Simplified Racer Tracking System
Single Roboflow model detects horse+jockey compound entities
Dual pose estimation (SuperAnimal + ViTPose) on same bounding box
"""

import cv2
import numpy as np
import torch
import warnings
import time
from pathlib import Path

# Local imports
from config import Config
from models import SuperAnimalQuadruped
from detectors import RacerDetectionManager
from pose_estimators import SimplifiedPoseEstimationManager
from visualizers import SimplifiedVisualizer
from reid_pipeline import SimplifiedReIDPipeline
from debug_logger import TrackingDebugLogger

# Suppress warnings
warnings.filterwarnings("ignore")

# Check dependencies
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Install tqdm for progress bars: pip install tqdm")

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


class SimplifiedRacerTrackingSystem:
    def __init__(self, video_path: str, config: Config):
        self.video_path = Path(video_path)
        self.config = config
        
        # Initialize debug logger
        self.debug_logger = TrackingDebugLogger(self.config)
        self.debug_logger.set_video_name(str(self.video_path))
        
        # Setup video
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        
        if self.total_frames <= 0:
            print("⚠️ Warning: Frame count not available")
            self.total_frames = float('inf')
        
        # Setup tracking
        self.setup_tracker()
        
        # Setup components
        self.setup_models()
        
        # Print configuration
        self.config.print_config()
        print(f"🏇 Simplified Racer Tracking System ready")
        print(f"📊 {self.total_frames} frames @ {self.fps} FPS")
        print(f"🎯 Strategy: Roboflow compound detection + dual pose overlay")
    
    def setup_tracker(self):
        """Initialize Deep OC-SORT tracker for compound racers"""
        if not sv:
            raise RuntimeError("❌ Supervision not available")
        
        if not DEEPOCSORT_AVAILABLE:
            raise RuntimeError("❌ DeepOCSORT not available")
        
        try:
            deepocsort_config = getattr(self.config, 'deepocsort_config', {})
            
            # Quality defaults for compound entities
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
                except (ValueError, TypeError):
                    typed_config[key] = quality_defaults.get(key, value)
            
            device_id = 0 if self.config.device == 'cuda' else 'cpu'
            reid_weights_path = Path('osnet_x0_25_msmt17.pt')
            
            self.racer_tracker = DeepOcSort(
                reid_weights=reid_weights_path,
                device=device_id,
                half=True,
                **typed_config
            )
            
            print(f"✅ Deep OC-SORT initialized for compound racers")
            
        except Exception as e:
            print(f"❌ Deep OC-SORT initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize tracker: {e}")
    
    def setup_models(self):
        """Setup all system components"""
        # SuperAnimal model for pose estimation
        self.superanimal = None
        if self.config.horse_pose_estimator in ['superanimal', 'both']:
            self.superanimal = SuperAnimalQuadruped(device=self.config.device, config=self.config)
        
        # Roboflow detection manager
        self.detection_manager = RacerDetectionManager(self.config)
        
        # Pose estimation manager
        self.pose_manager = SimplifiedPoseEstimationManager(self.config, self.superanimal)
        
        # Visualizer
        self.visualizer = SimplifiedVisualizer(self.config, self.superanimal)
        
        # ReID pipeline (optional)
        self.reid_pipeline = None
        if getattr(self.config, 'enable_reid', False):
            self.reid_pipeline = SimplifiedReIDPipeline(self.config)
        else:
            print("🔄 ReID: DISABLED")
    
    def limit_detections(self, detections, max_count):
        """Limit detections to top-quality ones"""
        if not sv or len(detections) == 0 or len(detections) <= max_count:
            return detections
        
        # Sort by confidence and take top N
        sorted_indices = np.argsort(detections.confidence)[::-1]
        top_indices = sorted_indices[:max_count]
        
        limited_detections = sv.Detections(
            xyxy=detections.xyxy[top_indices],
            confidence=detections.confidence[top_indices],
            class_id=detections.class_id[top_indices] if hasattr(detections, 'class_id') and detections.class_id is not None else None
        )
        
        # Copy mask data if available
        if hasattr(detections, 'mask') and detections.mask is not None:
            limited_detections.mask = detections.mask[top_indices]
        
        print(f"🎯 Limited to {max_count} best racers from {len(detections)} detections")
        return limited_detections
    
    def update_tracker(self, tracker, detections, frame):
        """Update Deep OC-SORT with error handling"""
        if not sv or len(detections) == 0:
            return sv.Detections.empty()
        
        try:
            xyxy = np.array(detections.xyxy, dtype=np.float64)
            confidence = np.array([float(x) for x in detections.confidence], dtype=np.float64)
            
            if hasattr(detections, 'class_id') and detections.class_id is not None:
                class_ids = np.array([float(x) for x in detections.class_id], dtype=np.float64)
            else:
                class_ids = np.zeros(len(detections), dtype=np.float64)
            
            dets_np = np.column_stack((xyxy, confidence, class_ids)).astype(np.float64)
            
            tracks = tracker.update(dets_np, frame)
            
            if tracks is None or len(tracks) == 0:
                return sv.Detections.empty()
            
            tracked_detections = sv.Detections(
                xyxy=tracks[:, :4],
                confidence=tracks[:, 5] if tracks.shape[1] > 5 else confidence[:len(tracks)],
                class_id=tracks[:, 6].astype(np.int32) if tracks.shape[1] > 6 else class_ids[:len(tracks)].astype(np.int32),
                tracker_id=tracks[:, 4].astype(np.int32)
            )
            
            # Copy mask data if available
            if hasattr(detections, 'mask') and detections.mask is not None:
                tracked_detections.mask = detections.mask[:len(tracked_detections)]
            
            return tracked_detections
            
        except Exception as e:
            print(f"🔧 Tracker error: {str(e)[:50]}...")
            # Fallback with dummy track IDs
            return sv.Detections(
                xyxy=detections.xyxy,
                confidence=detections.confidence,
                class_id=detections.class_id if hasattr(detections, 'class_id') else None,
                tracker_id=np.arange(len(detections)) + 1000
            )
    
    def process_video(self):
        """Process video with simplified racer tracking pipeline"""
        # Determine output path
        if self.config.output_path:
            output_path = self.config.output_path
        else:
            input_stem = self.video_path.stem
            output_path = str(self.video_path.parent / f"{input_stem}_racer_output.mp4")
        
        print(f"🎬 Processing: {self.video_path}")
        print(f"📤 Output: {output_path}")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        frame_count = 0
        max_frames = self.config.max_frames or (self.total_frames if self.total_frames != float('inf') else 10000)
        paused = False
        
        stats = {
            'racers_detected': 0,
            'total_poses': 0,
            'superanimal_poses': 0,
            'vitpose_poses': 0,
            'tracked_racers': 0,
            'active_tracks': set()
        }
        
        # Initialize progress bar
        if TQDM_AVAILABLE and not self.config.display and self.total_frames != float('inf'):
            pbar = tqdm(total=max_frames, desc="Processing Simplified Racer Tracking", 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
        else:
            pbar = None
        
        # Setup display window
        if self.config.display:
            window_name = "Simplified Racer Tracking System"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            display_width = min(1200, self.width)
            display_height = int(self.height * (display_width / self.width))
            cv2.resizeWindow(window_name, display_width, display_height)

        while frame_count < max_frames:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_start_time = time.time()
            self.debug_logger.log_frame_start(frame_count, frame.shape)
            
            # STEP 1: Detect compound racers using Roboflow
            racer_detections = self.detection_manager.detect_racers(frame)
            
            # Debug: Print detection info
            if frame_count < 5:  # Only for first few frames
                print(f"Frame {frame_count}: Got {len(racer_detections)} detections")
                if len(racer_detections) > 0:
                    print(f"  Confidences: {racer_detections.confidence}")
                    print(f"  Class IDs: {racer_detections.class_id if hasattr(racer_detections, 'class_id') else 'None'}")
            
            # Log detections
            self.debug_logger.log_detections(sv.Detections.empty(), racer_detections, "Roboflow")
            
            # STEP 2: Limit to maximum expected racers
            max_racers = getattr(self.config, 'max_racers', 10)
            racer_detections = self.limit_detections(racer_detections, max_racers)
            
            # STEP 3: Track compound racers
            tracked_racers = self.update_tracker(self.racer_tracker, racer_detections, frame)
            
            # Log tracking
            self.debug_logger.log_tracking_update(sv.Detections.empty(), tracked_racers, "DeepOCSORT")
            
            # Update stats - Fix the counting issue
            current_racers = len(tracked_racers) if sv and hasattr(tracked_racers, '__len__') else 0
            if hasattr(tracked_racers, 'tracker_id') and tracked_racers.tracker_id is not None:
                stats['active_tracks'].update(tracked_racers.tracker_id[tracked_racers.tracker_id >= 0])
            
            stats['racers_detected'] += current_racers
            stats['tracked_racers'] = len(stats['active_tracks'])
            
            # STEP 4: Apply ReID enhancement (optional)
            reid_info = {}
            if self.reid_pipeline:
                # Get masks for ReID
                masks = self.detection_manager.get_masks(tracked_racers)
                tracked_racers = self.reid_pipeline.enhance_tracking(tracked_racers, masks, frame)
                reid_info = self.reid_pipeline.get_tracking_info()
            
            # STEP 5: Estimate poses on compound entities
            all_poses = self.pose_manager.estimate_poses_on_racers(frame, tracked_racers)
            
            # Filter poses by confidence
            filtered_poses = self.pose_manager.filter_poses_by_confidence(all_poses)
            
            # Get pose statistics
            pose_stats = self.pose_manager.get_pose_statistics(filtered_poses)
            stats['total_poses'] += pose_stats['total_poses']
            stats['superanimal_poses'] += pose_stats['superanimal_count']
            stats['vitpose_poses'] += pose_stats['vitpose_count']
            
            # Log poses
            self.debug_logger.log_pose_estimation([], filtered_poses)
            
            # STEP 6: Visualize everything
            # Annotate racer detections
            frame = self.visualizer.annotate_racer_detections(frame, tracked_racers)
            
            # Draw compound poses (both SuperAnimal and ViTPose)
            frame = self.visualizer.draw_compound_poses(frame, filtered_poses)
            
            # Draw pose information labels
            frame = self.visualizer.draw_pose_info_labels(frame, filtered_poses)
            
            # Draw info overlay
            frame = self.visualizer.draw_info_overlay(
                frame, frame_count, max_frames, len(tracked_racers), 
                pose_stats, reid_info
            )
            
            # Write frame to output
            out.write(frame)
            
            # Display if requested
            if self.config.display:
                cv2.imshow(window_name, frame)
                
                key = cv2.waitKey(1 if not paused else 0) & 0xFF
                
                if key == ord('q') or key == 27:  # Q or ESC
                    print("\n🛑 User quit")
                    break
                elif key == ord(' '):  # SPACE
                    paused = not paused
                    print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")
            
            frame_count += 1
            
            # Log frame end
            frame_end_time = time.time()
            self.debug_logger.log_frame_end(frame_end_time - frame_start_time)
            
            # Update progress bar
            if pbar:
                racer_count = len(tracked_racers) if hasattr(tracked_racers, '__len__') else 0
                pose_count = pose_stats.get('total_poses', 0)
                reid_status = f"ReID:{reid_info.get('total_reassignments', 0)}" if self.reid_pipeline else "ReID:OFF"
                pbar.set_postfix_str(f"Racers:{racer_count}/{max_racers} Poses:{pose_count} {reid_status}")
                pbar.update(1)
        
        # Cleanup
        self.cap.release()
        out.release()
        if self.config.display:
            cv2.destroyAllWindows()
        if pbar:
            pbar.close()
        
        # Save debug logs
        print(f"📊 Saving debug logs...")
        self.debug_logger.save_logs(output_path)
        
        # Print final statistics
        print(f"✅ Simplified racer tracking complete!")
        print(f"📊 Final Stats:")
        print(f"   Racers detected: {stats['racers_detected']}")
        print(f"   Unique tracks: {stats['tracked_racers']}")
        print(f"   Total poses: {stats['total_poses']}")
        print(f"   - SuperAnimal: {stats['superanimal_poses']} (39kp)")
        print(f"   - ViTPose: {stats['vitpose_poses']} (17kp)")
        
        if self.reid_pipeline:
            reid_info = self.reid_pipeline.get_tracking_info()
            print(f"🔄 ReID Performance:")
            print(f"   Total reassignments: {reid_info['total_reassignments']}")
            print(f"   Memory tracks: {len(reid_info['memory_tracks'])}")
        
        print(f"🎯 Output: {output_path}")
        
        return output_path


def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python main.py config.yaml")
        print("\n🏇 Simplified Racer Tracking System")
        print("Features:")
        print("- Single Roboflow API for compound racer detection")
        print("- Built-in segmentation masks")
        print("- Dual pose estimation (SuperAnimal + ViTPose)")
        print("- Deep OC-SORT tracking for compound entities")
        print("- Optional MegaDescriptor ReID")
        print("\nRequirements:")
        print("- Roboflow API key and model ID")
        print("- pip install inference supervision transformers torch boxmot")
        sys.exit(1)
    
    # Load configuration
    config_file = sys.argv[1]
    config = Config(config_file)
    
    # Validate required settings
    video_path = getattr(config, 'video_path', None)
    if not video_path:
        print("❌ Error: video_path must be specified in config")
        sys.exit(1)
    
    if not hasattr(config, 'roboflow_api_key') or not config.roboflow_api_key:
        print("❌ Error: roboflow_api_key must be specified in config")
        sys.exit(1)
    
    if not hasattr(config, 'roboflow_model_id') or not config.roboflow_model_id:
        print("❌ Error: roboflow_model_id must be specified in config")
        sys.exit(1)
    
    # Auto-detect device
    if config.device == "cpu" and torch.cuda.is_available():
        config.device = "cuda"
        print("🔧 Auto-detected CUDA device")
    
    # Check video file
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"❌ Error: Video file '{video_path}' does not exist")
        sys.exit(1)
    
    try:
        system = SimplifiedRacerTrackingSystem(video_path, config)
        system.process_video()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()