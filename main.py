import cv2
import numpy as np
import torch
import warnings
import re
from pathlib import Path

# Local imports
from config import Config
from models import SuperAnimalQuadruped
from detectors import DetectionManager
from pose_estimators import PoseEstimationManager
from visualizers import Visualizer
from reid_pipeline import ReIDPipeline

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
    print("✓ Official ViTPose (HuggingFace Transformers) available")
except ImportError:
    VITPOSE_AVAILABLE = False
    print("⚠️ ViTPose not available - install with: pip install transformers torch")

try:
    from dlclibrary import download_huggingface_model
    SUPERANIMAL_AVAILABLE = True
    print("✓ Official DLClibrary + SuperAnimal available")
except ImportError:
    SUPERANIMAL_AVAILABLE = False
    print("⚠️ DLClibrary not available - install with: pip install dlclibrary")

try:
    import supervision as sv
    from supervision import ByteTrack
    print("✓ Supervision 0.25.1 with ByteTrack for professional tracking")
except ImportError:
    print("❌ Supervision not available - install with: pip install supervision")
    sv = None

class HybridPoseSystem:
    def __init__(self, video_path: str, config: Config):
        self.video_path = Path(video_path)
        self.config = config
        
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
        
        # Setup re-identification pipeline
        self.reid_pipeline = None
        if self.config.enable_reid_pipeline:
            try:
                self.reid_pipeline = ReIDPipeline(self.config)
                print("✅ Re-identification pipeline initialized")
            except Exception as e:
                print(f"❌ Re-identification pipeline failed to initialize: {e}")
                print("🔄 Continuing without re-identification")
                self.config.enable_reid_pipeline = False
        
        # Print configuration
        self.config.print_config()
        print(f"🎯 Expected: {self.expected_horses} horses, {self.expected_jockeys} jockeys")
        print(f"🐴🏇 Configurable Pose System with Re-ID ready: {self.total_frames} frames @ {self.fps} FPS")
    
    def parse_filename_counts(self):
        """Parse filename to extract expected horse/jockey counts"""
        filename = self.video_path.stem
        
        # Look for pattern like horse_11, horse_22, etc.
        match = re.search(r'horse_(\d+)', filename, re.IGNORECASE)
        if match:
            count = int(match.group(1))
            return count, count  # Same number of horses and jockeys
        
        # Default fallback
        print(f"⚠️ Could not parse count from filename '{filename}', using defaults")
        return 10, 10  # Default to 10 horses, 10 jockeys
    
    def setup_trackers(self):
        """Initialize ByteTracks for horses and humans"""
        if not sv:
            print("❌ Tracking disabled - supervision not available")
            self.horse_tracker = None
            self.human_tracker = None
            return
        
        # Initialize trackers with video FPS
        self.horse_tracker = ByteTrack(frame_rate=int(self.fps))
        self.human_tracker = ByteTrack(frame_rate=int(self.fps))
        
        print(f"✅ ByteTracks initialized @ {self.fps} FPS")
    
    def setup_models(self):
        # Setup SuperAnimal model if needed - 🔥 PASS CONFIG
        self.superanimal = None
        if self.config.horse_detector in ['superanimal', 'both'] or self.config.horse_pose_estimator in ['superanimal', 'dual']:
            self.superanimal = SuperAnimalQuadruped(device=self.config.device, config=self.config)  # 🔥 Pass config
        
        # Setup detection manager
        self.detection_manager = DetectionManager(self.config, self.superanimal)
        
        # Setup pose estimation manager
        self.pose_manager = PoseEstimationManager(self.config, self.superanimal)
        
        # Setup visualizer
        self.visualizer = Visualizer(self.config, self.superanimal)
    
    def limit_detections(self, detections, max_count, detection_type="object"):
        """Limit detections to expected count, keeping highest confidence ones"""
        if not sv or len(detections) == 0:
            return detections
        
        if len(detections) <= max_count:
            return detections
        
        # Sort by confidence (descending) and take top N
        sorted_indices = np.argsort(detections.confidence)[::-1]
        top_indices = sorted_indices[:max_count]
        
        limited_detections = sv.Detections(
            xyxy=detections.xyxy[top_indices],
            confidence=detections.confidence[top_indices],
            class_id=detections.class_id[top_indices]
        )
        
        return limited_detections
    
    def associate_poses_with_tracks(self, poses, tracked_detections):
        """Associate pose results with tracked detection IDs"""
        if not sv or not poses or len(tracked_detections) == 0:
            return poses
        
        # Add track IDs to pose results
        for i, pose in enumerate(poses):
            if i < len(tracked_detections.tracker_id):
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
            output_path = str(self.video_path.parent / f"{input_stem}_reid_tracked_output.mp4")
        
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
            'reid_reassignments': 0
        }
        
        # Initialize progress bar (only if not displaying)
        if TQDM_AVAILABLE and not self.config.display and self.total_frames != float('inf'):
            pbar = tqdm(total=max_frames, desc="Processing with Re-ID pipeline", 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
        else:
            pbar = None
        
        # Setup display window
        if self.config.display:
            window_name = "Horse Racing System with Re-Identification"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            display_width = min(1200, self.width)
            display_height = int(self.height * (display_width / self.width))
            cv2.resizeWindow(window_name, display_width, display_height)
        
        while frame_count < max_frames:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 🔥 STEP 1: Detect objects based on config
            human_detections = self.detection_manager.detect_humans(frame)
            horse_detections = self.detection_manager.detect_horses(frame)
            
            # 🔥 STEP 2: Limit detections to expected counts
            human_detections = self.limit_detections(human_detections, self.expected_jockeys, "jockeys")
            horse_detections = self.limit_detections(horse_detections, self.expected_horses, "horses")
            
            # 🔥 STEP 3: Track detections
            if sv and self.horse_tracker and self.human_tracker:
                tracked_horses = self.horse_tracker.update_with_detections(horse_detections)
                tracked_humans = self.human_tracker.update_with_detections(human_detections)
                
                # Update tracking stats
                if len(tracked_horses) > 0:
                    stats['active_horse_tracks'].update(tracked_horses.tracker_id)
                if len(tracked_humans) > 0:
                    stats['active_human_tracks'].update(tracked_humans.tracker_id)
            else:
                tracked_horses = horse_detections
                tracked_humans = human_detections
            
            # Filter humans to jockeys only (using tracked detections)
            if sv:
                jockey_detections = self.detection_manager.filter_jockeys(tracked_humans, tracked_horses)
            else:
                jockey_detections = tracked_humans
            
            # 🔥 STEP 4: Apply Re-identification Pipeline
            depth_map = None  # Initialize depth_map
            if self.reid_pipeline and self.config.enable_reid_pipeline:
                # Process horses with re-identification (new logic: full image depth + bbox SAM)
                horse_rgb_crops, horse_depth_crops, depth_map, horse_reid_features = self.reid_pipeline.process_frame(frame, tracked_horses)
                tracked_horses_enhanced = self.reid_pipeline.enhance_tracking(tracked_horses, horse_reid_features)
                
                # Process jockeys with re-identification  
                jockey_rgb_crops, jockey_depth_crops, _, jockey_reid_features = self.reid_pipeline.process_frame(frame, jockey_detections)
                jockey_detections_enhanced = self.reid_pipeline.enhance_tracking(jockey_detections, jockey_reid_features)
                
                # Update detections with enhanced tracking
                tracked_horses = tracked_horses_enhanced
                jockey_detections = jockey_detections_enhanced
            
            stats['humans_detected'] += len(jockey_detections) if sv else len(jockey_detections)
            stats['horses_detected'] += len(tracked_horses) if sv else len(tracked_horses)
            stats['tracked_horses'] = len(stats['active_horse_tracks'])
            stats['tracked_humans'] = len(stats['active_human_tracks'])
            
            # 🔥 STEP 5: Estimate poses based on config
            human_poses = self.pose_manager.estimate_human_poses(frame, jockey_detections)
            horse_poses = self.pose_manager.estimate_horse_poses(frame, tracked_horses)
            
            # 🔥 STEP 6: Associate poses with track IDs
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
            
            # 🔥 STEP 7: Visualize everything with tracking and re-ID
            frame = self.visualizer.annotate_detections_with_tracking(frame, jockey_detections, tracked_horses)
            
            # Draw poses with track IDs
            for pose_result in human_poses:
                frame = self.visualizer.draw_human_pose_with_tracking(frame, pose_result)
            
            for pose_result in horse_poses:
                frame = self.visualizer.draw_horse_pose_with_tracking(frame, pose_result)
            
            # Add pose method labels
            frame = self.visualizer.draw_pose_labels(frame, horse_poses)
            
            # Add depth visualization if available
            if self.reid_pipeline and self.config.enable_reid_pipeline and depth_map is not None:
                # Add small depth map overlay
                depth_display = cv2.resize(depth_map, (200, 150))
                depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                frame[10:160, self.width-210:self.width-10] = depth_colored
                cv2.putText(frame, "Depth", (self.width-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            # Add info overlay with tracking and re-ID stats
            human_count = len(jockey_detections) if sv else len(jockey_detections)
            horse_count = len(tracked_horses) if sv else len(tracked_horses)
            
            frame = self.visualizer.draw_info_overlay_with_tracking(
                frame, frame_count, max_frames, human_count, horse_count,
                len(human_poses), len(horse_poses), stats, self.expected_horses, self.expected_jockeys
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
            
            # Update progress bar (only if available)
            if pbar:
                reid_status = "Re-ID:ON" if self.config.enable_reid_pipeline else "Re-ID:OFF"
                pbar.set_postfix_str(f"H:{human_count}/{self.expected_jockeys} Ho:{horse_count}/{self.expected_horses} T:H{stats['tracked_humans']}+Ho{stats['tracked_horses']} {reid_status}")
                pbar.update(1)
        
        self.cap.release()
        out.release()
        
        if self.config.display:
            cv2.destroyAllWindows()
        
        if pbar:
            pbar.close()
        
        print(f"✅ Processing complete!")
        print(f"📊 Final Stats:")
        print(f"   Expected: {self.expected_horses} horses, {self.expected_jockeys} jockeys")
        print(f"   Humans detected: {stats['humans_detected']}")
        print(f"   Horses detected: {stats['horses_detected']}")
        print(f"   Human poses: {stats['human_poses']}")
        print(f"   Horse poses: {stats['horse_poses']}")
        print(f"   🔄 Unique tracks: {stats['tracked_humans']} humans, {stats['tracked_horses']} horses")
        
        if self.config.enable_reid_pipeline:
            print(f"🔍 Re-identification enabled:")
            print(f"   - Track reassignments: {stats.get('reid_reassignments', 0)}")
        
        if self.config.horse_pose_estimator == 'dual':
            print(f"🥊 Competition Results:")
            print(f"   SuperAnimal wins: {stats['superanimal_wins']} (39 keypoints)")
            print(f"   ViTPose wins: {stats['vitpose_wins']} (17 keypoints)")
        
        print(f"🎯 Output: {output_path}")
        
        return output_path

def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python main.py config.yaml")
        sys.exit(1)
    
    # Load config from file
    config_file = sys.argv[1]
    config = Config(config_file)
    
    # Debug: print what was loaded
    print(f"Debug: video_path = {getattr(config, 'video_path', 'NOT FOUND')}")
    
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

if __name__ == "__main__":
    main()