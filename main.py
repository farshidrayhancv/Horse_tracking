import cv2
import numpy as np
import torch
import warnings
from pathlib import Path

# Local imports
from config import Config
from models import SuperAnimalQuadruped
from detectors import DetectionManager
from pose_estimators import PoseEstimationManager
from visualizers import Visualizer

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
    print("✓ Supervision 0.25.1 for professional visualizations and Detection class")
except ImportError:
    print("⚠️ Supervision not available - install with: pip install supervision")
    sv = None

class HybridPoseSystem:
    def __init__(self, video_path: str, config: Config):
        self.video_path = Path(video_path)
        self.config = config
        
        # Setup video
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.total_frames <= 0:
            print("⚠️ Warning: Frame count not available, will process until end of video")
            self.total_frames = float('inf')
        
        # Setup models and components
        self.setup_models()
        
        # Print configuration
        self.config.print_config()
        print(f"🐴🏇 Configurable Pose System ready: {self.total_frames} frames")
    
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
    
    def process_video(self):
        # Determine output path
        if self.config.output_path:
            output_path = self.config.output_path
        else:
            # Create safe output filename
            input_stem = self.video_path.stem if self.video_path.suffix else self.video_path.name
            output_path = str(self.video_path.parent / f"{input_stem}_pose_output.mp4")
        
        print(f"🎬 Processing: {self.video_path}")
        print(f"📤 Output: {output_path}")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        out = cv2.VideoWriter(output_path, fourcc, fps, (self.width, self.height))
        
        frame_count = 0
        max_frames = self.config.max_frames or (self.total_frames if self.total_frames != float('inf') else 10000)
        paused = False
        
        stats = {
            'humans_detected': 0, 'horses_detected': 0,
            'human_poses': 0, 'horse_poses': 0,
            'superanimal_wins': 0, 'vitpose_wins': 0
        }
        
        # Initialize progress bar (only if not displaying)
        if TQDM_AVAILABLE and not self.config.display and self.total_frames != float('inf'):
            pbar = tqdm(total=max_frames, desc="Processing pose estimation", 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
        else:
            pbar = None
        
        # Setup display window
        if self.config.display:
            window_name = "Configurable Pose System"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            display_width = min(1200, self.width)
            display_height = int(self.height * (display_width / self.width))
            cv2.resizeWindow(window_name, display_width, display_height)
        
        while frame_count < max_frames:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Detect objects based on config
            human_detections = self.detection_manager.detect_humans(frame)
            horse_detections = self.detection_manager.detect_horses(frame)
            
            # Filter humans to jockeys only
            if sv:
                jockey_detections = self.detection_manager.filter_jockeys(human_detections, horse_detections)
            else:
                jockey_detections = human_detections
            
            stats['humans_detected'] += len(jockey_detections) if sv else len(jockey_detections)
            stats['horses_detected'] += len(horse_detections) if sv else len(horse_detections)
            
            # 🔥 Estimate poses based on config - SOURCE-LEVEL FILTERING HAPPENS HERE
            human_poses = self.pose_manager.estimate_human_poses(frame, jockey_detections)
            horse_poses = self.pose_manager.estimate_horse_poses(frame, horse_detections)
            
            stats['human_poses'] += len(human_poses)
            stats['horse_poses'] += len(horse_poses)
            
            # Count method wins for dual mode
            if self.config.horse_pose_estimator == 'dual':
                superanimal_count = sum(1 for pose in horse_poses if pose.get('method') == 'SuperAnimal')
                vitpose_count = sum(1 for pose in horse_poses if pose.get('method') == 'ViTPose')
                stats['superanimal_wins'] += superanimal_count
                stats['vitpose_wins'] += vitpose_count
            
            # 🔥 Visualize everything - NO FILTERING NEEDED (already done at source)
            frame = self.visualizer.annotate_detections(frame, jockey_detections, horse_detections)
            
            # Draw poses
            for pose_result in human_poses:
                frame = self.visualizer.draw_human_pose(frame, pose_result)
            
            for pose_result in horse_poses:
                frame = self.visualizer.draw_horse_pose(frame, pose_result)
            
            # Add pose method labels
            frame = self.visualizer.draw_pose_labels(frame, horse_poses)
            
            # Add info overlay
            human_count = len(jockey_detections) if sv else len(jockey_detections)
            horse_count = len(horse_detections) if sv else len(horse_detections)
            
            frame = self.visualizer.draw_info_overlay(
                frame, frame_count, max_frames, human_count, horse_count,
                len(human_poses), len(horse_poses), stats
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
                pbar.set_postfix_str(f"H:{human_count} Horses:{horse_count}")
                pbar.update(1)
        
        self.cap.release()
        out.release()
        
        if self.config.display:
            cv2.destroyAllWindows()
        
        if pbar:
            pbar.close()
        
        print(f"✅ Processing complete!")
        print(f"📊 Final Stats:")
        print(f"   Humans detected: {stats['humans_detected']}")
        print(f"   Horses detected: {stats['horses_detected']}")
        print(f"   Human poses: {stats['human_poses']}")
        print(f"   Horse poses: {stats['horse_poses']}")
        
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