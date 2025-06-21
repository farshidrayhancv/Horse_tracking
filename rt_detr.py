"""
Side-by-side tracker comparison: BoostTrack vs ByteTrack vs OCSORT
Creates a single video with 3 frames side by side showing each tracker's results
"""

import cv2
import numpy as np
import torch
import time
from pathlib import Path
import json
import random

from ultralytics import YOLO

# Import trackers
try:
    from boxmot import BoostTrack, ByteTrack, OcSort
    print("✅ BoxMOT trackers imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

class TrackerVideoComparison:
    def __init__(self, video_path: str, output_path: str = None):
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Output video path
        if output_path is None:
            output_path = self.video_path.parent / f"{self.video_path.stem}_tracker_comparison.mp4"
        self.output_path = Path(output_path)
        
        # Setup YOLO detector
        print("🔄 Loading YOLOv11...")
        self.model = YOLO("yolo11x.pt")
        
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"💻 Using device: {self.device}")

        # Color scheme for each tracker
        self.tracker_colors = {
            'BoostTrack': (0, 255, 0),    # Green
            'ByteTrack': (255, 0, 0),     # Blue  
            'OCSORT': (0, 0, 255)         # Red
        }
        
        # Track ID color cache for consistent colors per ID
        self.id_colors = {}
        
        # Setup video writer for side-by-side output
        self.setup_video_writer()
        
        print(f"📺 Input: {self.width}x{self.height} @ {self.fps} FPS ({self.total_frames} frames)")
        print(f"📹 Output: {self.output_width}x{self.output_height} @ {self.fps} FPS")
    
    def setup_video_writer(self):
        """Setup video writer for side-by-side output"""
        # Calculate output dimensions (3 frames side by side + borders)
        border_width = 10
        self.frame_width = self.width // 3  # Resize each frame to 1/3 width
        self.frame_height = int(self.height * (self.frame_width / self.width))  # Maintain aspect ratio
        
        self.output_width = (self.frame_width * 3) + (border_width * 4)  # 3 frames + 4 borders
        self.output_height = self.frame_height + (border_width * 2) + 60  # Extra space for labels
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(self.output_path), 
            fourcc, 
            self.fps, 
            (self.output_width, self.output_height)
        )
    
    def get_tracker_color(self, track_id, tracker_name):
        """Get consistent color for track ID"""
        key = f"{tracker_name}_{track_id}"
        if key not in self.id_colors:
            # Generate random color but keep it bright
            self.id_colors[key] = (
                random.randint(100, 255),
                random.randint(100, 255), 
                random.randint(100, 255)
            )
        return self.id_colors[key]
    
    def detect_humans(self, frame):
        """Detect humans using YOLOv11"""
        results = self.model(frame, conf=0.5, device=self.device, verbose=False, classes=[0])[0]
        
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                
                if cls == 0:  # Person class
                    detections.append([x1, y1, x2, y2, conf, cls])
        
        return np.array(detections) if detections else np.empty((0, 6))
    
    def draw_tracks(self, frame, tracks, tracker_name):
        """Draw tracking results on frame"""
        frame_copy = frame.copy()
        
        if tracks is not None and len(tracks) > 0:
            for track in tracks:
                if len(track) >= 5:
                    x1, y1, x2, y2, track_id = track[:5]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    track_id = int(track_id)
                    
                    if track_id >= 0:
                        # Get color for this track
                        color = self.get_tracker_color(track_id, tracker_name)
                        
                        # Draw bounding box
                        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw track ID
                        label = f"ID:{track_id}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(frame_copy, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame_copy
    
    def create_side_by_side_frame(self, frames, stats):
        """Create side-by-side frame with labels"""
        border_width = 10
        
        # Create output canvas
        canvas = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
        canvas.fill(50)  # Dark gray background
        
        # Resize and place frames
        tracker_names = ['BoostTrack', 'ByteTrack', 'OCSORT']
        
        for i, (frame, tracker_name) in enumerate(zip(frames, tracker_names)):
            # Resize frame
            resized_frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            
            # Calculate position
            x_pos = border_width + i * (self.frame_width + border_width)
            y_pos = border_width + 50  # Leave space for title
            
            # Place frame
            canvas[y_pos:y_pos + self.frame_height, 
                   x_pos:x_pos + self.frame_width] = resized_frame
            
            # Add tracker name label
            label = tracker_name
            font_scale = 0.8
            thickness = 2
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            
            # Center the label above the frame
            label_x = x_pos + (self.frame_width - label_size[0]) // 2
            label_y = 30
            
            # Add background for label
            cv2.rectangle(canvas, (label_x - 5, label_y - label_size[1] - 5), 
                         (label_x + label_size[0] + 5, label_y + 5), 
                         self.tracker_colors[tracker_name], -1)
            
            cv2.putText(canvas, label, (label_x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            # Add stats below frame
            if tracker_name in stats:
                stat_text = f"IDs: {stats[tracker_name]['unique_ids']} | Det: {stats[tracker_name]['total_detections']}"
                stat_size = cv2.getTextSize(stat_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                stat_x = x_pos + (self.frame_width - stat_size[0]) // 2
                stat_y = y_pos + self.frame_height + 25
                
                cv2.putText(canvas, stat_text, (stat_x, stat_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return canvas
    
    def run_comparison(self):
        """Run side-by-side tracker comparison and save video"""
        print(f"\n🚀 Starting side-by-side tracker comparison")
        print(f"📁 Input: {self.video_path}")
        print(f"📹 Output: {self.output_path}")
        print("=" * 80)
        
        # Initialize trackers with your original configurations
        reid_model_name = "osnet_x0_25_msmt17.pt"
        reid_model = Path(reid_model_name)
        
        trackers = {}
        tracker_configs = {
            'BoostTrack': {
                'class': BoostTrack,
                'params': {
                    'reid_weights': reid_model,
                    'half': False,
                    'device': 0,
                    'with_reid': True,
                    'use_rich_s':  True,
                    'use_sb': True,
                    'use_vt': True,
                    'with_reid': True,
                }
            },
            'ByteTrack': {
                'class': ByteTrack,
                'params': {
                    'per_class': True,
                }
            },
            'OCSORT': {
                'class': OcSort,
                'params': {
                    'per_class': True,
                    'min_conf': 0.4,
                    'det_thresh': 0.4,
                    'max_age': 30,
                    'min_hits': 5,
                }
            }
        }
        
        # Initialize all trackers
        for name, config in tracker_configs.items():
            try:
                tracker = config['class'](**config['params'])
                trackers[name] = {
                    'tracker': tracker,
                    'unique_ids': set(),
                    'total_detections': 0
                }
                print(f"  ✅ {name} initialized")
            except Exception as e:
                print(f"  ❌ {name} failed: {e}")
                trackers[name] = None
        
        # Remove failed trackers
        trackers = {k: v for k, v in trackers.items() if v is not None}
        
        if len(trackers) == 0:
            print("❌ No trackers initialized successfully")
            return
        
        frame_count = 0
        processing_times = []
        
        print(f"\n🎬 Processing {self.total_frames} frames...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # Detect humans once for all trackers
            detections = self.detect_humans(frame)
            
            # Run all trackers on the same frame
            annotated_frames = []
            current_stats = {}
            
            for name, tracker_data in trackers.items():
                tracker = tracker_data['tracker']
                
                # Track
                tracks = None
                if len(detections) > 0:
                    try:
                        tracks = tracker.update(detections, frame)
                        
                        # Update statistics
                        if tracks is not None and len(tracks) > 0:
                            for track in tracks:
                                if len(track) >= 5:
                                    track_id = int(track[4])
                                    if track_id >= 0:
                                        tracker_data['unique_ids'].add(track_id)
                                        tracker_data['total_detections'] += 1
                    except Exception as e:
                        print(f"⚠️ {name} tracking error on frame {frame_count}: {e}")
                        tracks = None
                
                # Draw annotations
                annotated_frame = self.draw_tracks(frame, tracks, name)
                annotated_frames.append(annotated_frame)
                
                # Prepare stats for display
                current_stats[name] = {
                    'unique_ids': len(tracker_data['unique_ids']),
                    'total_detections': tracker_data['total_detections']
                }
            
            # Create side-by-side frame
            combined_frame = self.create_side_by_side_frame(annotated_frames, current_stats)
            
            # Write frame
            self.video_writer.write(combined_frame)
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            frame_count += 1
            
            # Progress update
            if frame_count % 100 == 0:
                fps = 1.0 / processing_time if processing_time > 0 else 0
                progress = (frame_count / self.total_frames) * 100
                print(f"    Progress: {progress:.1f}% | Frame {frame_count}/{self.total_frames} | FPS: {fps:.1f}")
        
        # Cleanup
        self.cap.release()
        self.video_writer.release()
        
        # Final results
        print(f"\n🏆 COMPARISON RESULTS")
        print("=" * 60)
        print(f"{'Tracker':<12} {'Unique IDs':<12} {'Total Detections':<18}")
        print("-" * 50)
        
        results = []
        for name, tracker_data in trackers.items():
            unique_count = len(tracker_data['unique_ids'])
            total_count = tracker_data['total_detections']
            results.append({
                'name': name,
                'unique_ids': unique_count,
                'total_detections': total_count
            })
            print(f"{name:<12} {unique_count:<12} {total_count:<18}")
        
        # Find winner
        if results:
            winner = max(results, key=lambda x: x['unique_ids'])
            print(f"\n🥇 WINNER: {winner['name']} with {winner['unique_ids']} unique IDs")
        
        avg_fps = 1.0 / np.mean(processing_times) if processing_times else 0
        print(f"\n📊 Average processing speed: {avg_fps:.1f} FPS")
        print(f"💾 Comparison video saved: {self.output_path}")
        
        # Save JSON results
        results_file = self.output_path.parent / f"{self.output_path.stem}_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'input_video': str(self.video_path),
                'output_video': str(self.output_path),
                'results': results,
                'winner': winner['name'] if results else None,
                'avg_fps': round(avg_fps, 2),
                'total_frames': frame_count
            }, f, indent=2)
        
        print(f"📋 Results saved: {results_file}")
        return results

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python tracker_video_comparison.py input_video.mp4 [output_video.mp4]")
        print("Creates side-by-side comparison video of BoostTrack vs ByteTrack vs OCSORT")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)
    
    try:
        comparison = TrackerVideoComparison(video_path, output_path)
        comparison.run_comparison()
        print("\n✅ Comparison completed successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()