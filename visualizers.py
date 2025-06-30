import cv2
import numpy as np
from typing import Dict

try:
    import supervision as sv
except ImportError:
    sv = None

class Visualizer:
    def __init__(self, config, superanimal_model=None):
        self.config = config
        self.superanimal = superanimal_model
        
        # Setup automatic color assignment using supervision
        self.color_palette = sv.ColorPalette.DEFAULT if sv else None
        
        # Fixed colors for elements
        self.keypoint_color = (0, 255, 255)  # Cyan
        self.text_color = (255, 255, 255)    # White
        self.track_color = (0, 255, 0)       # Bright Green for track IDs
        
        # Setup annotators with automatic track-based coloring
        if sv:
            self.triangle_annotator = sv.TriangleAnnotator(
                base=20, height=25,  # Bigger triangles for horses
                color_lookup=sv.ColorLookup.TRACK
            )
    
    def get_track_color(self, track_id):
        """Get unique color for a track ID using supervision's color palette"""
        if sv and self.color_palette and track_id >= 0:
            color = self.color_palette.by_idx(track_id)
            return (color.b, color.g, color.r)  # Convert RGB to BGR for OpenCV
        else:
            # Fallback colors for untracked objects
            fallback_colors = [
                (0, 255, 0),    # Green
                (255, 0, 0),    # Red  
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
                (255, 255, 0),  # Yellow
                (255, 165, 0),  # Orange
                (128, 0, 128),  # Purple
                (255, 192, 203) # Pink
            ]
            return fallback_colors[abs(track_id) % len(fallback_colors)] if track_id >= 0 else (128, 128, 128)
    
    def draw_horse_pose_with_tracking(self, frame: np.ndarray, pose_result: Dict, min_confidence: float = None):
        """Draw horse pose with BIG track ID numbers"""
        frame = self.draw_horse_pose(frame, pose_result, min_confidence)
        
        # Add BIG track ID if available
        if 'track_id' in pose_result and 'keypoints' in pose_result:
            track_id = pose_result['track_id']
            if track_id >= 0:
                keypoints = pose_result['keypoints']
                
                # Find head position (nose or highest point)
                head_pos = None
                for i in [0, 1, 2]:  # nose, upper_jaw, lower_jaw
                    if i < len(keypoints) and keypoints[i][0] != -1 and keypoints[i][2] > 0:
                        head_pos = (int(keypoints[i][0]), int(keypoints[i][1]) - 40)  # Higher offset
                        break
                
                if head_pos:
                    # BIGGER FONT for horse numbers
                    font_scale = 1.5  # Much bigger
                    thickness = 6     # Much thicker
                    
                    # Draw number with OCR-detected ID
                    cv2.putText(frame, f"{track_id}", head_pos, 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 4)  # Black outline
                    cv2.putText(frame, f"{track_id}", head_pos, 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.track_color, thickness)  # Green number
        
        return frame
    
    def draw_horse_pose(self, frame: np.ndarray, pose_result: Dict, min_confidence: float = None):
        """Draw horse pose keypoints with track-based colors"""
        if 'keypoints' not in pose_result or 'method' not in pose_result:
            return frame
        
        keypoints = pose_result['keypoints']
        method = pose_result['method']
        
        # Get unique color based on track ID
        track_id = pose_result.get('track_id', -1)
        color = self.get_track_color(track_id)
        
        # Choose skeleton based on method
        if method == 'SuperAnimal':
            skeleton = self.superanimal.skeleton if self.superanimal else []
        else:  # ViTPose
            skeleton = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                (5, 11), (6, 12), (11, 12),  # Torso
                (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
            ]
        
        # Draw skeleton - only valid keypoints
        for start_idx, end_idx in skeleton:
            if (start_idx < len(keypoints) and end_idx < len(keypoints)):
                start_valid = keypoints[start_idx][0] != -1 and keypoints[start_idx][2] > 0
                end_valid = keypoints[end_idx][0] != -1 and keypoints[end_idx][2] > 0
                
                if start_valid and end_valid:
                    start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                    end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                    cv2.line(frame, start_point, end_point, color, 3)  # Thicker lines
        
        # Draw keypoints - only valid ones
        for i, kpt in enumerate(keypoints):
            x, y, conf = kpt
            if x != -1 and y != -1 and conf > 0:
                center = (int(x), int(y))
                cv2.circle(frame, center, 6, self.keypoint_color, -1)  # Bigger keypoints
                cv2.circle(frame, center, 7, color, 2)  # Bigger outline
        
        return frame
    
    def annotate_detections_with_tracking(self, frame: np.ndarray, human_detections, horse_detections):
        """Annotate horse detections only with BIG numbers"""
        if not sv:
            return frame
        
        try:
            # Horse detections with automatic track-based colors (triangles)
            if len(horse_detections) > 0:
                frame = self.triangle_annotator.annotate(frame, horse_detections)
                
                # Add BIG track ID labels
                if hasattr(horse_detections, 'tracker_id'):
                    for i, (box, track_id) in enumerate(zip(horse_detections.xyxy, horse_detections.tracker_id)):
                        x1, y1, x2, y2 = box.astype(int)
                        
                        # BIGGER FONT for horse numbers
                        font_scale = 1.2
                        thickness = 5
                        
                        # Draw number above bounding box
                        label_pos = (x1, y1-10)
                        cv2.putText(frame, f"{track_id}", label_pos, 
                                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3)  # Black outline
                        cv2.putText(frame, f"{track_id}", label_pos, 
                                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.track_color, thickness)  # Green number
                
        except Exception as e:
            print(f"⚠️ Supervision annotation failed, using fallback: {e}")
            # Fallback to manual rectangles
            if len(horse_detections) > 0:
                for i, box in enumerate(horse_detections.xyxy):
                    x1, y1, x2, y2 = box.astype(int)
                    track_id = horse_detections.tracker_id[i] if hasattr(horse_detections, 'tracker_id') and i < len(horse_detections.tracker_id) else i+1
                    color = self.get_track_color(track_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)  # Thicker rectangle
                    
                    # BIG number
                    cv2.putText(frame, f"{track_id}", (x1, y1-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 8)  # Black outline
                    cv2.putText(frame, f"{track_id}", (x1, y1-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.track_color, 5)  # Green number
        
        return frame
    
    def draw_pose_labels(self, frame: np.ndarray, horse_poses: list):
        """Draw pose method labels with track-based colors"""
        for pose_result in horse_poses:
            if 'box' in pose_result and 'method' in pose_result:
                box = pose_result['box']
                method = pose_result['method']
                conf = pose_result.get('confidence', 0)
                track_id = pose_result.get('track_id', -1)
                x1, y1, x2, y2 = box.astype(int)
                
                # Use track-based color
                color = self.get_track_color(track_id)
                kp_count = "39kp" if method == 'SuperAnimal' else "17kp"
                
                track_str = f" #{track_id}" if track_id >= 0 else ""
                cv2.putText(frame, f"{method} {conf:.2f} ({kp_count}){track_str}", 
                           (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame