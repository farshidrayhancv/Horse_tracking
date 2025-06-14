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
        
        # Fixed colors for non-tracking elements
        self.keypoint_color = (0, 255, 255) # Cyan
        self.text_color = (255, 255, 255)   # White
        self.track_color = (255, 255, 0)    # Yellow for track IDs
        self.reid_color = (0, 255, 0)       # Green for re-ID indicators
        
        # Setup annotators with automatic track-based coloring
        if sv:
            self.triangle_annotator = sv.TriangleAnnotator(
                base=15, height=20, 
                color_lookup=sv.ColorLookup.TRACK
            )
        
        # Define skeletons
        self.human_skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
    
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
    
    def draw_human_pose(self, frame: np.ndarray, pose_result: Dict, min_confidence: float = None):
        """
        🔥 SIMPLIFIED: Just draws keypoints with track-based colors - confidence filtering already done at source
        """
        if 'keypoints' not in pose_result or 'scores' not in pose_result:
            return frame
        
        # Get unique color based on track ID
        track_id = pose_result.get('track_id', -1)
        color = self.get_track_color(track_id)
        
        keypoints = pose_result['keypoints'].cpu().numpy() if hasattr(pose_result['keypoints'], 'cpu') else pose_result['keypoints']
        scores = pose_result['scores'].cpu().numpy() if hasattr(pose_result['scores'], 'cpu') else pose_result['scores']
        
        # Draw skeleton - only valid keypoints (confidence > 0)
        for start_idx, end_idx in self.human_skeleton:
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                scores[start_idx] > 0 and scores[end_idx] > 0):
                start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                cv2.line(frame, start_point, end_point, color, 2)
        
        # Draw keypoints - only valid ones (confidence > 0)
        valid_count = 0
        for i, (kpt, score) in enumerate(zip(keypoints, scores)):
            if score > 0:  # Valid keypoint (already filtered at source)
                center = (int(kpt[0]), int(kpt[1]))
                cv2.circle(frame, center, 4, self.keypoint_color, -1)
                cv2.circle(frame, center, 5, color, 1)
                valid_count += 1
        
        return frame
    
    def draw_human_pose_with_tracking(self, frame: np.ndarray, pose_result: Dict, min_confidence: float = None):
        """Draw human pose with track ID using track-based colors"""
        frame = self.draw_human_pose(frame, pose_result, min_confidence)
        
        # Add track ID if available
        if 'track_id' in pose_result and 'keypoints' in pose_result:
            track_id = pose_result['track_id']
            if track_id >= 0:
                keypoints = pose_result['keypoints'].cpu().numpy() if hasattr(pose_result['keypoints'], 'cpu') else pose_result['keypoints']
                scores = pose_result['scores'].cpu().numpy() if hasattr(pose_result['scores'], 'cpu') else pose_result['scores']
                
                # Find head position (nose/top of head)
                head_pos = None
                for i in [0, 1, 2]:  # nose, left_eye, right_eye
                    if i < len(keypoints) and scores[i] > 0:
                        head_pos = (int(keypoints[i][0]), int(keypoints[i][1]) - 20)
                        break
                
                if head_pos:
                    cv2.putText(frame, f"J{track_id}", head_pos, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.track_color, 2)
        
        return frame
    
    def draw_horse_pose(self, frame: np.ndarray, pose_result: Dict, min_confidence: float = None):
        """
        🔥 SIMPLIFIED: Just draws keypoints with track-based colors - confidence filtering already done at source
        """
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
            skeleton = self.human_skeleton  # Use human skeleton for ViTPose horses
        
        # Draw skeleton - only valid keypoints (coordinates != -1, confidence > 0)
        for start_idx, end_idx in skeleton:
            if (start_idx < len(keypoints) and end_idx < len(keypoints)):
                start_valid = keypoints[start_idx][0] != -1 and keypoints[start_idx][2] > 0
                end_valid = keypoints[end_idx][0] != -1 and keypoints[end_idx][2] > 0
                
                if start_valid and end_valid:
                    start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                    end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                    cv2.line(frame, start_point, end_point, color, 2)
        
        # Draw keypoints - only valid ones (coordinates != -1, confidence > 0)
        valid_count = 0
        for i, kpt in enumerate(keypoints):
            x, y, conf = kpt
            if x != -1 and y != -1 and conf > 0:  # Valid keypoint (already filtered at source)
                center = (int(x), int(y))
                cv2.circle(frame, center, 4, self.keypoint_color, -1)
                cv2.circle(frame, center, 5, color, 1)
                valid_count += 1
        
        return frame
    
    def draw_horse_pose_with_tracking(self, frame: np.ndarray, pose_result: Dict, min_confidence: float = None):
        """Draw horse pose with track ID using track-based colors"""
        frame = self.draw_horse_pose(frame, pose_result, min_confidence)
        
        # Add track ID if available
        if 'track_id' in pose_result and 'keypoints' in pose_result:
            track_id = pose_result['track_id']
            if track_id >= 0:
                keypoints = pose_result['keypoints']
                
                # Find head position (nose or highest point)
                head_pos = None
                for i in [0, 1, 2]:  # nose, upper_jaw, lower_jaw
                    if i < len(keypoints) and keypoints[i][0] != -1 and keypoints[i][2] > 0:
                        head_pos = (int(keypoints[i][0]), int(keypoints[i][1]) - 25)
                        break
                
                if head_pos:
                    method = pose_result.get('method', 'Horse')
                    prefix = "SA" if method == 'SuperAnimal' else "VP"
                    cv2.putText(frame, f"{prefix}{track_id}", head_pos, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.track_color, 2)
        
        return frame
    
    def annotate_detections(self, frame: np.ndarray, human_detections, horse_detections):
        """Legacy method - use annotate_detections_with_tracking for new code"""
        return self.annotate_detections_with_tracking(frame, human_detections, horse_detections)
    
    def annotate_detections_with_tracking(self, frame: np.ndarray, human_detections, horse_detections):
        """Annotate detections with automatic track-based colors using supervision"""
        if not sv:
            return frame
        
        try:
            # Human detections with automatic track-based colors (triangles)
            if len(human_detections) > 0:
                frame = self.triangle_annotator.annotate(frame, human_detections)
                
                # Add track ID labels with re-ID indicator
                if hasattr(human_detections, 'tracker_id'):
                    for i, (box, track_id) in enumerate(zip(human_detections.xyxy, human_detections.tracker_id)):
                        x1, y1, x2, y2 = box.astype(int)
                        
                        # Check if this was re-identified (you can add a flag to detections)
                        reid_indicator = ""
                        if hasattr(human_detections, 'reid_flag') and human_detections.reid_flag[i]:
                            reid_indicator = "*"
                            cv2.circle(frame, (x1+10, y1+10), 5, self.reid_color, -1)
                        
                        cv2.putText(frame, f"J{track_id}{reid_indicator}", (x1, y1-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.track_color, 2)
            
            # Horse detections with automatic track-based colors (triangles)
            if len(horse_detections) > 0:
                frame = self.triangle_annotator.annotate(frame, horse_detections)
                
                # Add track ID labels with re-ID indicator
                if hasattr(horse_detections, 'tracker_id'):
                    for i, (box, track_id) in enumerate(zip(horse_detections.xyxy, horse_detections.tracker_id)):
                        x1, y1, x2, y2 = box.astype(int)
                        
                        # Check if this was re-identified
                        reid_indicator = ""
                        if hasattr(horse_detections, 'reid_flag') and horse_detections.reid_flag[i]:
                            reid_indicator = "*"
                            cv2.circle(frame, (x1+10, y1+10), 5, self.reid_color, -1)
                        
                        cv2.putText(frame, f"H{track_id}{reid_indicator}", (x1, y1-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.track_color, 2)
                
        except Exception as e:
            print(f"⚠️ Supervision annotation failed, using fallback: {e}")
            # Fallback to manual rectangles with track-based colors
            if len(human_detections) > 0:
                for i, box in enumerate(human_detections.xyxy):
                    x1, y1, x2, y2 = box.astype(int)
                    track_id = human_detections.tracker_id[i] if hasattr(human_detections, 'tracker_id') and i < len(human_detections.tracker_id) else i+1
                    color = self.get_track_color(track_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"J{track_id}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.track_color, 2)
            
            if len(horse_detections) > 0:
                for i, box in enumerate(horse_detections.xyxy):
                    x1, y1, x2, y2 = box.astype(int)
                    track_id = horse_detections.tracker_id[i] if hasattr(horse_detections, 'tracker_id') and i < len(horse_detections.tracker_id) else i+1
                    color = self.get_track_color(track_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"H{track_id}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.track_color, 2)
        
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
                
                # Use track-based color instead of method-based color
                color = self.get_track_color(track_id)
                kp_count = "39kp" if method == 'SuperAnimal' else "17kp"
                
                track_str = f" T{track_id}" if track_id >= 0 else ""
                cv2.putText(frame, f"{method} {conf:.2f} ({kp_count}){track_str}", 
                           (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def draw_info_overlay(self, frame: np.ndarray, frame_count: int, max_frames: int, 
                         human_count: int, horse_count: int, human_poses: int, 
                         horse_poses: int, stats: dict = None):
        """Legacy method - use draw_info_overlay_with_tracking for new code"""
        return self.draw_info_overlay_with_tracking(frame, frame_count, max_frames, 
                                                   human_count, horse_count, human_poses, 
                                                   horse_poses, stats, 10, 10)
    
    def draw_info_overlay_with_tracking(self, frame: np.ndarray, frame_count: int, max_frames: int, 
                                       human_count: int, horse_count: int, human_poses: int, 
                                       horse_poses: int, stats: dict = None, 
                                       expected_horses: int = 10, expected_jockeys: int = 10):
        """Draw info overlay with RGB-D tracking and re-identification statistics"""
        total_display = str(max_frames) if max_frames != float('inf') else "∞"
        
        info_lines = [
            f"Frame: {frame_count+1}/{total_display}",
            f"Expected: {expected_horses} horses, {expected_jockeys} jockeys",
            f"Config: H-Det:{self.config.human_detector} H-Pose:{self.config.human_pose_estimator}",
            f"        Horse-Det:{self.config.horse_detector} Horse-Pose:{self.config.horse_pose_estimator}",
            f"Detected - Jockeys:{human_count}/{expected_jockeys} Horses:{horse_count}/{expected_horses}",
            f"Poses - Humans:{human_poses} Horses:{horse_poses}",
        ]
        
        # Add tracking statistics
        if stats:
            tracked_humans = stats.get('tracked_humans', 0)
            tracked_horses = stats.get('tracked_horses', 0)
            info_lines.append(f"🔄 Unique Tracks - Humans:{tracked_humans} Horses:{tracked_horses}")
        
        # Add RGB-D re-identification info with SAM model information
        if self.config.enable_reid_pipeline:
            sam_model_display = self.config.SAM_MODELS.get(self.config.sam_model, self.config.sam_model)
            info_lines.append(f"🔍 RGB-D Re-ID Pipeline: FullImage-Depth → BBox-{sam_model_display} → RGB-D-MegaDescriptor")
            reid_reassignments = stats.get('reid_reassignments', 0) if stats else 0
            info_lines.append(f"   Track reassignments: {reid_reassignments} | Threshold: {self.config.reid_similarity_threshold}")
            info_lines.append(f"   RGB-D Fusion: RGB(70%) + Depth(30%) + Shape consistency")
            info_lines.append(f"   Components: Depth-Anything:{self.config.enable_depth_anything} SAM:{self.config.sam_model.upper()} ReID:{self.config.enable_megadescriptor}")
        else:
            info_lines.append(f"🔍 RGB-D Re-ID Pipeline: DISABLED")
        
        info_lines.append(f"🎨 Auto Colors: Each track gets unique color from supervision palette")
        info_lines.append(f"SOURCE Filtering: Human:{self.config.confidence_human_pose} Horse-ViTPose:{self.config.confidence_horse_pose_vitpose}")
        
        if self.config.horse_pose_estimator == 'dual' and stats:
            info_lines.append(f"Competition - SuperAnimal:{stats.get('superanimal_wins', 0)} ViTPose:{stats.get('vitpose_wins', 0)}")
        
        if self.config.display:
            info_lines.append(f"Controls: SPACE=Pause Q=Quit | Green dots = Re-identified tracks")
            sam_info = "SAM2" if self.config.sam_model == 'sam2' else "MobileSAM" if self.config.sam_model == 'mobilesam' else "No SAM"
            info_lines.append(f"Visualizations: Top-right=Depth+{sam_info} Masks, Bottom-left=Segmentation only")
        
        # Semi-transparent background
        overlay = frame.copy()
        overlay_height = 25 + len(info_lines) * 18
        cv2.rectangle(overlay, (5, 5), (1000, overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        for i, line in enumerate(info_lines):
            y_pos = 25 + i * 18
            cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        
        return frame