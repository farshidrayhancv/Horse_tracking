import cv2
import numpy as np
from typing import Dict, List

try:
    import supervision as sv
except ImportError:
    sv = None

class SimplifiedVisualizer:
    def __init__(self, config, superanimal_model=None):
        self.config = config
        self.superanimal = superanimal_model
        
        # Setup automatic color assignment using supervision
        self.color_palette = sv.ColorPalette.DEFAULT if sv else None
        
        # Colors for different pose methods
        self.superanimal_color = (0, 0, 255)    # Red for SuperAnimal (39 keypoints)
        self.vitpose_color = (255, 0, 255)      # Magenta for ViTPose (17 keypoints)
        self.keypoint_center_color = (0, 255, 255)  # Cyan for keypoint centers
        self.text_color = (255, 255, 255)       # White
        self.track_id_color = (0, 255, 0)       # Green for track IDs
        
        # Setup annotators with automatic track-based coloring
        if sv:
            # Use new supervision API
            try:
                self.bounding_box_annotator = sv.BoundingBoxAnnotator(
                    color_lookup=sv.ColorLookup.TRACK,
                    thickness=3
                )
                self.label_annotator = sv.LabelAnnotator(
                    text_scale=0.7,
                    text_thickness=2,
                    color_lookup=sv.ColorLookup.TRACK
                )
            except AttributeError:
                # Fallback for older supervision versions
                self.triangle_annotator = sv.TriangleAnnotator(
                    base=25, height=30,
                    color_lookup=sv.ColorLookup.TRACK
                )
                self.label_annotator = sv.LabelAnnotator(
                    text_scale=0.7,
                    text_thickness=2,
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
                (255, 0, 0),    # Blue  
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
                (255, 255, 0),  # Yellow
                (255, 165, 0),  # Orange
                (128, 0, 128),  # Purple
                (255, 192, 203) # Pink
            ]
            return fallback_colors[abs(track_id) % len(fallback_colors)] if track_id >= 0 else (128, 128, 128)
    
    def annotate_racer_detections(self, frame: np.ndarray, detections) -> np.ndarray:
        """Annotate compound racer detections with tracking info"""
        if not sv or len(detections) == 0:
            return frame
        
        try:
            # Try new supervision API first
            if hasattr(self, 'bounding_box_annotator'):
                frame = self.bounding_box_annotator.annotate(frame, detections)
            else:
                # Fallback to triangle annotator
                frame = self.triangle_annotator.annotate(frame, detections)
            
            # Add track ID labels if available
            if hasattr(detections, 'tracker_id'):
                labels = [f"Racer #{track_id}" if track_id >= 0 else "Racer" 
                         for track_id in detections.tracker_id]
                frame = self.label_annotator.annotate(frame, detections, labels)
            
        except Exception as e:
            print(f"⚠️ Supervision annotation failed, using fallback: {e}")
            # Fallback to manual rectangles
            for i, bbox in enumerate(detections.xyxy):
                x1, y1, x2, y2 = bbox.astype(int)
                track_id = detections.tracker_id[i] if hasattr(detections, 'tracker_id') and i < len(detections.tracker_id) else i+1
                color = self.get_track_color(track_id)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw track ID
                cv2.putText(frame, f"Racer #{track_id}", (x1, y1-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 6)  # Black outline
                cv2.putText(frame, f"Racer #{track_id}", (x1, y1-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.track_id_color, 3)  # Green text
        
        return frame
    
    def draw_compound_poses(self, frame: np.ndarray, poses: List[Dict]) -> np.ndarray:
        """Draw all poses (SuperAnimal + ViTPose) on compound entities"""
        for pose in poses:
            frame = self.draw_single_pose(frame, pose)
        
        return frame
    
    def draw_single_pose(self, frame: np.ndarray, pose: Dict) -> np.ndarray:
        """Draw a single pose with method-specific styling"""
        if 'keypoints' not in pose or 'method' not in pose:
            return frame
        
        keypoints = pose['keypoints']
        method = pose['method']
        
        # Choose colors and skeleton based on method
        if method == 'SuperAnimal':
            pose_color = self.superanimal_color
            skeleton = self.superanimal.skeleton if self.superanimal else []
            line_thickness = 2
            keypoint_radius = 3
        else:  # ViTPose
            pose_color = self.vitpose_color
            skeleton = self.get_vitpose_skeleton()
            line_thickness = 2
            keypoint_radius = 3
        
        # Draw skeleton connections
        for start_idx, end_idx in skeleton:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_kpt = keypoints[start_idx]
                end_kpt = keypoints[end_idx]
                
                # Check if both keypoints are valid
                start_valid = len(start_kpt) >= 3 and start_kpt[0] != -1 and start_kpt[2] > 0
                end_valid = len(end_kpt) >= 3 and end_kpt[0] != -1 and end_kpt[2] > 0
                
                if start_valid and end_valid:
                    start_point = (int(start_kpt[0]), int(start_kpt[1]))
                    end_point = (int(end_kpt[0]), int(end_kpt[1]))
                    cv2.line(frame, start_point, end_point, pose_color, line_thickness)
        
        # Draw keypoints
        for i, kpt in enumerate(keypoints):
            if len(kpt) >= 3 and kpt[0] != -1 and kpt[1] != -1 and kpt[2] > 0:
                center = (int(kpt[0]), int(kpt[1]))
                # Draw keypoint center
                cv2.circle(frame, center, keypoint_radius, self.keypoint_center_color, -1)
                # Draw colored outline
                cv2.circle(frame, center, keypoint_radius + 1, pose_color, 1)
        
        return frame
    
    def draw_pose_info_labels(self, frame: np.ndarray, poses: List[Dict]) -> np.ndarray:
        """Draw pose information labels"""
        # Group poses by compound box
        pose_groups = {}
        for pose in poses:
            box_idx = pose.get('compound_box_index', -1)
            if box_idx not in pose_groups:
                pose_groups[box_idx] = []
            pose_groups[box_idx].append(pose)
        
        # Draw labels for each group
        for box_idx, group_poses in pose_groups.items():
            if not group_poses:
                continue
            
            # Get position from first pose
            first_pose = group_poses[0]
            if 'box' in first_pose:
                x1, y1, x2, y2 = first_pose['box'].astype(int)
                label_y = y2 + 20
            else:
                continue
            
            # Create label text
            method_counts = {}
            total_confidence = 0.0
            
            for pose in group_poses:
                method = pose.get('method', 'unknown')
                confidence = pose.get('confidence', 0.0)
                method_counts[method] = method_counts.get(method, 0) + 1
                total_confidence += confidence
            
            avg_confidence = total_confidence / len(group_poses)
            
            # Format label
            method_parts = []
            for method, count in method_counts.items():
                kp_count = "39kp" if method == 'SuperAnimal' else "17kp"
                method_parts.append(f"{method}:{count}({kp_count})")
            
            label_text = f"Box{box_idx}: {', '.join(method_parts)} conf:{avg_confidence:.2f}"
            
            # Draw label with background
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, label_y - text_height - 5), (x1 + text_width + 10, label_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, label_text, (x1 + 5, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        
        return frame
    
    def draw_info_overlay(self, frame: np.ndarray, frame_count: int, max_frames: int, 
                         detection_count: int, pose_stats: Dict, reid_info: Dict = None) -> np.ndarray:
        """Draw comprehensive info overlay"""
        total_display = str(max_frames) if max_frames != float('inf') else "∞"
        
        info_lines = [
            f"Frame: {frame_count+1}/{total_display}",
            f"System: Roboflow Racer Detection + Compound Pose Estimation",
            f"Racers Detected: {detection_count}",
        ]
        
        # Add pose statistics
        if pose_stats:
            info_lines.extend([
                f"Poses: Total:{pose_stats.get('total_poses', 0)} " +
                f"SuperAnimal:{pose_stats.get('superanimal_count', 0)}(39kp) " +
                f"ViTPose:{pose_stats.get('vitpose_count', 0)}(17kp)",
                f"Pose Confidence: {pose_stats.get('avg_confidence', 0.0):.3f}"
            ])
        
        # Add ReID info if available
        if reid_info:
            info_lines.extend([
                f"ReID: Active:{reid_info.get('active_tracks', 0)} " +
                f"Reassignments:{reid_info.get('total_reassignments', 0)}",
            ])
        
        # Add configuration info
        info_lines.extend([
            f"Config: {self.config.horse_pose_estimator} pose, {self.config.tracker_type} tracking",
            f"ReID: {'ON' if getattr(self.config, 'enable_reid', False) else 'OFF'}"
        ])
        
        if self.config.display:
            info_lines.append("Controls: SPACE=Pause Q=Quit")
        
        # Draw semi-transparent background
        overlay = frame.copy()
        overlay_height = 25 + len(info_lines) * 20
        cv2.rectangle(overlay, (5, 5), (min(1000, frame.shape[1]-10), overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text
        for i, line in enumerate(info_lines):
            y_pos = 25 + i * 20
            cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        
        return frame
    
    def get_vitpose_skeleton(self) -> List[tuple]:
        """Get ViTPose skeleton connections (17 keypoints)"""
        return [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]