import cv2
import numpy as np
from typing import Dict, List, Tuple
import math

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
        
        # 3D pose visualization colors
        self.pose_2d_color = (0, 255, 255)  # Cyan for 2D keypoints
        self.pose_3d_color = (255, 0, 255)  # Magenta for 3D keypoints
        self.depth_gradient_colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]  # Blue to Red
        
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
        
        # 3D visualization parameters
        self.depth_scale_factor = getattr(config, 'depth_visualization_scale', 2.0)
        self.pose_3d_alpha = getattr(config, 'pose_3d_transparency', 0.7)
        self.enable_depth_color_coding = getattr(config, 'enable_depth_color_coding', True)
        self.show_3d_pose_quality = getattr(config, 'show_3d_pose_quality', True)
        
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
    
    def get_depth_color(self, depth_value: float, min_depth: float, max_depth: float) -> Tuple[int, int, int]:
        """Get color based on depth value for 3D visualization"""
        if max_depth <= min_depth:
            return self.depth_gradient_colors[2]  # Default middle color
        
        # Normalize depth to 0-1 range
        normalized_depth = (depth_value - min_depth) / (max_depth - min_depth)
        normalized_depth = max(0, min(1, normalized_depth))
        
        # Map to color gradient (0 = blue/close, 1 = red/far)
        color_index = normalized_depth * (len(self.depth_gradient_colors) - 1)
        idx = int(color_index)
        frac = color_index - idx
        
        if idx >= len(self.depth_gradient_colors) - 1:
            return self.depth_gradient_colors[-1]
        
        # Interpolate between adjacent colors
        color1 = np.array(self.depth_gradient_colors[idx])
        color2 = np.array(self.depth_gradient_colors[idx + 1])
        interpolated = color1 * (1 - frac) + color2 * frac
        
        return tuple(map(int, interpolated))
    
    def draw_3d_pose_keypoints(self, frame: np.ndarray, pose_3d: Dict, track_color: Tuple[int, int, int]) -> np.ndarray:
        """Draw 3D pose keypoints with depth-based visualization"""
        if 'keypoints_3d' not in pose_3d:
            return frame
        
        keypoints_3d = pose_3d['keypoints_3d']
        if len(keypoints_3d) == 0:
            return frame
        
        # Extract valid keypoints (confidence > 0)
        valid_keypoints = keypoints_3d[keypoints_3d[:, 3] > 0]
        if len(valid_keypoints) == 0:
            return frame
        
        # Get depth range for color coding
        depths = valid_keypoints[:, 2]
        min_depth, max_depth = np.min(depths), np.max(depths)
        
        # Draw keypoints with depth-based coloring
        for i, (x, y, depth, confidence) in enumerate(keypoints_3d):
            if confidence <= 0:
                continue
            
            center = (int(x), int(y))
            
            # Choose color based on depth if enabled
            if self.enable_depth_color_coding and max_depth > min_depth:
                depth_color = self.get_depth_color(depth, min_depth, max_depth)
            else:
                depth_color = self.pose_3d_color
            
            # Draw keypoint with size proportional to confidence
            radius = max(3, int(6 * confidence))
            cv2.circle(frame, center, radius, depth_color, -1)
            cv2.circle(frame, center, radius + 1, track_color, 2)
            
            # Draw depth value as small text if quality display is enabled
            if self.show_3d_pose_quality:
                cv2.putText(frame, f"{depth:.0f}", (center[0] + 8, center[1] - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, depth_color, 1)
        
        return frame
    
    def draw_3d_pose_skeleton(self, frame: np.ndarray, pose_3d: Dict, skeleton: List[Tuple[int, int]], 
                             track_color: Tuple[int, int, int]) -> np.ndarray:
        """Draw 3D pose skeleton with depth-aware line thickness"""
        if 'keypoints_3d' not in pose_3d:
            return frame
        
        keypoints_3d = pose_3d['keypoints_3d']
        
        # Get depth range for line thickness calculation
        valid_depths = [kpt[2] for kpt in keypoints_3d if kpt[3] > 0]
        if len(valid_depths) < 2:
            return frame
        
        min_depth, max_depth = min(valid_depths), max(valid_depths)
        
        # Draw skeleton connections
        for start_idx, end_idx in skeleton:
            if (start_idx < len(keypoints_3d) and end_idx < len(keypoints_3d)):
                start_kpt = keypoints_3d[start_idx]
                end_kpt = keypoints_3d[end_idx]
                
                # Check if both keypoints are valid
                if start_kpt[3] > 0 and end_kpt[3] > 0:
                    start_point = (int(start_kpt[0]), int(start_kpt[1]))
                    end_point = (int(end_kpt[0]), int(end_kpt[1]))
                    
                    # Calculate line thickness based on average depth
                    avg_depth = (start_kpt[2] + end_kpt[2]) / 2
                    if max_depth > min_depth:
                        depth_ratio = (avg_depth - min_depth) / (max_depth - min_depth)
                        thickness = max(1, int(4 * (1 - depth_ratio)))  # Closer = thicker
                    else:
                        thickness = 2
                    
                    # Get line color
                    if self.enable_depth_color_coding:
                        line_color = self.get_depth_color(avg_depth, min_depth, max_depth)
                    else:
                        line_color = track_color
                    
                    cv2.line(frame, start_point, end_point, line_color, thickness)
        
        return frame
    
    def draw_3d_pose_quality_info(self, frame: np.ndarray, pose_3d: Dict, bbox_top_left: Tuple[int, int]) -> np.ndarray:
        """Draw 3D pose quality information"""
        if not self.show_3d_pose_quality or 'confidence_3d' not in pose_3d:
            return frame
        
        x, y = bbox_top_left
        confidence_3d = pose_3d['confidence_3d']
        valid_keypoints_3d = pose_3d.get('valid_keypoints_3d', 0)
        
        # Quality text with background
        quality_text = f"3D: {confidence_3d:.2f} ({valid_keypoints_3d}kp)"
        text_size = cv2.getTextSize(quality_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y - 25), (x + text_size[0] + 10, y - 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Quality color based on confidence
        if confidence_3d > 0.7:
            quality_color = (0, 255, 0)  # Green - high quality
        elif confidence_3d > 0.4:
            quality_color = (0, 255, 255)  # Yellow - medium quality
        else:
            quality_color = (0, 0, 255)  # Red - low quality
        
        cv2.putText(frame, quality_text, (x + 5, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, quality_color, 1)
        
        return frame
    
    def draw_human_pose_3d(self, frame: np.ndarray, pose_result: Dict, min_confidence: float = None):
        """Draw 3D human pose with enhanced depth visualization"""
        if 'keypoints_3d' not in pose_result:
            # Fallback to 2D pose
            return self.draw_human_pose(frame, pose_result, min_confidence)
        
        # Get unique color based on track ID
        track_id = pose_result.get('track_id', -1)
        track_color = self.get_track_color(track_id)
        
        # Draw 3D skeleton
        frame = self.draw_3d_pose_skeleton(frame, pose_result, self.human_skeleton, track_color)
        
        # Draw 3D keypoints
        frame = self.draw_3d_pose_keypoints(frame, pose_result, track_color)
        
        # Draw quality information
        if 'bbox' in pose_result:
            bbox = pose_result['bbox']
            bbox_top_left = (int(bbox[0]), int(bbox[1]))
            frame = self.draw_3d_pose_quality_info(frame, pose_result, bbox_top_left)
        
        return frame
    
    def draw_horse_pose_3d(self, frame: np.ndarray, pose_result: Dict, min_confidence: float = None):
        """Draw 3D horse pose with enhanced depth visualization"""
        if 'keypoints_3d' not in pose_result:
            # Fallback to 2D pose
            return self.draw_horse_pose(frame, pose_result, min_confidence)
        
        # Get unique color based on track ID
        track_id = pose_result.get('track_id', -1)
        track_color = self.get_track_color(track_id)
        
        # Choose skeleton based on method
        method = pose_result.get('method', 'unknown')
        if method == 'SuperAnimal':
            skeleton = self.superanimal.skeleton if self.superanimal else []
        else:  # ViTPose
            skeleton = self.human_skeleton
        
        # Draw 3D skeleton
        frame = self.draw_3d_pose_skeleton(frame, pose_result, skeleton, track_color)
        
        # Draw 3D keypoints
        frame = self.draw_3d_pose_keypoints(frame, pose_result, track_color)
        
        # Draw quality information
        if 'bbox' in pose_result:
            bbox = pose_result['bbox']
            bbox_top_left = (int(bbox[0]), int(bbox[1]))
            frame = self.draw_3d_pose_quality_info(frame, pose_result, bbox_top_left)
        
        return frame
    
    def draw_human_pose(self, frame: np.ndarray, pose_result: Dict, min_confidence: float = None):
        """Draw 2D human pose (fallback or when 3D not available)"""
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
        for i, (kpt, score) in enumerate(zip(keypoints, scores)):
            if score > 0:  # Valid keypoint
                center = (int(kpt[0]), int(kpt[1]))
                cv2.circle(frame, center, 4, self.keypoint_color, -1)
                cv2.circle(frame, center, 5, color, 1)
        
        return frame
    
    def draw_horse_pose(self, frame: np.ndarray, pose_result: Dict, min_confidence: float = None):
        """Draw 2D horse pose (fallback or when 3D not available)"""
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
            skeleton = self.human_skeleton
        
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
        for i, kpt in enumerate(keypoints):
            x, y, conf = kpt
            if x != -1 and y != -1 and conf > 0:  # Valid keypoint
                center = (int(x), int(y))
                cv2.circle(frame, center, 4, self.keypoint_color, -1)
                cv2.circle(frame, center, 5, color, 1)
        
        return frame
    
    def draw_human_pose_with_tracking(self, frame: np.ndarray, pose_result: Dict, min_confidence: float = None):
        """Draw human pose with track ID using track-based colors (3D if available)"""
        if 'keypoints_3d' in pose_result:
            frame = self.draw_human_pose_3d(frame, pose_result, min_confidence)
        else:
            frame = self.draw_human_pose(frame, pose_result, min_confidence)
        
        # Add track ID if available
        if 'track_id' in pose_result:
            track_id = pose_result['track_id']
            if track_id >= 0:
                # Find head position for label
                head_pos = self._find_head_position(pose_result)
                if head_pos:
                    pose_type = "3D" if 'keypoints_3d' in pose_result else "2D"
                    cv2.putText(frame, f"J{track_id}({pose_type})", head_pos, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.track_color, 2)
        
        return frame
    
    def draw_horse_pose_with_tracking(self, frame: np.ndarray, pose_result: Dict, min_confidence: float = None):
        """Draw horse pose with track ID using track-based colors (3D if available)"""
        if 'keypoints_3d' in pose_result:
            frame = self.draw_horse_pose_3d(frame, pose_result, min_confidence)
        else:
            frame = self.draw_horse_pose(frame, pose_result, min_confidence)
        
        # Add track ID if available
        if 'track_id' in pose_result:
            track_id = pose_result['track_id']
            if track_id >= 0:
                # Find head position for label
                head_pos = self._find_head_position(pose_result)
                if head_pos:
                    method = pose_result.get('method', 'Horse')
                    pose_type = "3D" if 'keypoints_3d' in pose_result else "2D"
                    prefix = "SA" if method == 'SuperAnimal' else "VP"
                    cv2.putText(frame, f"{prefix}{track_id}({pose_type})", head_pos, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.track_color, 2)
        
        return frame
    
    def _find_head_position(self, pose_result: Dict) -> Tuple[int, int]:
        """Find head position for label placement"""
        head_pos = None
        
        # Try 3D keypoints first
        if 'keypoints_3d' in pose_result:
            keypoints = pose_result['keypoints_3d']
            for i in [0, 1, 2]:  # nose, upper_jaw, lower_jaw (or equivalent)
                if i < len(keypoints) and keypoints[i][3] > 0:
                    head_pos = (int(keypoints[i][0]), int(keypoints[i][1]) - 25)
                    break
        
        # Fallback to 2D keypoints
        if head_pos is None and 'keypoints' in pose_result:
            keypoints = pose_result['keypoints']
            if hasattr(keypoints, 'cpu'):
                keypoints = keypoints.cpu().numpy()
            
            # For human poses
            if 'scores' in pose_result:
                scores = pose_result['scores']
                if hasattr(scores, 'cpu'):
                    scores = scores.cpu().numpy()
                for i in [0, 1, 2]:  # nose, left_eye, right_eye
                    if i < len(keypoints) and scores[i] > 0:
                        head_pos = (int(keypoints[i][0]), int(keypoints[i][1]) - 20)
                        break
            
            # For horse poses (SuperAnimal format)
            else:
                for i in [0, 1, 2]:  # nose, upper_jaw, lower_jaw
                    if (i < len(keypoints) and len(keypoints[i]) >= 3 and 
                        keypoints[i][0] != -1 and keypoints[i][2] > 0):
                        head_pos = (int(keypoints[i][0]), int(keypoints[i][1]) - 25)
                        break
        
        return head_pos
    
    def create_3d_pose_depth_visualization(self, poses_3d: List[Dict], frame_shape: Tuple[int, int]) -> np.ndarray:
        """Create a specialized visualization showing 3D pose depth distribution"""
        h, w = frame_shape[:2]
        depth_viz = np.zeros((h, w, 3), dtype=np.uint8)
        
        if not poses_3d:
            return depth_viz
        
        # Collect all depth values for normalization
        all_depths = []
        for pose in poses_3d:
            if pose and 'keypoints_3d' in pose:
                keypoints_3d = pose['keypoints_3d']
                valid_depths = [kpt[2] for kpt in keypoints_3d if kpt[3] > 0]
                all_depths.extend(valid_depths)
        
        if not all_depths:
            return depth_viz
        
        min_depth, max_depth = min(all_depths), max(all_depths)
        
        # Draw each pose with depth-based visualization
        for pose in poses_3d:
            if not pose or 'keypoints_3d' not in pose:
                continue
            
            keypoints_3d = pose['keypoints_3d']
            track_id = pose.get('track_id', -1)
            
            # Draw keypoints with depth-based size and color
            for x, y, depth, confidence in keypoints_3d:
                if confidence <= 0:
                    continue
                
                center = (int(x), int(y))
                depth_color = self.get_depth_color(depth, min_depth, max_depth)
                
                # Size based on depth (closer = larger)
                if max_depth > min_depth:
                    depth_ratio = 1 - (depth - min_depth) / (max_depth - min_depth)
                    radius = max(2, int(8 * depth_ratio))
                else:
                    radius = 4
                
                cv2.circle(depth_viz, center, radius, depth_color, -1)
                
                # Add depth value text
                cv2.putText(depth_viz, f"{depth:.0f}", (center[0] + 10, center[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add depth scale legend
        self._draw_depth_scale_legend(depth_viz, min_depth, max_depth)
        
        return depth_viz
    
    def _draw_depth_scale_legend(self, frame: np.ndarray, min_depth: float, max_depth: float):
        """Draw depth scale legend on visualization"""
        legend_x, legend_y = 20, frame.shape[0] - 100
        legend_width, legend_height = 200, 20
        
        # Draw gradient bar
        for i in range(legend_width):
            normalized_pos = i / legend_width
            color = self.get_depth_color(min_depth + normalized_pos * (max_depth - min_depth), 
                                       min_depth, max_depth)
            cv2.line(frame, (legend_x + i, legend_y), (legend_x + i, legend_y + legend_height), color, 1)
        
        # Add labels
        cv2.putText(frame, f"Close: {min_depth:.0f}", (legend_x, legend_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Far: {max_depth:.0f}", (legend_x + legend_width - 50, legend_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
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
                        
                        # Check if this was re-identified
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
        """Draw pose method labels with track-based colors and 3D information"""
        for pose_result in horse_poses:
            if 'box' in pose_result and 'method' in pose_result:
                box = pose_result['box']
                method = pose_result['method']
                conf = pose_result.get('confidence', 0)
                track_id = pose_result.get('track_id', -1)
                x1, y1, x2, y2 = box.astype(int)
                
                # Use track-based color
                color = self.get_track_color(track_id)
                
                # Determine keypoint info
                if 'keypoints_3d' in pose_result:
                    kp_count = f"{pose_result.get('valid_keypoints_3d', 0)}/3D"
                    conf_3d = pose_result.get('confidence_3d', 0)
                    pose_info = f"{method} 2D:{conf:.2f} 3D:{conf_3d:.2f} ({kp_count})"
                else:
                    kp_count = "39kp" if method == 'SuperAnimal' else "17kp"
                    pose_info = f"{method} {conf:.2f} ({kp_count})"
                
                track_str = f" T{track_id}" if track_id >= 0 else ""
                cv2.putText(frame, f"{pose_info}{track_str}", 
                           (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame