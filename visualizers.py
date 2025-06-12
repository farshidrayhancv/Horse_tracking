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
        
        # Setup colors
        self.human_color = (0, 255, 0)      # Green for jockeys
        self.superanimal_color = (255, 0, 0)  # Red for SuperAnimal horses
        self.vitpose_horse_color = (255, 0, 255)  # Magenta for ViTPose horses
        self.keypoint_color = (0, 255, 255) # Cyan
        self.text_color = (255, 255, 255)   # White
        
        # Define skeletons
        self.human_skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
    
    def draw_human_pose(self, frame: np.ndarray, pose_result: Dict, min_confidence: float = None):
        """
        🔥 SIMPLIFIED: Just draws keypoints - confidence filtering already done at source
        """
        if 'keypoints' not in pose_result or 'scores' not in pose_result:
            return frame
        
        keypoints = pose_result['keypoints'].cpu().numpy() if hasattr(pose_result['keypoints'], 'cpu') else pose_result['keypoints']
        scores = pose_result['scores'].cpu().numpy() if hasattr(pose_result['scores'], 'cpu') else pose_result['scores']
        
        # Draw skeleton - only valid keypoints (confidence > 0)
        for start_idx, end_idx in self.human_skeleton:
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                scores[start_idx] > 0 and scores[end_idx] > 0):
                start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                cv2.line(frame, start_point, end_point, self.human_color, 2)
        
        # Draw keypoints - only valid ones (confidence > 0)
        valid_count = 0
        for i, (kpt, score) in enumerate(zip(keypoints, scores)):
            if score > 0:  # Valid keypoint (already filtered at source)
                center = (int(kpt[0]), int(kpt[1]))
                cv2.circle(frame, center, 4, self.keypoint_color, -1)
                cv2.circle(frame, center, 5, self.human_color, 1)
                valid_count += 1
        
        # print(f"✅ Human pose: Drew {valid_count}/{len(keypoints)} valid keypoints")
        return frame
    
    def draw_horse_pose(self, frame: np.ndarray, pose_result: Dict, min_confidence: float = None):
        """
        🔥 SIMPLIFIED: Just draws keypoints - confidence filtering already done at source
        """
        if 'keypoints' not in pose_result or 'method' not in pose_result:
            return frame
        
        keypoints = pose_result['keypoints']
        method = pose_result['method']
        
        # Choose color and skeleton based on method
        if method == 'SuperAnimal':
            color = self.superanimal_color
            skeleton = self.superanimal.skeleton if self.superanimal else []
        else:  # ViTPose
            color = self.vitpose_horse_color
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
        
        # print(f"✅ {method} pose: Drew {valid_count}/{len(keypoints)} valid keypoints")
        return frame
    
    def annotate_detections(self, frame: np.ndarray, human_detections, horse_detections):
        if not sv:
            return frame
        
        try:
            # Human detections (green triangles)
            if len(human_detections) > 0:
                triangle_annotator = sv.TriangleAnnotator(
                    base=15, height=20, color=sv.Color.GREEN
                )
                frame = triangle_annotator.annotate(frame, human_detections)
            
            # Horse detections (red triangles)
            if len(horse_detections) > 0:
                triangle_annotator = sv.TriangleAnnotator(
                    base=15, height=20, color=sv.Color.RED
                )
                frame = triangle_annotator.annotate(frame, horse_detections)
                
        except Exception:
            # Fallback to rectangles
            if len(human_detections) > 0:
                for i, box in enumerate(human_detections.xyxy):
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), self.human_color, 2)
                    cv2.putText(frame, f"Human {i+1}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.human_color, 2)
            
            if len(horse_detections) > 0:
                for i, box in enumerate(horse_detections.xyxy):
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), self.superanimal_color, 2)
                    cv2.putText(frame, f"Horse {i+1}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.superanimal_color, 2)
        
        return frame
    
    def draw_pose_labels(self, frame: np.ndarray, horse_poses: list):
        for pose_result in horse_poses:
            if 'box' in pose_result and 'method' in pose_result:
                box = pose_result['box']
                method = pose_result['method']
                conf = pose_result.get('confidence', 0)
                x1, y1, x2, y2 = box.astype(int)
                
                color = self.superanimal_color if method == 'SuperAnimal' else self.vitpose_horse_color
                kp_count = "39kp" if method == 'SuperAnimal' else "17kp"
                
                cv2.putText(frame, f"{method} {conf:.2f} ({kp_count})", 
                           (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def draw_info_overlay(self, frame: np.ndarray, frame_count: int, max_frames: int, 
                         human_count: int, horse_count: int, human_poses: int, 
                         horse_poses: int, stats: dict = None):
        total_display = str(max_frames) if max_frames != float('inf') else "∞"
        
        info_lines = [
            f"Frame: {frame_count+1}/{total_display}",
            f"Config: H-Det:{self.config.human_detector} H-Pose:{self.config.human_pose_estimator}",
            f"        Horse-Det:{self.config.horse_detector} Horse-Pose:{self.config.horse_pose_estimator}",
            f"Detected - Jockeys:{human_count} Horses:{horse_count}",
            f"Poses - Humans:{human_poses} Horses:{horse_poses}",
            f"SOURCE Filtering: Human:{self.config.confidence_human_pose} Horse-ViTPose:{self.config.confidence_horse_pose_vitpose}"
        ]
        
        if self.config.horse_pose_estimator == 'dual' and stats:
            info_lines.append(f"Competition - SuperAnimal:{stats.get('superanimal_wins', 0)} ViTPose:{stats.get('vitpose_wins', 0)}")
        
        if self.config.display:
            info_lines.append(f"Controls: SPACE=Pause Q=Quit")
        
        # Semi-transparent background
        overlay = frame.copy()
        overlay_height = 25 + len(info_lines) * 18
        cv2.rectangle(overlay, (5, 5), (950, overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        for i, line in enumerate(info_lines):
            y_pos = 25 + i * 18
            cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        
        return frame