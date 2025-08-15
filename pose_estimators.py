import numpy as np
import torch
import cv2
from typing import List, Dict

try:
    import supervision as sv
except ImportError:
    sv = None

try:
    from transformers import AutoProcessor, VitPoseForPoseEstimation
    VITPOSE_AVAILABLE = True
except ImportError:
    VITPOSE_AVAILABLE = False

class SimplifiedPoseEstimationManager:
    def __init__(self, config, superanimal_model=None):
        self.config = config
        self.superanimal = superanimal_model
        
        # Setup ViTPose models
        self.vitpose_processor = None
        self.vitpose_model = None
        
        self.setup_vitpose_model()
    
    def setup_vitpose_model(self):
        """Initialize ViTPose model if needed"""
        if self.config.horse_pose_estimator in ['vitpose', 'both']:
            if VITPOSE_AVAILABLE:
                try:
                    self.vitpose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
                    self.vitpose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")
                    self.vitpose_model.to(self.config.device)
                    print("✅ ViTPose loaded for compound racer detection")
                except Exception as e:
                    print(f"⚠️ ViTPose failed: {e}")
    
    def estimate_poses_on_racers(self, frame: np.ndarray, detections) -> List[Dict]:
        """Estimate poses on compound racer entities (horse+jockey in same box)"""
        if not sv or len(detections) == 0:
            return []
        
        if self.config.human_pose_estimator == 'vitpose':
            poses = self._estimate_vitpose_human(frame, detections)
            # Ensure one pose per box
            return self._ensure_one_pose_per_box(poses, len(detections), "Human ViTPose")
        else:
            return []
    
    def estimate_horse_poses(self, frame: np.ndarray, detections):
        """Horse pose estimation - ONE pose per detection box"""
        if self.config.horse_pose_estimator == 'none':
            return []
        
        if not sv or len(detections) == 0:
            return []
        
        if self.config.horse_pose_estimator == 'superanimal':
            poses = self._estimate_superanimal_only(frame, detections)
            return self._ensure_one_pose_per_box(poses, len(detections), "SuperAnimal")
        elif self.config.horse_pose_estimator == 'vitpose':
            poses = self._estimate_vitpose_horse_only(frame, detections)
            return self._ensure_one_pose_per_box(poses, len(detections), "Horse ViTPose")
        elif self.config.horse_pose_estimator == 'dual':
            return self._estimate_dual_competition(frame, detections)
        else:
            return []
    
    def _ensure_one_pose_per_box(self, poses, num_boxes, method_name):
        """
        Ensure exactly one pose per detection box.
        If multiple poses per box, select the best one.
        If no pose for a box, that's OK (empty slot).
        """
        if not poses:
            return []
        
        # Group poses by box index (assuming poses are in same order as detection boxes)
        final_poses = []
        
        for box_idx in range(num_boxes):
            # Get poses for this box (should be just one, but might be multiple or zero)
            poses_for_this_box = [p for i, p in enumerate(poses) if i == box_idx]
            
            if poses_for_this_box:
                # Select best pose for this box
                best_pose = self.select_best_pose_in_box(poses_for_this_box, method_name)
                if best_pose:
                    final_poses.append(best_pose)
            # If no pose for this box, we skip it (don't add empty placeholder)
        
        return all_poses
    
    def _estimate_superanimal_on_box(self, frame: np.ndarray, bbox: np.ndarray, box_index: int) -> List[Dict]:
        """Run SuperAnimal pose estimation on single racer box"""
        if not self.superanimal:
            return []
        
        try:
            # Create a single detection for SuperAnimal
            single_detection = sv.Detections(
                xyxy=np.array([bbox]),
                confidence=np.array([0.8]),
                class_id=np.array([0])
            )
            
            # Estimate pose using SuperAnimal
            poses = self.superanimal.estimate_pose(frame, single_detection)
            
            # Add metadata
            for pose in poses:
                pose['compound_box_index'] = box_index
                pose['pose_source'] = 'SuperAnimal'
                if 'method' not in pose:
                    pose['method'] = 'SuperAnimal'
            
            return poses
            
        except Exception as e:
            print(f"❌ SuperAnimal pose estimation failed on box {box_index}: {e}")
            return []
    
    def _estimate_vitpose_on_box(self, frame: np.ndarray, bbox: np.ndarray, box_index: int) -> List[Dict]:
        """Run ViTPose on single racer box"""
        if not self.vitpose_model or not self.vitpose_processor:
            return []
        
        try:
            from PIL import Image
            
            # Convert frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Convert bbox to COCO format
            x1, y1, x2, y2 = bbox
            coco_box = [x1, y1, x2-x1, y2-y1]
            
            # Process with ViTPose
            inputs = self.vitpose_processor(pil_image, boxes=[[coco_box]], return_tensors="pt").to(self.config.device)
            
            with torch.no_grad():
                outputs = self.vitpose_model(**inputs)
            
            pose_results = self.vitpose_processor.post_process_pose_estimation(outputs, boxes=[[coco_box]])
            
            converted_poses = []
            if pose_results and len(pose_results[0]) > 0:
                for pose_result in pose_results[0]:
                    keypoints = pose_result['keypoints'].cpu().numpy() if hasattr(pose_result['keypoints'], 'cpu') else pose_result['keypoints']
                    scores = pose_result['scores'].cpu().numpy() if hasattr(pose_result['scores'], 'cpu') else pose_result['scores']
                    
                    # Apply confidence filtering
                    conf_threshold = self.config.confidence_horse_pose_vitpose
                    filtered_keypoints = []
                    valid_count = 0
                    total_confidence = 0.0
                    
                    for kpt, score in zip(keypoints, scores):
                        if score > conf_threshold:
                            filtered_keypoints.append([kpt[0], kpt[1], score])
                            valid_count += 1
                            total_confidence += score
                        else:
                            filtered_keypoints.append([-1.0, -1.0, 0.0])
                    
                    avg_confidence = total_confidence / valid_count if valid_count > 0 else 0.0
                    
                    converted_pose = {
                        'keypoints': np.array(filtered_keypoints),
                        'box': bbox,
                        'method': 'ViTPose',
                        'confidence': avg_confidence,
                        'compound_box_index': box_index,
                        'pose_source': 'ViTPose',
                        'valid_keypoints': valid_count
                    }
                    
                    converted_poses.append(converted_pose)
            
            return converted_poses
            
        except Exception as e:
            print(f"❌ ViTPose estimation failed on box {box_index}: {e}")
            return []
    
    def filter_poses_by_confidence(self, poses: List[Dict]) -> List[Dict]:
        """Filter poses by confidence thresholds"""
        filtered_poses = []
        
        for pose in poses:
            method = pose.get('method', 'unknown')
            confidence = pose.get('confidence', 0.0)
            
            # Apply method-specific thresholds
            if method == 'SuperAnimal':
                threshold = self.config.confidence_horse_pose_superanimal
            elif method == 'ViTPose':
                threshold = self.config.confidence_horse_pose_vitpose
            else:
                threshold = 0.3  # Default
            
            if confidence >= threshold:
                filtered_poses.append(pose)
            else:
                print(f"🔽 Filtered out {method} pose (conf: {confidence:.3f} < {threshold})")
        
        return filtered_poses
    
    def group_poses_by_racer(self, poses: List[Dict]) -> Dict[int, List[Dict]]:
        """Group poses by compound racer box index"""
        grouped = {}
        
        for pose in poses:
            box_index = pose.get('compound_box_index', -1)
            if box_index not in grouped:
                grouped[box_index] = []
            grouped[box_index].append(pose)
        
        return grouped
    
    def get_pose_statistics(self, poses: List[Dict]) -> Dict:
        """Get statistics about pose estimation results"""
        if not poses:
            return {
                'total_poses': 0,
                'superanimal_count': 0,
                'vitpose_count': 0,
                'avg_confidence': 0.0,
                'racers_with_poses': 0
            }
        
        superanimal_count = len([p for p in poses if p.get('method') == 'SuperAnimal'])
        vitpose_count = len([p for p in poses if p.get('method') == 'ViTPose'])
        avg_confidence = np.mean([p.get('confidence', 0.0) for p in poses])
        unique_racers = len(set(p.get('compound_box_index', -1) for p in poses))
        
        return {
            'total_poses': len(poses),
            'superanimal_count': superanimal_count,
            'vitpose_count': vitpose_count,
            'avg_confidence': float(avg_confidence),
            'racers_with_poses': unique_racers
        }