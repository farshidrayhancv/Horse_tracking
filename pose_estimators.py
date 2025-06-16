import numpy as np
import torch
import cv2

try:
    import supervision as sv
except ImportError:
    sv = None

try:
    from transformers import AutoProcessor, VitPoseForPoseEstimation
    VITPOSE_AVAILABLE = True
except ImportError:
    VITPOSE_AVAILABLE = False

class PoseEstimationManager:
    def __init__(self, config, superanimal_model=None):
        self.config = config
        self.superanimal = superanimal_model
        
        # Setup ViTPose models
        self.vitpose_processor = None
        self.vitpose_model = None
        self.vitpose_horse_processor = None
        self.vitpose_horse_model = None
        
        self.setup_vitpose_models()
    
    def setup_vitpose_models(self):
        # ViTPose for humans
        if self.config.human_pose_estimator == 'vitpose':
            if VITPOSE_AVAILABLE:
                try:
                    self.vitpose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
                    self.vitpose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")
                    self.vitpose_model.to(self.config.device)
                    print("✅ ViTPose (humans) loaded")
                except Exception as e:
                    print(f"⚠️ ViTPose failed: {e}")
        
        # ViTPose for horses (if needed)
        if self.config.horse_pose_estimator in ['vitpose', 'dual']:
            if VITPOSE_AVAILABLE:
                try:
                    self.vitpose_horse_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
                    self.vitpose_horse_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")
                    self.vitpose_horse_model.to(self.config.device)
                    print("✅ ViTPose (for horses) loaded")
                except Exception as e:
                    print(f"⚠️ ViTPose for horses failed: {e}")
    
    def select_best_pose_in_box(self, poses_in_box, method_name="Unknown"):
        """
        If multiple pose candidates are found within a single detection box,
        select the one with highest confidence (main subject).
        """
        if not poses_in_box or len(poses_in_box) == 0:
            return None
        
        if len(poses_in_box) == 1:
            return poses_in_box[0]
        
        # Multiple poses detected in same box - pick best one
        best_pose = None
        best_confidence = -1
        
        for pose in poses_in_box:
            if 'confidence' in pose and pose['confidence'] > best_confidence:
                best_confidence = pose['confidence']
                best_pose = pose
        
        if best_pose is None:
            best_pose = poses_in_box[0]  # Fallback to first
        
        print(f"🎯 {method_name}: Selected best pose (conf: {best_confidence:.3f}) from {len(poses_in_box)} candidates in same box")
        return best_pose
    
    def estimate_human_poses(self, frame: np.ndarray, detections):
        """Human pose estimation - ONE pose per detection box"""
        if self.config.human_pose_estimator == 'none':
            return []
        
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
            poses_for_this_box = [p for p in poses if p.get('box_index') == box_idx]
            
            if poses_for_this_box:
                # Select best pose for this box
                best_pose = self.select_best_pose_in_box(poses_for_this_box, method_name)
                if best_pose:
                    final_poses.append(best_pose)
            # If no pose for this box, we skip it (don't add empty placeholder)
        
        return final_poses
    
    def _estimate_vitpose_human(self, frame: np.ndarray, detections):
        """ViTPose for humans - processes each detection box independently"""
        if not self.vitpose_model or not sv or len(detections) == 0:
            return []
        
        try:
            from PIL import Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Process each detection box independently to ensure one pose per box
            all_poses = []
            
            for box_idx, box in enumerate(detections.xyxy):
                # Convert single box to COCO format
                x1, y1, x2, y2 = box
                coco_box = [x1, y1, x2-x1, y2-y1]  # [x, y, width, height]
                
                inputs = self.vitpose_processor(pil_image, boxes=[[coco_box]], return_tensors="pt").to(self.config.device)
                
                with torch.no_grad():
                    outputs = self.vitpose_model(**inputs)
                
                pose_results = self.vitpose_processor.post_process_pose_estimation(outputs, boxes=[[coco_box]])
                
                # Process result for this specific box
                if pose_results and len(pose_results[0]) > 0:
                    # Take only the first (best) pose for this box
                    pose_result = pose_results[0][0]
                    
                    keypoints = pose_result['keypoints'].cpu().numpy() if hasattr(pose_result['keypoints'], 'cpu') else pose_result['keypoints']
                    scores = pose_result['scores'].cpu().numpy() if hasattr(pose_result['scores'], 'cpu') else pose_result['scores']
                    
                    # Apply confidence filtering
                    conf_threshold = self.config.confidence_human_pose
                    filtered_keypoints = []
                    filtered_scores = []
                    valid_count = 0
                    total_confidence = 0.0
                    
                    for kpt, score in zip(keypoints, scores):
                        if score > conf_threshold:
                            filtered_keypoints.append(kpt)
                            filtered_scores.append(score)
                            valid_count += 1
                            total_confidence += score
                        else:
                            filtered_keypoints.append([-1.0, -1.0])
                            filtered_scores.append(0.0)
                    
                    avg_confidence = total_confidence / valid_count if valid_count > 0 else 0.0
                    
                    # Create pose result with confidence
                    pose_with_confidence = {
                        'keypoints': torch.tensor(filtered_keypoints) if hasattr(pose_result['keypoints'], 'cpu') else np.array(filtered_keypoints),
                        'scores': torch.tensor(filtered_scores) if hasattr(pose_result['scores'], 'cpu') else np.array(filtered_scores),
                        'confidence': avg_confidence,
                        'box_index': box_idx
                    }
                    
                    all_poses.append(pose_with_confidence)
            
            return all_poses
            
        except Exception as e:
            print(f"Error in human pose estimation: {e}")
            return []
    
    def _estimate_superanimal_only(self, frame: np.ndarray, detections):
        """SuperAnimal pose estimation - already processes each box independently"""
        if not self.superanimal:
            return []
        
        if not sv or len(detections) == 0:
            return []
        
        # SuperAnimal.estimate_pose already processes each box independently
        # and applies confidence filtering at source
        poses = self.superanimal.estimate_pose(frame, detections)
        
        # Add box index for tracking
        for i, pose in enumerate(poses):
            pose['box_index'] = i
        
        return poses
    
    def _estimate_vitpose_horse_only(self, frame: np.ndarray, detections):
        """ViTPose for horses - processes each detection box independently"""
        if not self.vitpose_horse_model or not sv or len(detections) == 0:
            return []
        
        try:
            from PIL import Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Process each detection box independently
            all_poses = []
            
            for box_idx, box in enumerate(detections.xyxy):
                # Convert single box to COCO format
                x1, y1, x2, y2 = box
                coco_box = [x1, y1, x2-x1, y2-y1]
                
                inputs = self.vitpose_horse_processor(pil_image, boxes=[[coco_box]], return_tensors="pt").to(self.config.device)
                
                with torch.no_grad():
                    outputs = self.vitpose_horse_model(**inputs)
                
                pose_results = self.vitpose_horse_processor.post_process_pose_estimation(outputs, boxes=[[coco_box]])
                
                if pose_results and len(pose_results[0]) > 0:
                    # Take only the first (best) pose for this box
                    pose_result = pose_results[0][0]
                    
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
                        'box': box,
                        'method': 'ViTPose',
                        'confidence': avg_confidence,
                        'box_index': box_idx
                    }
                    
                    all_poses.append(converted_pose)
            
            return all_poses
            
        except Exception as e:
            print(f"Error in ViTPose horse estimation: {e}")
            return []
    
    def _estimate_dual_competition(self, frame: np.ndarray, detections):
        """Dual competition mode - ONE pose per box, best method wins"""
        if not sv or len(detections) == 0:
            return []
        
        # Get poses from both methods - each processes boxes independently
        superanimal_poses = self._estimate_superanimal_only(frame, detections)
        vitpose_poses = self._estimate_vitpose_horse_only(frame, detections)
        
        # Competition: Pick best pose per detection box
        best_poses = []
        num_boxes = len(detections)
        
        for box_idx in range(num_boxes):
            superanimal_conf = 0.0
            vitpose_conf = 0.0
            superanimal_pose = None
            vitpose_pose = None
            
            # Find SuperAnimal pose for this box
            for pose in superanimal_poses:
                if pose.get('box_index') == box_idx:
                    superanimal_conf = pose['confidence']
                    superanimal_pose = pose
                    break
            
            # Find ViTPose pose for this box
            for pose in vitpose_poses:
                if pose.get('box_index') == box_idx:
                    vitpose_conf = pose['confidence']
                    vitpose_pose = pose
                    break
            
            # Pick the winner for this specific box
            if superanimal_conf > vitpose_conf and superanimal_pose:
                best_poses.append(superanimal_pose)
                print(f"🏆 Box {box_idx+1}: SuperAnimal wins (conf: {superanimal_conf:.3f} vs {vitpose_conf:.3f})")
            elif vitpose_conf > 0 and vitpose_pose:
                best_poses.append(vitpose_pose)
                print(f"🏆 Box {box_idx+1}: ViTPose wins (conf: {vitpose_conf:.3f} vs {superanimal_conf:.3f})")
            # If neither has confidence > 0, no pose for this box
        
        return best_poses