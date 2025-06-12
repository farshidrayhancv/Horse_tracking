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
    
    def estimate_human_poses(self, frame: np.ndarray, detections):
        """Human pose estimation ONLY runs inside detected human/jockey bounding boxes."""
        if self.config.human_pose_estimator == 'none':
            return []
        
        if not sv or len(detections) == 0:
            return []
        
        if self.config.human_pose_estimator == 'vitpose':
            return self._estimate_vitpose_human(frame, detections)
        else:
            return []
    
    def estimate_horse_poses(self, frame: np.ndarray, detections):
        """Horse pose estimation ONLY runs inside detected horse bounding boxes."""
        if self.config.horse_pose_estimator == 'none':
            return []
        
        if not sv or len(detections) == 0:
            return []
        
        if self.config.horse_pose_estimator == 'superanimal':
            return self._estimate_superanimal_only(frame, detections)
        elif self.config.horse_pose_estimator == 'vitpose':
            return self._estimate_vitpose_horse_only(frame, detections)
        elif self.config.horse_pose_estimator == 'dual':
            return self._estimate_dual_competition(frame, detections)
        else:
            return []
    
    def _estimate_vitpose_human(self, frame: np.ndarray, detections):
        """ViTPose for humans - ONLY processes detected human/jockey bounding boxes"""
        if not self.vitpose_model or not sv or len(detections) == 0:
            return []
        
        try:
            from PIL import Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Convert supervision detections to COCO format bounding boxes
            coco_boxes = detections.xyxy.copy()
            coco_boxes[:, 2] = coco_boxes[:, 2] - coco_boxes[:, 0]  # width
            coco_boxes[:, 3] = coco_boxes[:, 3] - coco_boxes[:, 1]  # height
            
            inputs = self.vitpose_processor(pil_image, boxes=[coco_boxes], return_tensors="pt").to(self.config.device)
            
            with torch.no_grad():
                outputs = self.vitpose_model(**inputs)
            
            pose_results = self.vitpose_processor.post_process_pose_estimation(outputs, boxes=[coco_boxes])
            
            # 🔥 CRITICAL: Apply confidence filtering at SOURCE
            if pose_results and len(pose_results[0]) > 0:
                filtered_poses = []
                conf_threshold = self.config.confidence_human_pose
                
                for pose_result in pose_results[0]:
                    keypoints = pose_result['keypoints'].cpu().numpy() if hasattr(pose_result['keypoints'], 'cpu') else pose_result['keypoints']
                    scores = pose_result['scores'].cpu().numpy() if hasattr(pose_result['scores'], 'cpu') else pose_result['scores']
                    
                    # Filter keypoints at source: set invalid ones to (-1, -1, 0.0)
                    filtered_keypoints = []
                    filtered_scores = []
                    valid_count = 0
                    
                    for kpt, score in zip(keypoints, scores):
                        if score > conf_threshold:
                            filtered_keypoints.append(kpt)
                            filtered_scores.append(score)
                            valid_count += 1
                        else:
                            filtered_keypoints.append([-1.0, -1.0])  # Invalid keypoint marker
                            filtered_scores.append(0.0)
                    
                    # print(f"🔥 Human ViTPose SOURCE filtering: {valid_count}/{len(keypoints)} keypoints above {conf_threshold}")
                    
                    # Update the pose result with filtered data
                    pose_result['keypoints'] = torch.tensor(filtered_keypoints) if hasattr(pose_result['keypoints'], 'cpu') else np.array(filtered_keypoints)
                    pose_result['scores'] = torch.tensor(filtered_scores) if hasattr(pose_result['scores'], 'cpu') else np.array(filtered_scores)
                    
                    filtered_poses.append(pose_result)
                
                return filtered_poses
            
            return pose_results[0] if pose_results else []
            
        except Exception as e:
            print(f"Error in human pose estimation: {e}")
            return []
    
    def _estimate_superanimal_only(self, frame: np.ndarray, detections):
        """SuperAnimal pose estimation - ONLY processes detected horse bounding boxes"""
        if not self.superanimal:
            return []
        
        if not sv or len(detections) == 0:
            return []
        
        # SuperAnimal already handles confidence filtering in estimate_pose()
        return self.superanimal.estimate_pose(frame, detections)
    
    def _estimate_vitpose_horse_only(self, frame: np.ndarray, detections):
        """ViTPose for horses - ONLY processes detected horse bounding boxes"""
        if not self.vitpose_horse_model or not sv or len(detections) == 0:
            return []
        
        try:
            from PIL import Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Convert supervision detections to COCO format bounding boxes
            coco_boxes = detections.xyxy.copy()
            coco_boxes[:, 2] = coco_boxes[:, 2] - coco_boxes[:, 0]  # width
            coco_boxes[:, 3] = coco_boxes[:, 3] - coco_boxes[:, 1]  # height
            
            inputs = self.vitpose_horse_processor(pil_image, boxes=[coco_boxes], return_tensors="pt").to(self.config.device)
            
            with torch.no_grad():
                outputs = self.vitpose_horse_model(**inputs)
            
            pose_results = self.vitpose_horse_processor.post_process_pose_estimation(outputs, boxes=[coco_boxes])
            
            if pose_results and len(pose_results[0]) > 0:
                converted_poses = []
                conf_threshold = self.config.confidence_horse_pose_vitpose
                
                for i, pose_result in enumerate(pose_results[0]):
                    keypoints = pose_result['keypoints'].cpu().numpy() if hasattr(pose_result['keypoints'], 'cpu') else pose_result['keypoints']
                    scores = pose_result['scores'].cpu().numpy() if hasattr(pose_result['scores'], 'cpu') else pose_result['scores']
                    
                    # 🔥 CRITICAL: Apply confidence filtering at SOURCE
                    filtered_keypoints = []
                    valid_count = 0
                    total_confidence = 0.0
                    
                    for kpt, score in zip(keypoints, scores):
                        if score > conf_threshold:
                            filtered_keypoints.append([kpt[0], kpt[1], score])
                            valid_count += 1
                            total_confidence += score
                        else:
                            filtered_keypoints.append([-1.0, -1.0, 0.0])  # Invalid keypoint marker
                    
                    # Calculate average confidence only from valid keypoints
                    avg_confidence = total_confidence / valid_count if valid_count > 0 else 0.0
                    
                    # print(f"🔥 Horse ViTPose SOURCE filtering: {valid_count}/{len(keypoints)} keypoints above {conf_threshold}, avg_conf: {avg_confidence:.3f}")
                    
                    converted_poses.append({
                        'keypoints': np.array(filtered_keypoints),
                        'box': detections.xyxy[i],
                        'method': 'ViTPose',
                        'confidence': avg_confidence
                    })
                
                return converted_poses
            
            return []
            
        except Exception as e:
            print(f"Error in ViTPose horse estimation: {e}")
            return []
    
    def _estimate_dual_competition(self, frame: np.ndarray, detections):
        """Dual competition mode - ONLY processes detected horse bounding boxes with both methods"""
        if not sv or len(detections) == 0:
            return []
        
        # Get poses from both methods - both already have source-level filtering
        superanimal_poses = self._estimate_superanimal_only(frame, detections)
        vitpose_poses = self._estimate_vitpose_horse_only(frame, detections)
        
        # Competition: Pick best pose per horse based on confidence
        best_poses = []
        num_horses = len(detections)
        
        for i in range(num_horses):
            superanimal_conf = 0.0
            vitpose_conf = 0.0
            
            # Get SuperAnimal confidence
            if i < len(superanimal_poses):
                superanimal_conf = superanimal_poses[i]['confidence']
            
            # Get ViTPose confidence
            if i < len(vitpose_poses):
                vitpose_conf = vitpose_poses[i]['confidence']
            
            # Pick the winner
            if superanimal_conf > vitpose_conf and i < len(superanimal_poses):
                best_poses.append(superanimal_poses[i])
                print(f"🏆 Horse {i+1}: SuperAnimal wins (conf: {superanimal_conf:.3f} vs {vitpose_conf:.3f})")
            elif vitpose_conf > 0 and i < len(vitpose_poses):
                best_poses.append(vitpose_poses[i])
                print(f"🏆 Horse {i+1}: ViTPose wins (conf: {vitpose_conf:.3f} vs {superanimal_conf:.3f})")
        
        return best_poses