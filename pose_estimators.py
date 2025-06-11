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
        if self.config.human_pose_estimator == 'none':
            return []
        elif self.config.human_pose_estimator == 'vitpose':
            return self._estimate_vitpose_human(frame, detections)
        else:
            return []
    
    def estimate_horse_poses(self, frame: np.ndarray, detections):
        if self.config.horse_pose_estimator == 'none':
            return []
        elif self.config.horse_pose_estimator == 'superanimal':
            return self._estimate_superanimal_only(frame, detections)
        elif self.config.horse_pose_estimator == 'vitpose':
            return self._estimate_vitpose_horse_only(frame, detections)
        elif self.config.horse_pose_estimator == 'dual':
            return self._estimate_dual_competition(frame, detections)
        else:
            return []
    
    def _estimate_vitpose_human(self, frame: np.ndarray, detections):
        if not self.vitpose_model or not sv or len(detections) == 0:
            return []
        
        try:
            from PIL import Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Convert to COCO format for ViTPose
            coco_boxes = detections.xyxy.copy()
            coco_boxes[:, 2] = coco_boxes[:, 2] - coco_boxes[:, 0]  # width
            coco_boxes[:, 3] = coco_boxes[:, 3] - coco_boxes[:, 1]  # height
            
            inputs = self.vitpose_processor(pil_image, boxes=[coco_boxes], return_tensors="pt").to(self.config.device)
            
            with torch.no_grad():
                outputs = self.vitpose_model(**inputs)
            
            pose_results = self.vitpose_processor.post_process_pose_estimation(outputs, boxes=[coco_boxes])
            return pose_results[0] if pose_results else []
            
        except Exception as e:
            print(f"Error in human pose estimation: {e}")
            return []
    
    def _estimate_superanimal_only(self, frame: np.ndarray, detections):
        if not self.superanimal:
            return []
        
        return self.superanimal.estimate_pose(frame, detections)
    
    def _estimate_vitpose_horse_only(self, frame: np.ndarray, detections):
        if not self.vitpose_horse_model or not sv or len(detections) == 0:
            return []
        
        try:
            from PIL import Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Convert to COCO format
            coco_boxes = detections.xyxy.copy()
            coco_boxes[:, 2] = coco_boxes[:, 2] - coco_boxes[:, 0]  # width
            coco_boxes[:, 3] = coco_boxes[:, 3] - coco_boxes[:, 1]  # height
            
            inputs = self.vitpose_horse_processor(pil_image, boxes=[coco_boxes], return_tensors="pt").to(self.config.device)
            
            with torch.no_grad():
                outputs = self.vitpose_horse_model(**inputs)
            
            pose_results = self.vitpose_horse_processor.post_process_pose_estimation(outputs, boxes=[coco_boxes])
            
            if pose_results and len(pose_results[0]) > 0:
                converted_poses = []
                for i, pose_result in enumerate(pose_results[0]):
                    keypoints = pose_result['keypoints'].cpu().numpy() if hasattr(pose_result['keypoints'], 'cpu') else pose_result['keypoints']
                    scores = pose_result['scores'].cpu().numpy() if hasattr(pose_result['scores'], 'cpu') else pose_result['scores']
                    
                    # Convert to our format
                    kpts_with_conf = []
                    for kpt, score in zip(keypoints, scores):
                        kpts_with_conf.append([kpt[0], kpt[1], score])
                    
                    converted_poses.append({
                        'keypoints': np.array(kpts_with_conf),
                        'box': detections.xyxy[i],
                        'method': 'ViTPose',
                        'confidence': np.mean(scores)
                    })
                
                return converted_poses
            
            return []
            
        except Exception as e:
            print(f"Error in ViTPose horse estimation: {e}")
            return []
    
    def _estimate_dual_competition(self, frame: np.ndarray, detections):
        if not sv or len(detections) == 0:
            return []
        
        # Get poses from both methods
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
            elif vitpose_conf > 0 and i < len(vitpose_poses):
                best_poses.append(vitpose_poses[i])
        
        return best_poses