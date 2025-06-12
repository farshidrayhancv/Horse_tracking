import torch
import cv2
import numpy as np
from typing import List

try:
    import supervision as sv
except ImportError:
    sv = None

try:
    from transformers import AutoProcessor, RTDetrForObjectDetection
    RTDETR_AVAILABLE = True
except ImportError:
    RTDETR_AVAILABLE = False

class DetectionManager:
    def __init__(self, config, superanimal_model=None):
        self.config = config
        self.superanimal = superanimal_model
        
        # If no SuperAnimal provided but needed, create it with config
        if not self.superanimal and self.config.horse_detector in ['superanimal', 'both']:
            from models import SuperAnimalQuadruped
            self.superanimal = SuperAnimalQuadruped(device=self.config.device, config=self.config)  # 🔥 Pass config
        
        # Setup RT-DETR detector
        self.rtdetr_detector = None
        self.rtdetr_processor = None
        self.setup_rtdetr()
    
    def setup_rtdetr(self):
        if self.config.human_detector == 'rtdetr' or self.config.horse_detector in ['rtdetr', 'both']:
            if RTDETR_AVAILABLE:
                try:
                    self.rtdetr_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
                    self.rtdetr_detector = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
                    self.rtdetr_detector.to(self.config.device)
                    print("✅ RT-DETR detector loaded")
                except Exception as e:
                    print(f"⚠️ RT-DETR failed: {e}")
    
    def detect_humans(self, frame: np.ndarray):
        if self.config.human_detector == 'rtdetr' and self.rtdetr_detector:
            return self._detect_rtdetr(frame, class_filter=[0], confidence=self.config.confidence_human_detection)
        else:
            return sv.Detections.empty() if sv else []
    
    def detect_horses(self, frame: np.ndarray):
        if self.config.horse_detector == 'rtdetr' and self.rtdetr_detector:
            return self._detect_rtdetr(frame, class_filter=[17], confidence=self.config.confidence_horse_detection)
        elif self.config.horse_detector == 'superanimal' and self.superanimal:
            return self.superanimal.detect_quadrupeds(frame, self.config.confidence_horse_detection)
        elif self.config.horse_detector == 'both':
            horse_detections = self._detect_rtdetr(frame, class_filter=[17], confidence=self.config.confidence_horse_detection)
            if sv and len(horse_detections) == 0 and self.superanimal:
                horse_detections = self.superanimal.detect_quadrupeds(frame, self.config.confidence_horse_detection)
            return horse_detections
        else:
            return sv.Detections.empty() if sv else []
    
    def _detect_rtdetr(self, frame: np.ndarray, class_filter: List[int], confidence: float = None):
        if not self.rtdetr_detector or not self.rtdetr_processor:
            return sv.Detections.empty() if sv else []
        
        conf_threshold = confidence if confidence is not None else 0.3
        
        try:
            from PIL import Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            inputs = self.rtdetr_processor(images=pil_image, return_tensors="pt").to(self.config.device)
            
            with torch.no_grad():
                outputs = self.rtdetr_detector(**inputs)
            
            results = self.rtdetr_processor.post_process_object_detection(
                outputs, 
                target_sizes=torch.tensor([(pil_image.height, pil_image.width)]), 
                threshold=conf_threshold
            )
            
            if len(results) > 0:
                result = results[0]
                
                # Filter for desired classes - ensure class_filter tensor is on same device
                class_filter_tensor = torch.tensor(class_filter, device=result["labels"].device)
                class_mask = torch.isin(result["labels"], class_filter_tensor)
                
                if class_mask.any():
                    filtered_boxes = result["boxes"][class_mask].cpu().numpy()
                    filtered_scores = result["scores"][class_mask].cpu().numpy()
                    filtered_labels = result["labels"][class_mask].cpu().numpy()
                    
                    if sv:
                        return sv.Detections(
                            xyxy=filtered_boxes,
                            confidence=filtered_scores,
                            class_id=filtered_labels
                        )
                    else:
                        # Convert to COCO format for legacy support
                        coco_boxes = filtered_boxes.copy()
                        coco_boxes[:, 2] = coco_boxes[:, 2] - coco_boxes[:, 0]  # width
                        coco_boxes[:, 3] = coco_boxes[:, 3] - coco_boxes[:, 1]  # height
                        return coco_boxes
            
            return sv.Detections.empty() if sv else []
            
        except Exception as e:
            print(f"Error in RT-DETR detection: {e}")
            return sv.Detections.empty() if sv else []
    
    def filter_jockeys(self, human_detections, horse_detections):
        if not sv:
            return human_detections
        
        if len(human_detections) == 0 or len(horse_detections) == 0:
            return sv.Detections.empty()
        
        jockey_indices = []
        
        for i, human_box in enumerate(human_detections.xyxy):
            hx1, hy1, hx2, hy2 = human_box
            hw, hh = hx2 - hx1, hy2 - hy1
            
            for horse_box in horse_detections.xyxy:
                rx1, ry1, rx2, ry2 = horse_box
                
                # Calculate intersection
                ix1 = max(hx1, rx1)
                iy1 = max(hy1, ry1)
                ix2 = min(hx2, rx2)
                iy2 = min(hy2, ry2)
                
                if ix1 < ix2 and iy1 < iy2:
                    intersection_area = (ix2 - ix1) * (iy2 - iy1)
                    human_area = hw * hh
                    overlap_ratio = intersection_area / human_area if human_area > 0 else 0
                    
                    if overlap_ratio >= self.config.jockey_overlap_threshold:
                        jockey_indices.append(i)
                        break
        
        if jockey_indices:
            return sv.Detections(
                xyxy=human_detections.xyxy[jockey_indices],
                confidence=human_detections.confidence[jockey_indices],
                class_id=human_detections.class_id[jockey_indices]
            )
        else:
            return sv.Detections.empty()