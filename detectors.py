import cv2
import numpy as np
from typing import List, Dict, Any, Optional
import requests
import base64
from io import BytesIO
from PIL import Image

try:
    import supervision as sv
except ImportError:
    sv = None

try:
    from inference import get_model
    ROBOFLOW_SDK_AVAILABLE = True
except ImportError:
    ROBOFLOW_SDK_AVAILABLE = False
    print("⚠️ Roboflow inference SDK not available - install with: pip install inference")

class RacerDetectionManager:
    """Simplified detection using Roboflow model for horse+jockey compound entities"""
    
    def __init__(self, config):
        self.config = config
        self.roboflow_model = None
        self.setup_roboflow_model()
    
    def setup_roboflow_model(self):
        """Initialize Roboflow model"""
        if not hasattr(self.config, 'roboflow_api_key') or not self.config.roboflow_api_key:
            print("❌ Roboflow API key not configured")
            return
        
        if not hasattr(self.config, 'roboflow_model_id') or not self.config.roboflow_model_id:
            print("❌ Roboflow model ID not configured")
            return
        
        if ROBOFLOW_SDK_AVAILABLE:
            try:
                self.roboflow_model = get_model(
                    model_id=self.config.roboflow_model_id, 
                    api_key=self.config.roboflow_api_key
                )
                print(f"✅ Roboflow model loaded: {self.config.roboflow_model_id}")
            except Exception as e:
                print(f"❌ Failed to load Roboflow model: {e}")
                print("Check API key and model ID in config")
        else:
            print("❌ Roboflow SDK not available - install with: pip install inference")
    
    def detect_racers(self, frame: np.ndarray) -> 'sv.Detections':
        """Detect horse+jockey compound entities using Roboflow model"""
        if self.roboflow_model:
            return self._detect_with_sdk(frame)
        else:
            print("❌ No Roboflow model available")
            return sv.Detections.empty() if sv else []
    
    def _detect_with_sdk(self, frame: np.ndarray) -> 'sv.Detections':
        """Detect using Roboflow SDK - fixed to match working approach"""
        try:
            # Convert BGR to RGB for Roboflow
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference - get first result like in working code
            results = self.roboflow_model.infer(
                frame_rgb,
                confidence=getattr(self.config, 'roboflow_confidence', 0.5),
                iou_threshold=0.5
            )[0]  # Take first result like working code
            
            # Use supervision's built-in conversion - same as working code
            detections = sv.Detections.from_inference(results)
            
            print(f"🔍 Roboflow SDK: {len(detections)} detections")
            if len(detections) > 0:
                print(f"   Confidences: {detections.confidence}")
                print(f"   Has masks: {detections.mask is not None}")
            
            return detections
            
        except Exception as e:
            print(f"❌ Roboflow SDK detection failed: {e}")
            import traceback
            traceback.print_exc()
            return sv.Detections.empty() if sv else []
    
    def get_masks(self, detections) -> List[np.ndarray]:
        """Get segmentation masks from detections"""
        if hasattr(detections, 'mask') and detections.mask is not None:
            return [mask for mask in detections.mask]
        else:
            # Generate rectangular masks from bounding boxes if no masks
            masks = []
            if hasattr(detections, 'xyxy') and len(detections) > 0:
                # Default frame size - should be passed from actual frame
                frame_h, frame_w = 1080, 1920  
                for bbox in detections.xyxy:
                    mask = np.zeros((frame_h, frame_w), dtype=bool)
                    x1, y1, x2, y2 = map(int, bbox)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame_w, x2), min(frame_h, y2)
                    mask[y1:y2, x1:x2] = True
                    masks.append(mask)
            return masks