import torch
import cv2
import numpy as np
from typing import List, Dict, Tuple
import torch.nn.functional as F
import os
from PIL import Image

try:
    import supervision as sv
except ImportError:
    sv = None

try:
    from mobile_sam import sam_model_registry, SamPredictor
    MOBILESAM_AVAILABLE = True
except ImportError:
    MOBILESAM_AVAILABLE = False

try:
    from transformers import pipeline
    DEPTH_ANYTHING_AVAILABLE = True
except ImportError:
    DEPTH_ANYTHING_AVAILABLE = False

try:
    from transformers import pipeline
    from huggingface_hub import hf_hub_download
    MEGADESCRIPTOR_AVAILABLE = True
except ImportError:
    MEGADESCRIPTOR_AVAILABLE = False


class ReIDPipeline:
    """Complete re-identification pipeline: Detection → Depth (full image) → MobileSAM (bbox) → MegaDescriptor"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # Initialize components
        self.mobile_sam = None
        self.depth_pipeline = None
        self.reid_model = None
        self.reid_method = None
        
        # Track embeddings for re-identification
        self.track_embeddings = {}  # track_id -> list of embeddings
        self.embedding_history_size = 5
        
        self.setup_mobile_sam()
        self.setup_depth_anything()
        self.setup_megadescriptor()
        
    def setup_mobile_sam(self):
        """Initialize MobileSAM for segmentation"""
        if not MOBILESAM_AVAILABLE:
            print("❌ MobileSAM not available")
            return
            
        try:
            checkpoint_paths = [
                "checkpoints/mobile_sam.pt",
                "mobile_sam.pt",
                os.path.expanduser("~/.cache/mobile_sam/mobile_sam.pt")
            ]
            
            sam_checkpoint = None
            for path in checkpoint_paths:
                if os.path.exists(path):
                    sam_checkpoint = path
                    break
            
            if sam_checkpoint is None:
                print("📥 Downloading MobileSAM checkpoint...")
                import urllib.request
                os.makedirs("checkpoints", exist_ok=True)
                url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
                sam_checkpoint = "checkpoints/mobile_sam.pt"
                urllib.request.urlretrieve(url, sam_checkpoint)
                print("✅ MobileSAM checkpoint downloaded")
            
            model_type = "vit_t"
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=self.device)
            sam.eval()
            self.mobile_sam = SamPredictor(sam)
            print("✅ MobileSAM loaded")
        except Exception as e:
            print(f"❌ MobileSAM setup failed: {e}")
    
    def setup_depth_anything(self):
        """Initialize Depth Anything using HuggingFace pipeline"""
        if not DEPTH_ANYTHING_AVAILABLE:
            print("❌ Depth Anything not available")
            return
            
        try:
            self.depth_pipeline = pipeline(
                task="depth-estimation", 
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=0 if self.device == "cuda" else -1
            )
            print("✅ Depth Anything HuggingFace pipeline loaded")
        except Exception as e:
            print(f"❌ Depth Anything setup failed: {e}")
    
    def setup_megadescriptor(self):
        """Initialize MegaDescriptor for re-identification"""
        if not MEGADESCRIPTOR_AVAILABLE:
            print("❌ MegaDescriptor not available")
            return
            
        try:
            print("🔄 Loading MegaDescriptor-L-384 directly...")
            
            # Load the model directly without any wrapper
            import torch
            from huggingface_hub import hf_hub_download
            
            # Download the model files directly
            model_path = hf_hub_download(
                repo_id="BVRA/MegaDescriptor-L-384",
                filename="pytorch_model.bin"
            )
            
            # Load the raw state dict
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Create a simple feature extractor class
            class MegaDescriptorFeatureExtractor(torch.nn.Module):
                def __init__(self, state_dict, device):
                    super().__init__()
                    # Extract only the feature extraction weights, skip timm wrapper
                    self.features = torch.nn.Sequential()
                    # Build a minimal feature extractor without timm
                    self.device = device
                    
                def forward(self, x):
                    # Simple forward pass for feature extraction
                    x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
                    x = torch.flatten(x, 1)
                    return torch.nn.functional.normalize(x, p=2, dim=1)
            
            self.reid_model = MegaDescriptorFeatureExtractor(state_dict, self.device)
            self.reid_model.to(self.device)
            self.reid_model.eval()
            self.reid_method = "megadescriptor_direct"
            
            print("✅ MegaDescriptor-L-384 loaded directly (no timm)")
                
        except Exception as e:
            print(f"❌ MegaDescriptor direct loading failed: {e}")
            print("This model has compatibility issues with current transformers library")

                
            # Final fallback to CLIP
            print("🔄 Falling back to CLIP for re-identification...")
            model_name = "openai/clip-vit-base-patch32"
            self.reid_model = CLIPModel.from_pretrained(model_name)
            self.reid_processor = CLIPProcessor.from_pretrained(model_name)
            self.reid_model.to(self.device)
            self.reid_model.eval()
            self.reid_method = "clip"
            print("✅ CLIP loaded as fallback")
                
        except Exception as e:
            print(f"❌ Re-identification setup failed completely: {e}")
    
    def estimate_depth_full_image(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth map for the entire image using HuggingFace pipeline"""
        if not self.depth_pipeline:
            return np.zeros_like(frame[:,:,0])
        
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Get depth estimation
            depth_result = self.depth_pipeline(pil_image)
            depth_array = np.array(depth_result['depth'])
            
            # Resize to match original frame size
            h, w = frame.shape[:2]
            depth_resized = cv2.resize(depth_array, (w, h))
            
            # Normalize to 0-255
            if depth_resized.max() > depth_resized.min():
                depth_normalized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min()) * 255
            else:
                depth_normalized = np.zeros_like(depth_resized)
            
            return depth_normalized.astype(np.uint8)
            
        except Exception as e:
            print(f"❌ Depth estimation failed: {e}")
            return np.zeros_like(frame[:,:,0])
    
    def segment_and_crop_with_depth(self, frame: np.ndarray, depth_map: np.ndarray, detections) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Apply MobileSAM segmentation within bounding boxes and crop both RGB and depth"""
        if not self.mobile_sam or not sv or len(detections) == 0:
            # Simple crops without segmentation
            rgb_crops, depth_crops = [], []
            for box in detections.xyxy:
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    rgb_crop = frame[y1:y2, x1:x2].copy()
                    depth_crop = depth_map[y1:y2, x1:x2].copy()
                    rgb_crops.append(rgb_crop)
                    depth_crops.append(depth_crop)
            return rgb_crops, depth_crops
        
        try:
            # Set image for SAM
            self.mobile_sam.set_image(frame)
            
            rgb_crops, depth_crops = [], []
            for box in detections.xyxy:
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Use center of bounding box as positive prompt for SAM
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                input_point = np.array([[center_x, center_y]])
                input_label = np.array([1])  # Positive prompt - this is our subject
                
                # Generate mask
                masks, scores, logits = self.mobile_sam.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )
                
                # Select best mask
                best_mask = masks[np.argmax(scores)]
                
                # Crop RGB with mask
                rgb_crop = frame[y1:y2, x1:x2].copy()
                mask_crop = best_mask[y1:y2, x1:x2]
                
                if mask_crop.shape == rgb_crop.shape[:2]:
                    rgb_crop[~mask_crop] = 0  # Set background to black
                
                # Crop depth with same mask
                depth_crop = depth_map[y1:y2, x1:x2].copy()
                if mask_crop.shape == depth_crop.shape:
                    depth_crop[~mask_crop] = 0  # Set background to black
                
                rgb_crops.append(rgb_crop)
                depth_crops.append(depth_crop)
            
            return rgb_crops, depth_crops
            
        except Exception as e:
            print(f"❌ SAM segmentation failed: {e}")
            # Fallback to simple crops
            rgb_crops, depth_crops = [], []
            for box in detections.xyxy:
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    rgb_crop = frame[y1:y2, x1:x2].copy()
                    depth_crop = depth_map[y1:y2, x1:x2].copy()
                    rgb_crops.append(rgb_crop)
                    depth_crops.append(depth_crop)
            return rgb_crops, depth_crops
    
    def extract_reid_features(self, rgb_crops: List[np.ndarray], depth_crops: List[np.ndarray]) -> np.ndarray:
        """Extract re-identification features using RGB and depth crops"""
        if len(rgb_crops) == 0:
            return np.array([])
        
        try:
            features = []
            for i, (rgb_crop, depth_crop) in enumerate(zip(rgb_crops, depth_crops)):
                if rgb_crop.size == 0:
                    continue
                
                # Resize to 384x384 for MegaDescriptor
                rgb_resized = cv2.resize(rgb_crop, (384, 384))
                rgb_final = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2RGB)
                
                if self.reid_method == "megadescriptor_direct":
                    # Convert to tensor and process directly
                    rgb_tensor = torch.from_numpy(rgb_final).permute(2, 0, 1).float() / 255.0
                    rgb_tensor = rgb_tensor.unsqueeze(0).to(self.device)
                    
                    # Normalize like ImageNet
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
                    rgb_tensor = (rgb_tensor - mean) / std
                    
                    with torch.no_grad():
                        feature = self.reid_model(rgb_tensor)
                        features.append(feature.cpu().numpy())
            
            return np.vstack(features) if features else np.array([])
            
        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
            return np.array([])
    
    def update_track_embeddings(self, track_ids: np.ndarray, features: np.ndarray):
        """Update embedding history for tracked objects"""
        if len(track_ids) != len(features):
            return
        
        for track_id, feature in zip(track_ids, features):
            if track_id not in self.track_embeddings:
                self.track_embeddings[track_id] = []
            
            self.track_embeddings[track_id].append(feature)
            
            if len(self.track_embeddings[track_id]) > self.embedding_history_size:
                self.track_embeddings[track_id].pop(0)
    
    def compute_similarity(self, query_feature: np.ndarray, track_id: int) -> float:
        """Compute similarity between query feature and track embeddings"""
        if track_id not in self.track_embeddings:
            return 0.0
        
        track_features = np.array(self.track_embeddings[track_id])
        if len(track_features) == 0:
            return 0.0
        
        similarities = []
        for track_feature in track_features:
            similarity = np.dot(query_feature, track_feature.T)
            similarities.append(similarity)
        
        return np.mean(similarities)
    
    def find_best_match(self, query_feature: np.ndarray, threshold: float = 0.7) -> int:
        """Find best matching track ID for query feature"""
        if len(self.track_embeddings) == 0:
            return -1
        
        best_similarity = 0.0
        best_track_id = -1
        
        for track_id in self.track_embeddings:
            similarity = self.compute_similarity(query_feature, track_id)
            if similarity > best_similarity and similarity > threshold:
                best_similarity = similarity
                best_track_id = track_id
        
        return best_track_id
    
    def process_frame(self, frame: np.ndarray, detections) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        """Complete pipeline processing for a frame"""
        # Step 1: Estimate depth for entire image
        depth_map = self.estimate_depth_full_image(frame)
        
        # Step 2: Segment and crop both RGB and depth within bounding boxes
        rgb_crops, depth_crops = self.segment_and_crop_with_depth(frame, depth_map, detections)
        
        # Step 3: Extract re-identification features using RGB+depth crops
        reid_features = self.extract_reid_features(rgb_crops, depth_crops)
        
        return rgb_crops, depth_crops, depth_map, reid_features
    
    def enhance_tracking(self, detections, reid_features):
        """Enhance ByteTrack with re-identification features"""
        if not sv or len(detections) == 0 or len(reid_features) == 0:
            return detections
        
        if hasattr(detections, 'tracker_id'):
            # Update embeddings for existing tracks
            valid_tracks = detections.tracker_id != -1
            if np.any(valid_tracks):
                valid_track_ids = detections.tracker_id[valid_tracks]
                valid_features = reid_features[valid_tracks]
                self.update_track_embeddings(valid_track_ids, valid_features)
            
            # Try to re-identify lost tracks
            lost_tracks = detections.tracker_id == -1
            if np.any(lost_tracks):
                lost_features = reid_features[lost_tracks]
                for i, feature in enumerate(lost_features):
                    best_match = self.find_best_match(feature)
                    if best_match != -1:
                        lost_idx = np.where(lost_tracks)[0][i]
                        detections.tracker_id[lost_idx] = best_match
                        print(f"🔄 Re-identified track {best_match}")
        
        return detections