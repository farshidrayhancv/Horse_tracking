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
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False

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
    """Complete RGB-D re-identification pipeline: Detection → Depth (full image) → SAM (bbox) → RGB-D MegaDescriptor"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # Initialize components
        self.sam_predictor = None
        self.sam_model_type = None
        self.depth_pipeline = None
        self.reid_model = None
        self.reid_method = None
        
        # Track embeddings for re-identification
        self.track_embeddings = {}  # track_id -> list of RGB-D embeddings
        self.track_depth_stats = {}  # track_id -> list of depth shape statistics
        self.embedding_history_size = 10
        self.reassignment_count = 0
        
        # Store current frame's segmentation masks for visualization
        self.current_masks = []
        
        self.setup_sam_model()
        self.setup_depth_anything()
        self.setup_megadescriptor()
        
    def setup_sam_model(self):
        """Initialize SAM model (MobileSAM or SAM2) based on config"""
        if self.config.sam_model == 'none':
            print("🚫 SAM segmentation disabled - using simple crops only")
            return
        
        if self.config.sam_model == 'mobilesam':
            self.setup_mobile_sam()
        elif self.config.sam_model == 'sam2':
            self.setup_sam2()
        else:
            print(f"❌ Unknown SAM model: {self.config.sam_model}")
            print(f"Available options: {list(self.config.SAM_MODELS.keys())}")
    
    def setup_mobile_sam(self):
        """Initialize MobileSAM for segmentation"""
        if not MOBILESAM_AVAILABLE:
            print("❌ MobileSAM not available - install with: pip install git+https://github.com/ChaoningZhang/MobileSAM.git")
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
            self.sam_predictor = SamPredictor(sam)
            self.sam_model_type = 'mobilesam'
            print("✅ MobileSAM loaded")
        except Exception as e:
            print(f"❌ MobileSAM setup failed: {e}")
    
    def setup_sam2(self):
        """Initialize SAM2 for segmentation"""
        if not SAM2_AVAILABLE:
            print("❌ SAM2 not available - install with: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
            return
            
        try:
            print("🔄 Loading SAM2 model...")
            # Use the base model for a good balance between speed and accuracy
            self.sam_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-base-plus")
            self.sam_model_type = 'sam2'
            print("✅ SAM2 loaded (facebook/sam2.1-hiera-base-plus)")
        except Exception as e:
            print(f"❌ SAM2 setup failed: {e}")
            print("Falling back to MobileSAM...")
            self.setup_mobile_sam()
    
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
        """Apply SAM segmentation within bounding boxes and crop both RGB and depth"""
        self.current_masks = []  # Reset masks for current frame
        
        if not self.sam_predictor or not sv or len(detections) == 0:
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
                    self.current_masks.append(None)
            return rgb_crops, depth_crops
        
        try:
            # Set image for SAM
            self.sam_predictor.set_image(frame)
            
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
                
                # Generate mask with model-specific inference
                if self.sam_model_type == 'sam2':
                    # SAM2 requires torch inference mode and autocast
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        masks, scores, logits = self.sam_predictor.predict(
                            point_coords=input_point,
                            point_labels=input_label,
                            multimask_output=True,
                        )
                else:
                    # MobileSAM standard inference
                    masks, scores, logits = self.sam_predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True,
                    )
                
                # Select best mask and ensure proper data type
                best_mask = masks[np.argmax(scores)]
                
                # Convert mask to boolean type if needed (SAM2 compatibility)
                if best_mask.dtype != bool:
                    best_mask = best_mask.astype(bool)
                
                # Ensure mask is 2D
                if best_mask.ndim > 2:
                    best_mask = best_mask.squeeze()
                
                self.current_masks.append(best_mask)  # Store full-image mask
                
                # Crop RGB with mask
                rgb_crop = frame[y1:y2, x1:x2].copy()
                mask_crop = best_mask[y1:y2, x1:x2]
                
                # Ensure mask_crop is boolean
                
            return rgb_crops, depth_crops
            
        except Exception as e:
            print(f"❌ SAM segmentation failed: {e}")
            print(f"   Falling back to simple crops without segmentation")
            # Fallback to simple crops
            rgb_crops, depth_crops = [], []
            self.current_masks = []  # Clear masks on fallback
            for box in detections.xyxy:
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    rgb_crop = frame[y1:y2, x1:x2].copy()
                    depth_crop = depth_map[y1:y2, x1:x2].copy()
                    rgb_crops.append(rgb_crop)
                    depth_crops.append(depth_crop)
                    self.current_masks.append(None)
            return rgb_crops, depth_crops
    
    def extract_reid_features(self, rgb_crops: List[np.ndarray], depth_crops: List[np.ndarray]) -> np.ndarray:
        """Extract RGB-D re-identification features using both RGB and depth crops"""
        if len(rgb_crops) == 0:
            return np.array([])
        
        try:
            features = []
            for i, (rgb_crop, depth_crop) in enumerate(zip(rgb_crops, depth_crops)):
                if rgb_crop.size == 0:
                    continue
                
                # Resize both RGB and depth to 384x384 for MegaDescriptor
                rgb_resized = cv2.resize(rgb_crop, (384, 384))
                depth_resized = cv2.resize(depth_crop, (384, 384))
                rgb_final = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2RGB)
                
                if self.reid_method == "megadescriptor_direct":
                    # RGB-D Fusion - Process both modalities and combine features
                    depth_normalized = depth_resized.astype(np.float32) / 255.0
                    depth_3channel = np.stack([depth_normalized] * 3, axis=-1)  # Convert to 3-channel
                    
                    # Create tensors for both modalities
                    rgb_tensor = torch.from_numpy(rgb_final).permute(2, 0, 1).float() / 255.0
                    depth_tensor = torch.from_numpy(depth_3channel).permute(2, 0, 1).float()
                    
                    rgb_tensor = rgb_tensor.unsqueeze(0).to(self.device)
                    depth_tensor = depth_tensor.unsqueeze(0).to(self.device)
                    
                    # Normalize RGB like ImageNet
                    rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
                    rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
                    rgb_tensor = (rgb_tensor - rgb_mean) / rgb_std
                    
                    # Normalize depth differently (depth has different statistics)
                    depth_mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)
                    depth_std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)
                    depth_tensor = (depth_tensor - depth_mean) / depth_std
                    
                    with torch.no_grad():
                        # Extract features from both modalities
                        rgb_feature = self.reid_model(rgb_tensor)
                        depth_feature = self.reid_model(depth_tensor)
                        
                        # Fusion strategy: Weighted combination
                        rgb_weight = 0.7  # RGB is more important for visual appearance
                        depth_weight = 0.3  # Depth provides shape/structure information
                        
                        # L2 normalize each feature separately
                        rgb_feature = F.normalize(rgb_feature, p=2, dim=1)
                        depth_feature = F.normalize(depth_feature, p=2, dim=1)
                        
                        # Weighted fusion
                        fused_feature = rgb_weight * rgb_feature + depth_weight * depth_feature
                        
                        # Final normalization
                        fused_feature = F.normalize(fused_feature, p=2, dim=1)
                        
                        features.append(fused_feature.cpu().numpy())
            
            return np.vstack(features) if features else np.array([])
            
        except Exception as e:
            print(f"❌ RGB-D feature extraction failed: {e}")
            # Fallback to RGB-only processing
            return self.extract_reid_features_rgb_only(rgb_crops)
    
    def extract_reid_features_rgb_only(self, rgb_crops: List[np.ndarray]) -> np.ndarray:
        """Fallback RGB-only feature extraction"""
        if len(rgb_crops) == 0:
            return np.array([])
        
        try:
            features = []
            for i, rgb_crop in enumerate(rgb_crops):
                if rgb_crop.size == 0:
                    continue
                
                rgb_resized = cv2.resize(rgb_crop, (384, 384))
                rgb_final = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2RGB)
                
                if self.reid_method == "megadescriptor_direct":
                    rgb_tensor = torch.from_numpy(rgb_final).permute(2, 0, 1).float() / 255.0
                    rgb_tensor = rgb_tensor.unsqueeze(0).to(self.device)
                    
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
                    rgb_tensor = (rgb_tensor - mean) / std
                    
                    with torch.no_grad():
                        feature = self.reid_model(rgb_tensor)
                        features.append(feature.cpu().numpy())
            
            return np.vstack(features) if features else np.array([])
            
        except Exception as e:
            print(f"❌ RGB feature extraction failed: {e}")
            return np.array([])
    
    def extract_depth_shape_features(self, depth_crop: np.ndarray) -> Dict:
        """Extract geometric shape features from depth for additional matching"""
        if depth_crop.size == 0:
            return {}
        
        try:
            # Normalize depth
            depth_norm = depth_crop.astype(np.float32) / 255.0
            
            # Extract shape statistics
            depth_stats = {
                'mean_depth': np.mean(depth_norm),
                'depth_std': np.std(depth_norm),
                'depth_range': np.max(depth_norm) - np.min(depth_norm),
                'depth_median': np.median(depth_norm)
            }
            
            # Extract contour-based shape features
            depth_binary = (depth_norm > 0.1).astype(np.uint8) * 255
            contours, _ = cv2.findContours(depth_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Shape descriptors
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                else:
                    circularity = 0
                    
                # Bounding box aspect ratio
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio = w / h if h > 0 else 0
                
                depth_stats.update({
                    'area': area,
                    'circularity': circularity,
                    'aspect_ratio': aspect_ratio,
                    'compactness': area / (w * h) if w * h > 0 else 0
                })
            
            return depth_stats
            
        except Exception as e:
            print(f"❌ Depth shape feature extraction failed: {e}")
            return {}
    
    def match_with_depth_consistency(self, query_rgb_feature: np.ndarray, query_depth_stats: Dict, 
                                    track_id: int, rgb_weight: float = 0.8, depth_weight: float = 0.2) -> float:
        """Enhanced matching that combines RGB features with depth shape consistency"""
        
        # RGB feature similarity
        rgb_similarity = self.compute_similarity_with_depth_weighting(query_rgb_feature, track_id)
        
        # Depth shape consistency
        depth_similarity = 0.0
        if track_id in self.track_depth_stats and query_depth_stats:
            track_depth_history = self.track_depth_stats[track_id]
            
            if track_depth_history:
                # Compare shape statistics
                depth_similarities = []
                for historical_stats in track_depth_history:
                    shape_similarity = 0.0
                    comparisons = 0
                    
                    for key in ['aspect_ratio', 'circularity', 'compactness']:
                        if key in query_depth_stats and key in historical_stats:
                            # Normalized difference (smaller difference = higher similarity)
                            diff = abs(query_depth_stats[key] - historical_stats[key])
                            max_val = max(query_depth_stats[key], historical_stats[key], 0.01)
                            shape_similarity += 1.0 - min(1.0, diff / max_val)
                            comparisons += 1
                    
                    if comparisons > 0:
                        depth_similarities.append(shape_similarity / comparisons)
                
                if depth_similarities:
                    depth_similarity = np.max(depth_similarities)
        
        # Combined similarity
        combined_similarity = rgb_weight * rgb_similarity + depth_weight * depth_similarity
        
        return combined_similarity
    
    def compute_similarity_with_depth_weighting(self, query_feature: np.ndarray, track_id: int) -> float:
        """Enhanced similarity computation that considers depth consistency"""
        if track_id not in self.track_embeddings:
            return 0.0
        
        track_features = np.array(self.track_embeddings[track_id])
        if len(track_features) == 0:
            return 0.0
        
        similarities = []
        for track_feature in track_features:
            # Cosine similarity (features are already normalized)
            similarity = np.dot(query_feature.flatten(), track_feature.flatten())
            similarities.append(similarity)
        
        # Use maximum similarity for best match
        max_similarity = np.max(similarities)
        
        # Bonus: If we have multiple consistent matches, boost confidence
        high_similarity_count = np.sum(np.array(similarities) > 0.4)
        consistency_bonus = min(0.1, high_similarity_count * 0.02)
        
        return max_similarity + consistency_bonus
    
    def update_track_embeddings_with_depth(self, track_ids: np.ndarray, features: np.ndarray, depth_stats: List[Dict]):
        """Update embedding and depth statistics history for tracked objects"""
        if len(track_ids) != len(features) or len(track_ids) != len(depth_stats):
            return
        
        for track_id, feature, depth_stat in zip(track_ids, features, depth_stats):
            # Update RGB-D embeddings
            if track_id not in self.track_embeddings:
                self.track_embeddings[track_id] = []
            self.track_embeddings[track_id].append(feature)
            if len(self.track_embeddings[track_id]) > self.embedding_history_size:
                self.track_embeddings[track_id].pop(0)
            
            # Update depth shape statistics
            if track_id not in self.track_depth_stats:
                self.track_depth_stats[track_id] = []
            if depth_stat:  # Only add if we have valid depth statistics
                self.track_depth_stats[track_id].append(depth_stat)
                if len(self.track_depth_stats[track_id]) > self.embedding_history_size:
                    self.track_depth_stats[track_id].pop(0)
    
    def find_best_match_with_depth(self, query_feature: np.ndarray, query_depth_stats: Dict, threshold: float = 0.35) -> int:
        """Find best matching track ID using RGB-D features and depth shape consistency"""
        if len(self.track_embeddings) == 0:
            return -1
        
        best_similarity = 0.0
        best_track_id = -1
        
        for track_id in self.track_embeddings:
            # Enhanced similarity that combines RGB-D features with depth shape consistency
            similarity = self.match_with_depth_consistency(query_feature, query_depth_stats, track_id)
            
            if similarity > best_similarity and similarity > threshold:
                best_similarity = similarity
                best_track_id = track_id
        
        if best_track_id >= 0:
            model_name = "SAM2" if self.sam_model_type == 'sam2' else "MobileSAM"
            print(f"🔍 RGB-D+{model_name} match: Track {best_track_id} with similarity {best_similarity:.3f}")
        
        return best_track_id
    
    def process_frame(self, frame: np.ndarray, detections) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray, List[Dict]]:
        """Complete pipeline processing for a frame with enhanced depth utilization"""
        # Step 1: Estimate depth for entire image
        depth_map = self.estimate_depth_full_image(frame)
        
        # Step 2: Segment and crop both RGB and depth within bounding boxes
        rgb_crops, depth_crops = self.segment_and_crop_with_depth(frame, depth_map, detections)
        
        # Step 3: Extract RGB-D re-identification features
        rgbd_features = self.extract_reid_features(rgb_crops, depth_crops)
        
        # Step 4: Extract depth shape statistics for additional matching
        depth_shape_stats = []
        for depth_crop in depth_crops:
            stats = self.extract_depth_shape_features(depth_crop)
            depth_shape_stats.append(stats)
        
        return rgb_crops, depth_crops, depth_map, rgbd_features, depth_shape_stats
    
    def enhance_tracking(self, detections, reid_features, depth_stats=None):
        """Enhanced tracking with RGB-D re-identification and depth consistency"""
        if not sv or len(detections) == 0 or len(reid_features) == 0:
            return detections
        
        # Use empty depth stats if not provided
        if depth_stats is None:
            depth_stats = [{}] * len(reid_features)
        
        # Create a copy to modify
        enhanced_detections = detections
        
        if hasattr(detections, 'tracker_id'):
            original_track_ids = enhanced_detections.tracker_id.copy()
            
            # Update embeddings for existing valid tracks
            valid_tracks = enhanced_detections.tracker_id >= 0
            if np.any(valid_tracks):
                valid_track_ids = enhanced_detections.tracker_id[valid_tracks]
                valid_features = reid_features[valid_tracks]
                valid_depth_stats = [depth_stats[i] for i in np.where(valid_tracks)[0]]
                self.update_track_embeddings_with_depth(valid_track_ids, valid_features, valid_depth_stats)
            
            # Process lost/unassigned tracks for re-identification
            lost_tracks = enhanced_detections.tracker_id < 0
            if np.any(lost_tracks):
                lost_indices = np.where(lost_tracks)[0]
                lost_features = reid_features[lost_tracks]
                lost_depth_stats = [depth_stats[i] for i in lost_indices]
                
                for i, (feature, depth_stat) in enumerate(zip(lost_features, lost_depth_stats)):
                    original_idx = lost_indices[i]
                    
                    # Try to find a match using RGB-D features and depth consistency
                    best_match = self.find_best_match_with_depth(feature, depth_stat, threshold=0.35)
                    
                    if best_match >= 0:
                        # Reassign the track ID
                        enhanced_detections.tracker_id[original_idx] = best_match
                        self.reassignment_count += 1
                        model_name = "SAM2" if self.sam_model_type == 'sam2' else "MobileSAM"
                        print(f"🔄 RGB-D+{model_name} Re-identified: Detection → Track {best_match} (reassignment #{self.reassignment_count})")
                        
                        # Update the track's embeddings with this new feature
                        if best_match not in self.track_embeddings:
                            self.track_embeddings[best_match] = []
                        self.track_embeddings[best_match].append(feature)
                        
                        if len(self.track_embeddings[best_match]) > self.embedding_history_size:
                            self.track_embeddings[best_match].pop(0)
                        
                        # Update depth statistics
                        if depth_stat and best_match not in self.track_depth_stats:
                            self.track_depth_stats[best_match] = []
                        if depth_stat:
                            self.track_depth_stats[best_match].append(depth_stat)
                            if len(self.track_depth_stats[best_match]) > self.embedding_history_size:
                                self.track_depth_stats[best_match].pop(0)
        
        return enhanced_detections
    
    def get_current_masks(self):
        """Get current frame's segmentation masks for visualization"""
        return self.current_masks
    
    def get_reassignment_count(self):
        """Get total number of track reassignments performed"""
        return self.reassignment_count