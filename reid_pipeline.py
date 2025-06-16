"""
COMPLETE IMPROVED SAMURAI-Enhanced ReID Pipeline
Replace your existing reid_pipeline.py with this complete version
"""

import torch
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch.nn.functional as F
from collections import deque, defaultdict
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


class MotionAwareMemory:
    """Enhanced motion-aware memory bank for tracking objects across frames"""
    
    def __init__(self, memory_size=15, motion_weight=0.3):
        self.memory_size = memory_size
        self.motion_weight = motion_weight
        
        # Memory banks per track
        self.mask_memory = defaultdict(deque)
        self.bbox_memory = defaultdict(deque)
        self.feature_memory = defaultdict(deque)
        self.position_memory = defaultdict(deque)
        self.confidence_memory = defaultdict(deque)
        
        # Motion prediction
        self.velocity_memory = defaultdict(deque)
        self.acceleration_memory = defaultdict(deque)
        
        # Track quality metrics
        self.track_quality = defaultdict(float)
        self.track_consistency = defaultdict(list)
    
    def update_track_memory(self, track_id: int, mask: np.ndarray, bbox: np.ndarray, 
                           feature: np.ndarray, confidence: float):
        """Update memory bank for a track with enhanced quality tracking"""
        
        # Calculate center position
        x1, y1, x2, y2 = bbox
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        
        # Update memories
        self.mask_memory[track_id].append(mask)
        self.bbox_memory[track_id].append(bbox)
        self.feature_memory[track_id].append(feature)
        self.position_memory[track_id].append(center)
        self.confidence_memory[track_id].append(confidence)
        
        # Calculate motion if we have previous positions
        if len(self.position_memory[track_id]) >= 2:
            prev_center = self.position_memory[track_id][-2]
            velocity = center - prev_center
            self.velocity_memory[track_id].append(velocity)
            
            # Calculate acceleration
            if len(self.velocity_memory[track_id]) >= 2:
                prev_velocity = self.velocity_memory[track_id][-2]
                acceleration = velocity - prev_velocity
                self.acceleration_memory[track_id].append(acceleration)
        
        # Update track quality
        self.update_track_quality(track_id, confidence)
        
        # Maintain memory size limits
        for memory_bank in [self.mask_memory, self.bbox_memory, self.feature_memory,
                           self.position_memory, self.confidence_memory,
                           self.velocity_memory, self.acceleration_memory]:
            if len(memory_bank[track_id]) > self.memory_size:
                memory_bank[track_id].popleft()
    
    def update_track_quality(self, track_id: int, confidence: float):
        """Update track quality metrics"""
        # Moving average of confidence
        if track_id in self.track_quality:
            self.track_quality[track_id] = 0.8 * self.track_quality[track_id] + 0.2 * confidence
        else:
            self.track_quality[track_id] = confidence
        
        # Track consistency (variance in recent confidences)
        self.track_consistency[track_id].append(confidence)
        if len(self.track_consistency[track_id]) > 10:
            self.track_consistency[track_id].pop(0)
    
    def predict_next_position(self, track_id: int) -> Optional[np.ndarray]:
        """Enhanced position prediction with physics constraints"""
        if track_id not in self.position_memory or len(self.position_memory[track_id]) == 0:
            return None
        
        positions = list(self.position_memory[track_id])
        
        if len(positions) == 1:
            return positions[0]
        
        # Use multiple prediction methods and combine
        predictions = []
        
        # Method 1: Simple linear prediction
        if len(positions) >= 2:
            velocity = positions[-1] - positions[-2]
            linear_pred = positions[-1] + velocity
            predictions.append(('linear', linear_pred, 0.3))
        
        # Method 2: Velocity-based with acceleration
        if len(self.velocity_memory[track_id]) > 0:
            velocity = self.velocity_memory[track_id][-1]
            
            if len(self.acceleration_memory[track_id]) > 0:
                acceleration = self.acceleration_memory[track_id][-1]
                # Limit acceleration to realistic values
                acc_magnitude = np.linalg.norm(acceleration)
                if acc_magnitude > 5.0:  # Max acceleration constraint
                    acceleration = acceleration / acc_magnitude * 5.0
                
                accel_pred = positions[-1] + velocity + 0.5 * acceleration
                predictions.append(('acceleration', accel_pred, 0.4))
            else:
                vel_pred = positions[-1] + velocity
                predictions.append(('velocity', vel_pred, 0.3))
        
        # Method 3: Polynomial fit for smooth trajectories
        if len(positions) >= 4:
            try:
                times = np.arange(len(positions))
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                
                # Fit polynomial
                x_poly = np.polyfit(times, x_coords, min(2, len(positions)-1))
                y_poly = np.polyfit(times, y_coords, min(2, len(positions)-1))
                
                next_time = len(positions)
                poly_x = np.polyval(x_poly, next_time)
                poly_y = np.polyval(y_poly, next_time)
                poly_pred = np.array([poly_x, poly_y])
                
                predictions.append(('polynomial', poly_pred, 0.4))
            except:
                pass
        
        if not predictions:
            return positions[-1]
        
        # Weighted combination of predictions
        total_weight = sum(weight for _, _, weight in predictions)
        if total_weight == 0:
            return positions[-1]
        
        combined_pred = np.zeros(2)
        for method, pred, weight in predictions:
            combined_pred += pred * (weight / total_weight)
        
        # Apply speed constraints
        if len(positions) >= 2:
            max_speed = 25.0  # Maximum pixels per frame
            current_pos = positions[-1]
            movement = combined_pred - current_pos
            speed = np.linalg.norm(movement)
            
            if speed > max_speed:
                movement = movement / speed * max_speed
                combined_pred = current_pos + movement
        
        return combined_pred
    
    def get_memory_mask(self, track_id: int) -> Optional[np.ndarray]:
        """Get weighted memory mask with quality considerations"""
        if track_id not in self.mask_memory or len(self.mask_memory[track_id]) == 0:
            return None
        
        masks = list(self.mask_memory[track_id])
        confidences = list(self.confidence_memory[track_id])
        
        if len(confidences) == len(masks) and len(masks) > 0:
            # Quality-weighted average
            weights = np.array(confidences)
            
            # Boost recent masks
            recency_weights = np.linspace(0.5, 1.0, len(weights))
            weights = weights * recency_weights
            
            # Normalize weights
            weights = weights / (np.sum(weights) + 1e-8)
            
            weighted_mask = np.zeros_like(masks[0], dtype=np.float32)
            for mask, weight in zip(masks, weights):
                weighted_mask += mask.astype(np.float32) * weight
            
            return (weighted_mask > 0.4).astype(bool)  # Lower threshold for better recall
        else:
            # Simple average
            if len(masks) > 0:
                avg_mask = np.mean([mask.astype(np.float32) for mask in masks], axis=0)
                return (avg_mask > 0.4).astype(bool)
        
        return None
    
    def get_track_reliability(self, track_id: int) -> float:
        """Get reliability score for a track"""
        if track_id not in self.track_quality:
            return 0.0
        
        quality = self.track_quality[track_id]
        
        # Consistency score
        if track_id in self.track_consistency and len(self.track_consistency[track_id]) > 2:
            consistency = 1.0 - min(1.0, np.std(self.track_consistency[track_id]))
        else:
            consistency = 0.5
        
        # Age factor (older tracks are more reliable)
        age_factor = min(1.0, len(self.position_memory.get(track_id, [])) / 10.0)
        
        return quality * 0.5 + consistency * 0.3 + age_factor * 0.2


class ReIDPipeline:
    """Enhanced ReID Pipeline with SAMURAI motion-aware tracking"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # Initialize SAM
        self.sam_predictor = None
        self.sam_model_type = None
        self.setup_sam_model()
        
        # Initialize depth estimation
        self.depth_pipeline = None
        self.setup_depth_anything()
        
        # Initialize MegaDescriptor (simplified)
        self.reid_model = None
        self.reid_method = None
        self.setup_megadescriptor()
        
        # Enhanced motion-aware memory
        memory_size = getattr(config, 'samurai_memory_size', 15)
        motion_weight = getattr(config, 'samurai_motion_weight', 0.3)
        
        self.memory = MotionAwareMemory(
            memory_size=memory_size,
            motion_weight=motion_weight
        )
        
        # Enhanced tracking parameters  
        self.similarity_threshold = getattr(config, 'reid_similarity_threshold', 0.3)
        self.samurai_similarity_threshold = getattr(config, 'samurai_similarity_threshold', 0.4)
        self.max_lost_frames = getattr(config, 'samurai_max_lost_frames', 10)
        self.motion_distance_threshold = getattr(config, 'motion_distance_threshold', 100)
        self.visual_recovery_threshold = getattr(config, 'visual_recovery_threshold', 0.3)
        
        # Track management
        self.max_tracks_per_frame = getattr(config, 'max_tracks_per_frame', 20)
        self.track_consolidation_enabled = getattr(config, 'track_consolidation_enabled', False)
        self.consolidation_threshold = getattr(config, 'consolidation_similarity_threshold', 0.7)
        
        # Performance settings
        self.acceleration_prediction = getattr(config, 'acceleration_prediction', True)
        self.visual_memory_enabled = getattr(config, 'visual_memory_enabled', True)
        
        # Lost track recovery
        self.lost_tracks = {}
        self.reassignment_count = 0
        self.frame_count = 0
        
        # Store current frame's data
        self.current_masks = []
        self._last_depth_map = None
        
        print(f"🎯 Enhanced SAMURAI ReID Pipeline initialized")
        print(f"   SAM model: {self.sam_model_type or 'disabled'}")
        print(f"   Memory size: {memory_size} frames")
        print(f"   Similarity threshold: {self.similarity_threshold}")
        print(f"   Motion threshold: {self.motion_distance_threshold} pixels")
        print(f"   Track consolidation: {self.track_consolidation_enabled}")
    
    def setup_sam_model(self):
        """Initialize SAM model based on config"""
        if not hasattr(self.config, 'sam_model') or self.config.sam_model == 'none':
            print("🚫 SAM segmentation disabled")
            return
        
        if self.config.sam_model == 'sam2':
            self.setup_sam2()
        elif self.config.sam_model == 'mobilesam':
            self.setup_mobile_sam()
    
    def setup_sam2(self):
        """Initialize SAM2"""
        if not SAM2_AVAILABLE:
            print("❌ SAM2 not available")
            return
        try:
            self.sam_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-base-plus")
            self.sam_model_type = 'sam2'
            print("✅ SAM2 loaded for SAMURAI tracking")
        except Exception as e:
            print(f"❌ SAM2 setup failed: {e}")
    
    def setup_mobile_sam(self):
        """Initialize MobileSAM"""
        if not MOBILESAM_AVAILABLE:
            print("❌ MobileSAM not available")
            return
        try:
            checkpoint_paths = ["checkpoints/mobile_sam.pt", "mobile_sam.pt"]
            sam_checkpoint = None
            
            for path in checkpoint_paths:
                if os.path.exists(path):
                    sam_checkpoint = path
                    break
            
            if sam_checkpoint is None:
                os.makedirs("checkpoints", exist_ok=True)
                import urllib.request
                url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
                sam_checkpoint = "checkpoints/mobile_sam.pt"
                urllib.request.urlretrieve(url, sam_checkpoint)
            
            sam = sam_model_registry["vit_t"](checkpoint=sam_checkpoint)
            sam.to(device=self.device)
            self.sam_predictor = SamPredictor(sam)
            self.sam_model_type = 'mobilesam'
            print("✅ MobileSAM loaded for SAMURAI tracking")
        except Exception as e:
            print(f"❌ MobileSAM setup failed: {e}")
    
    def setup_depth_anything(self):
        """Initialize Depth Anything"""
        if not DEPTH_ANYTHING_AVAILABLE:
            print("❌ Depth Anything not available")
            return
        
        enable_depth = getattr(self.config, 'enable_depth_anything', True)
        if not enable_depth:
            print("🚫 Depth Anything disabled in config")
            return
            
        try:
            self.depth_pipeline = pipeline(
                task="depth-estimation", 
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=0 if self.device == "cuda" else -1
            )
            print("✅ Depth Anything loaded")
        except Exception as e:
            print(f"❌ Depth Anything setup failed: {e}")
    
    def setup_megadescriptor(self):
        """Initialize simplified MegaDescriptor"""
        if not MEGADESCRIPTOR_AVAILABLE:
            print("❌ MegaDescriptor not available")
            return
        
        enable_megadescriptor = getattr(self.config, 'enable_megadescriptor', True)
        if not enable_megadescriptor:
            print("🚫 MegaDescriptor disabled in config")
            return
        
        try:
            # Enhanced feature extractor
            class EnhancedFeatureExtractor(torch.nn.Module):
                def __init__(self, device):
                    super().__init__()
                    self.device = device
                    
                def forward(self, x):
                    # Multi-scale feature extraction
                    features = []
                    
                    # Global features
                    global_feat = torch.nn.functional.adaptive_avg_pool2d(x, (8, 8))
                    features.append(torch.flatten(global_feat, 1))
                    
                    # Local features
                    local_feat = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
                    features.append(torch.flatten(local_feat, 1))
                    
                    # Combine features
                    combined = torch.cat(features, dim=1)
                    return torch.nn.functional.normalize(combined, p=2, dim=1)
            
            self.reid_model = EnhancedFeatureExtractor(self.device)
            self.reid_model.to(self.device)
            self.reid_model.eval()
            self.reid_method = "enhanced_features"
            
            print("✅ Enhanced feature extractor loaded")
                
        except Exception as e:
            print(f"❌ Feature extractor setup failed: {e}")
    
    def estimate_depth_full_image(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth map for the entire image"""
        if not self.depth_pipeline:
            return np.zeros_like(frame[:,:,0])
        
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            depth_result = self.depth_pipeline(pil_image)
            depth_array = np.array(depth_result['depth'])
            
            h, w = frame.shape[:2]
            depth_resized = cv2.resize(depth_array, (w, h))
            
            if depth_resized.max() > depth_resized.min():
                depth_normalized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min()) * 255
            else:
                depth_normalized = np.zeros_like(depth_resized)
            
            return depth_normalized.astype(np.uint8)
        except Exception as e:
            print(f"❌ Depth estimation failed: {e}")
            return np.zeros_like(frame[:,:,0])
    
    def segment_with_motion_prediction(self, frame: np.ndarray, track_id: int, 
                                     detection_bbox: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """Enhanced segmentation using motion prediction and memory"""
        if not self.sam_predictor:
            h, w = frame.shape[:2]
            return np.zeros((h, w), dtype=bool), 0.0
        
        try:
            self.sam_predictor.set_image(frame)
            
            # Get predicted position with enhanced prediction
            predicted_pos = self.memory.predict_next_position(track_id)
            
            input_points = []
            input_labels = []
            
            if predicted_pos is not None:
                input_points.append(predicted_pos)
                input_labels.append(1)
            
            # Add detection center as additional point
            if detection_bbox is not None:
                center_x = int((detection_bbox[0] + detection_bbox[2]) / 2)
                center_y = int((detection_bbox[1] + detection_bbox[3]) / 2)
                input_points.append([center_x, center_y])
                input_labels.append(1)
            
            if not input_points:
                h, w = frame.shape[:2]
                return np.zeros((h, w), dtype=bool), 0.0
            
            input_points = np.array(input_points)
            input_labels = np.array(input_labels)
            input_box = detection_bbox if detection_bbox is not None else None
            
            # Generate masks
            if self.sam_model_type == 'sam2':
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    masks, scores, logits = self.sam_predictor.predict(
                        point_coords=input_points,
                        point_labels=input_labels,
                        box=input_box,
                        multimask_output=True,
                    )
            else:
                masks, scores, logits = self.sam_predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    box=input_box,
                    multimask_output=True,
                )
            
            best_mask = masks[np.argmax(scores)]
            best_score = np.max(scores)
            
            # Ensure mask is boolean
            if best_mask.dtype != bool:
                best_mask = best_mask.astype(bool)
            
            # Refine with memory if available
            if self.visual_memory_enabled:
                memory_mask = self.memory.get_memory_mask(track_id)
                if memory_mask is not None:
                    refined_mask = self.refine_mask_with_memory(best_mask, memory_mask)
                    return refined_mask, best_score
            
            return best_mask, best_score
                
        except Exception as e:
            print(f"❌ SAM segmentation failed: {e}")
            h, w = frame.shape[:2]
            return np.zeros((h, w), dtype=bool), 0.0
    
    def refine_mask_with_memory(self, current_mask: np.ndarray, memory_mask: np.ndarray) -> np.ndarray:
        """Enhanced mask refinement using memory template"""
        # Ensure both masks are boolean
        current_mask = current_mask.astype(bool)
        memory_mask = memory_mask.astype(bool)
        
        intersection = np.logical_and(current_mask, memory_mask)
        union = np.logical_or(current_mask, memory_mask)
        
        if np.sum(union) == 0:
            return current_mask
        
        iou = np.sum(intersection) / np.sum(union)
        
        if iou > 0.2:  # Lower threshold for more aggressive refinement
            # Combine masks with weighted fusion
            dilated_current = cv2.dilate(current_mask.astype(np.uint8), np.ones((3,3)), iterations=1).astype(bool)
            dilated_memory = cv2.dilate(memory_mask.astype(np.uint8), np.ones((3,3)), iterations=1).astype(bool)
            
            # Weighted combination
            refined = np.logical_or(
                np.logical_and(current_mask, memory_mask),  # High confidence: intersection
                np.logical_and(dilated_current, memory_mask),  # Medium confidence: memory-guided expansion
                np.logical_and(current_mask, dilated_memory)   # Medium confidence: current-guided expansion
            )
            return refined
        else:
            return current_mask
    
    def extract_enhanced_features(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extract enhanced multi-scale visual features"""
        try:
            # Ensure mask is boolean
            mask = mask.astype(bool)
            
            if not np.any(mask):
                return np.random.rand(150) * 0.01  # Small random features for empty masks
            
            features = []
            
            # 1. Color features (enhanced)
            masked_region = frame.copy()
            masked_region[~mask] = 0
            
            # RGB histograms with more bins
            for channel in range(3):
                hist = cv2.calcHist([masked_region], [channel], mask.astype(np.uint8), [16], [0, 256])
                features.extend(hist.flatten())
            
            # HSV color features
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            for channel in range(3):
                masked_hsv = hsv[:,:,channel][mask]
                if len(masked_hsv) > 0:
                    features.extend([
                        np.mean(masked_hsv),
                        np.std(masked_hsv),
                        np.percentile(masked_hsv, 25),
                        np.percentile(masked_hsv, 75)
                    ])
                else:
                    features.extend([0, 0, 0, 0])
            
            # 2. Texture features (enhanced)
            gray = cv2.cvtColor(masked_region, cv2.COLOR_BGR2GRAY)
            masked_gray = gray * mask.astype(np.uint8)
            
            if np.any(mask):
                # Gradient features
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                
                features.extend([
                    np.mean(grad_x[mask]),
                    np.std(grad_x[mask]),
                    np.mean(grad_y[mask]),
                    np.std(grad_y[mask]),
                    np.mean(np.sqrt(grad_x[mask]**2 + grad_y[mask]**2))  # Gradient magnitude
                ])
                
                # Local Binary Pattern-like features
                for kernel_size in [3, 5]:
                    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
                    smoothed = cv2.filter2D(masked_gray.astype(np.float32), -1, kernel)
                    variance = np.var(smoothed[mask]) if np.any(mask) else 0
                    features.append(variance)
            else:
                features.extend([0] * 7)
            
            # 3. Shape features (enhanced)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Largest contour
                contour = max(contours, key=cv2.contourArea)
                
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    circularity = 0
                
                # Bounding box features
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                extent = area / (w * h) if w * h > 0 else 0
                
                # Moments
                moments = cv2.moments(contour)
                if moments['m00'] != 0:
                    # Hu moments (scale, rotation, translation invariant)
                    hu_moments = cv2.HuMoments(moments).flatten()
                    # Take first 4 Hu moments and log transform
                    hu_features = [-np.sign(hu) * np.log10(np.abs(hu) + 1e-10) for hu in hu_moments[:4]]
                else:
                    hu_features = [0, 0, 0, 0]
                
                shape_features = [
                    area / 10000,  # Normalized area
                    perimeter / 1000,  # Normalized perimeter  
                    circularity,
                    aspect_ratio,
                    extent
                ] + hu_features
                
                features.extend(shape_features)
            else:
                features.extend([0] * 9)
            
            # 4. Position features (relative to frame)
            h, w = frame.shape[:2]
            coords = np.where(mask)
            if len(coords[0]) > 0:
                y_center = np.mean(coords[0]) / h
                x_center = np.mean(coords[1]) / w
                y_extent = (np.max(coords[0]) - np.min(coords[0])) / h
                x_extent = (np.max(coords[1]) - np.min(coords[1])) / w
                features.extend([x_center, y_center, x_extent, y_extent])
            else:
                features.extend([0, 0, 0, 0])
            
            # Convert to numpy array and normalize
            features = np.array(features, dtype=np.float32)
            
            # Handle any NaN or infinite values
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Normalize features
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return features
            
        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
            return np.random.rand(150) * 0.01
    
    def match_with_memory_improved(self, features: np.ndarray, track_id: int) -> float:
        """Enhanced memory matching with multiple similarity metrics"""
        
        if track_id not in self.memory.feature_memory:
            return 0.0
        
        memory_features = list(self.memory.feature_memory[track_id])
        confidences = list(self.memory.confidence_memory[track_id])
        
        if len(memory_features) == 0:
            return 0.0
        
        # Calculate multiple similarity metrics
        similarities = []
        
        for mem_feat, conf in zip(memory_features, confidences):
            # 1. Cosine similarity
            cos_sim = np.dot(features, mem_feat) / (np.linalg.norm(features) * np.linalg.norm(mem_feat) + 1e-8)
            
            # 2. L2 distance similarity  
            l2_dist = np.linalg.norm(features - mem_feat)
            l2_sim = 1.0 / (1.0 + l2_dist)
            
            # 3. Correlation similarity
            if len(features) == len(mem_feat):
                try:
                    corr_sim = np.corrcoef(features, mem_feat)[0, 1]
                    if np.isnan(corr_sim):
                        corr_sim = 0.0
                except:
                    corr_sim = 0.0
            else:
                corr_sim = 0.0
            
            # Combine similarities
            combined_sim = 0.5 * cos_sim + 0.3 * l2_sim + 0.2 * corr_sim
            
            # Weight by confidence and recency
            weight = max(0.3, conf)  # Minimum weight of 0.3
            weighted_sim = combined_sim * weight
            
            similarities.append(weighted_sim)
        
        # Use maximum similarity (most optimistic matching)
        max_similarity = max(similarities) if similarities else 0.0
        
        # Add track reliability bonus
        reliability = self.memory.get_track_reliability(track_id)
        final_similarity = max_similarity * (0.8 + 0.2 * reliability)
        
        return final_similarity
    
    def find_best_match_aggressive(self, query_feature: np.ndarray) -> int:
        """Enhanced aggressive matching with adaptive thresholds"""
        
        if len(self.memory.feature_memory) == 0:
            print(f"DEBUG: No memory tracks to match against")
            return -1
    
        print(f"DEBUG: Matching against {len(self.memory.feature_memory)} memory tracks")

        # Use adaptive threshold based on frame count and track count
        base_threshold = self.similarity_threshold
        
        # Lower threshold if we have many tracks (more aggressive)
        if len(self.memory.feature_memory) > 10:
            adaptive_threshold = base_threshold * 0.8
        else:
            adaptive_threshold = base_threshold
        
        print(f"   🔍 Checking against {len(self.memory.feature_memory)} tracks (threshold: {adaptive_threshold:.3f})")
        
        track_scores = []
        
        for track_id in self.memory.feature_memory:
            similarity = self.match_with_memory_improved(query_feature, track_id)
            print(f"DEBUG: Track {track_id} similarity: {similarity:.3f} (threshold: {adaptive_threshold:.3f})")
            track_scores.append((track_id, similarity))
            
            if similarity > best_similarity and similarity > adaptive_threshold:
                best_similarity = similarity
                best_track_id = track_id
        
        # Show top 3 matches for debugging
        track_scores.sort(key=lambda x: x[1], reverse=True)
        for i, (tid, sim) in enumerate(track_scores[:3]):
            print(f"     #{i+1}: Track {tid} = {sim:.3f}")
        
        if best_track_id >= 0:
            print(f"   🎯 SELECTED: Track #{best_track_id} (similarity: {best_similarity:.3f})")
        else:
            print(f"   ❌ No match found (best: {best_similarity:.3f})")
        
        return best_track_id
    
    def calculate_visual_similarity(self, feature: np.ndarray, track_id: int) -> float:
        """Calculate visual similarity for recovery validation"""
        
        if track_id not in self.memory.feature_memory:
            return 0.0
        
        # Get recent features (last 5 frames)
        recent_features = list(self.memory.feature_memory[track_id])[-5:]
        
        if len(recent_features) == 0:
            return 0.0
        
        similarities = []
        for mem_feat in recent_features:
            sim = np.dot(feature, mem_feat) / (np.linalg.norm(feature) * np.linalg.norm(mem_feat) + 1e-8)
            similarities.append(sim)
        
        return max(similarities)  # Return best similarity
    
    def consolidate_similar_tracks(self, detections, reid_features):
        """Consolidate tracks that are likely the same object"""
        
        if not self.track_consolidation_enabled:
            return
        
        if not hasattr(detections, 'tracker_id') or len(detections) < 2:
            return
        
        # Get unique active track IDs
        active_tracks = {}
        for i, track_id in enumerate(detections.tracker_id):
            if track_id >= 0:
                if track_id not in active_tracks:
                    active_tracks[track_id] = []
                active_tracks[track_id].append(i)
        
        if len(active_tracks) <= self.max_tracks_per_frame:
            return  # Not too many tracks
        
        print(f"🔧 Consolidating {len(active_tracks)} tracks...")
        
        # Find pairs of similar tracks
        track_ids = list(active_tracks.keys())
        consolidations = 0
        
        for i in range(len(track_ids)):
            for j in range(i+1, len(track_ids)):
                track_a, track_b = track_ids[i], track_ids[j]
                
                if track_a not in active_tracks or track_b not in active_tracks:
                    continue  # Already consolidated
                
                # Get representative features and positions
                idx_a = active_tracks[track_a][0]
                idx_b = active_tracks[track_b][0]
                
                feat_a = reid_features[idx_a]
                feat_b = reid_features[idx_b]
                
                # Visual similarity
                visual_sim = np.dot(feat_a, feat_b) / (np.linalg.norm(feat_a) * np.linalg.norm(feat_b) + 1e-8)
                
                # Position similarity
                bbox_a = detections.xyxy[idx_a]
                bbox_b = detections.xyxy[idx_b]
                
                center_a = np.array([(bbox_a[0] + bbox_a[2]) / 2, (bbox_a[1] + bbox_a[3]) / 2])
                center_b = np.array([(bbox_b[0] + bbox_b[2]) / 2, (bbox_b[1] + bbox_b[3]) / 2])
                
                distance = np.linalg.norm(center_a - center_b)
                
                # Consolidate if very similar and close
                if visual_sim > self.consolidation_threshold and distance < 80:
                    # Merge track_b into track_a (keep lower ID)
                    target_id = min(track_a, track_b)
                    source_id = max(track_a, track_b)
                    
                    # Update all detections with source_id to target_id
                    for idx in active_tracks[source_id]:
                        detections.tracker_id[idx] = target_id
                    
                    # Merge memories
                    self.merge_track_memories(source_id, target_id)
                    
                    # Update active_tracks
                    if target_id not in active_tracks:
                        active_tracks[target_id] = []
                    active_tracks[target_id].extend(active_tracks[source_id])
                    del active_tracks[source_id]
                    
                    consolidations += 1
                    print(f"🔄 Consolidated Track #{source_id} → Track #{target_id} (sim: {visual_sim:.3f}, dist: {distance:.1f})")
        
        if consolidations > 0:
            print(f"✅ Made {consolidations} consolidations")
    
    def merge_track_memories(self, source_id: int, target_id: int):
        """Merge memory from source track into target track"""
        
        if source_id not in self.memory.feature_memory:
            return
        
        # Merge feature memories (keep most recent features)
        if target_id not in self.memory.feature_memory:
            self.memory.feature_memory[target_id] = deque()
        
        # Merge memories
        for memory_bank_name in ['feature_memory', 'confidence_memory', 'position_memory', 
                                'bbox_memory', 'mask_memory']:
            source_memory = getattr(self.memory, memory_bank_name).get(source_id, deque())
            target_memory = getattr(self.memory, memory_bank_name).get(target_id, deque())
            
            # Merge and keep most recent entries
            combined = list(target_memory) + list(source_memory)
            combined = combined[-self.memory.memory_size:]  # Keep last N entries
            
            getattr(self.memory, memory_bank_name)[target_id] = deque(combined)
            
            # Remove source
            if source_id in getattr(self.memory, memory_bank_name):
                del getattr(self.memory, memory_bank_name)[source_id]
    
    def enhance_tracking(self, detections, reid_features, depth_stats=None):
        """ENHANCED aggressive tracking enhancement"""
        
        if not sv or len(detections) == 0 or len(reid_features) == 0:
            return detections
        
        self.frame_count += 1
        enhanced_detections = detections
        
        if hasattr(detections, 'tracker_id'):
            print(f"🔍 Frame {self.frame_count}: {len(detections)} detections, {np.sum(detections.tracker_id < 0)} untracked")
            print(f"🔍 Memory: {len(self.memory.feature_memory)} tracks")
            
            # Find untracked detections
            untracked_indices = np.where(enhanced_detections.tracker_id < 0)[0]
            recoveries_this_frame = 0
            
            if len(untracked_indices) > 0:
                print(f"🔄 Attempting aggressive recovery for {len(untracked_indices)} untracked detections...")
                
                for i in untracked_indices:
                    feature = reid_features[i]
                    
                    # Try aggressive matching
                    best_match = self.find_best_match_aggressive(feature)
                    
                    if best_match >= 0:
                        # Validate with motion and visual checks
                        bbox = detections.xyxy[i]
                        center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                        
                        predicted_pos = self.memory.predict_next_position(best_match)
                        
                        recovery_success = False
                        recovery_reason = ""
                        
                        if predicted_pos is not None:
                            distance = np.linalg.norm(center - predicted_pos)
                            if distance < self.motion_distance_threshold:
                                recovery_success = True
                                recovery_reason = f"motion (dist: {distance:.1f})"
                        
                        # Try pure visual recovery if motion fails
                        if not recovery_success:
                            visual_sim = self.calculate_visual_similarity(feature, best_match)
                            if visual_sim > self.visual_recovery_threshold:
                                recovery_success = True
                                recovery_reason = f"visual (sim: {visual_sim:.3f})"
                        
                        if recovery_success:
                            enhanced_detections.tracker_id[i] = best_match
                            self.reassignment_count += 1
                            recoveries_this_frame += 1
                            print(f"✅ RECOVERED detection #{i} → Track #{best_match} ({recovery_reason})")
            
            # Consolidate similar tracks if enabled
            if self.track_consolidation_enabled:
                self.consolidate_similar_tracks(enhanced_detections, reid_features)
            
            # Update lost track counters
            self.update_lost_track_counters(enhanced_detections)
            
            # print(f"📊 Frame {self.frame_count}: {recoveries_this_frame} recoveries, {self.reassignment_count} total")
        
        return enhanced_detections
    
    def update_lost_track_counters(self, detections):
        """Enhanced lost track counter management"""
        
        active_tracks = set(detections.tracker_id[detections.tracker_id >= 0])
        all_memory_tracks = set(self.memory.position_memory.keys())
        
        # Update lost track counters
        for track_id in all_memory_tracks:
            if track_id not in active_tracks:
                self.lost_tracks[track_id] = self.lost_tracks.get(track_id, 0) + 1
                
                # Clean up very old tracks
                if self.lost_tracks[track_id] > self.max_lost_frames:
                    self.cleanup_track(track_id)
            else:
                # Reset counter for active tracks
                self.lost_tracks.pop(track_id, None)
    
    def cleanup_track(self, track_id):
        """Clean up an old lost track"""
        
        self.lost_tracks.pop(track_id, None)
        
        # Remove from all memory banks
        for memory_bank in [self.memory.mask_memory, self.memory.bbox_memory, 
                           self.memory.feature_memory, self.memory.position_memory,
                           self.memory.confidence_memory, self.memory.velocity_memory,
                           self.memory.acceleration_memory]:
            memory_bank.pop(track_id, None)
        
        # Clean up track quality metrics
        self.memory.track_quality.pop(track_id, None)
        self.memory.track_consistency.pop(track_id, None)
        
        print(f"🗑️ Cleaned up old track #{track_id}")
    
    def process_frame(self, frame: np.ndarray, detections) -> Tuple:
        """Enhanced frame processing with SAMURAI"""
        # Estimate depth
        depth_map = self.estimate_depth_full_image(frame)
        self._last_depth_map = depth_map
        
        # Process each detection with enhanced motion-aware SAM
        rgb_crops, depth_crops = [], []
        features = []
        depth_stats = []
        self.current_masks = []
        
        if not sv or len(detections) == 0:
            return rgb_crops, depth_crops, depth_map, np.array([]), depth_stats
        
        for i, bbox in enumerate(detections.xyxy):
            track_id = detections.tracker_id[i] if hasattr(detections, 'tracker_id') and i < len(detections.tracker_id) else -1
            
            # Enhanced segmentation with motion prediction
            mask, confidence = self.segment_with_motion_prediction(frame, track_id, bbox)
            self.current_masks.append(mask)
            
            # Extract crops
            x1, y1, x2, y2 = bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                rgb_crop = frame[y1:y2, x1:x2].copy()
                depth_crop = depth_map[y1:y2, x1:x2].copy()
                
                # Apply mask to crops
                mask_crop = mask[y1:y2, x1:x2]
                if mask_crop.size > 0 and mask_crop.shape == rgb_crop.shape[:2]:
                    # Ensure mask_crop is boolean
                    mask_crop = mask_crop.astype(bool)
                    # Apply mask
                    rgb_crop[~mask_crop] = [240, 240, 240]
                    depth_crop[~mask_crop] = 0
                
                rgb_crops.append(rgb_crop)
                depth_crops.append(depth_crop)
                
                # Extract enhanced features
                feature = self.extract_enhanced_features(frame, mask)
                features.append(feature)
                
                # Enhanced depth stats
                depth_stats.append({
                    'area': np.sum(mask), 
                    'confidence': confidence,
                    'mask_quality': np.mean(mask.astype(float)),
                    'depth_variance': np.var(depth_crop[mask_crop]) if np.any(mask_crop) else 0
                })
                
                # Update memory if valid track
                if track_id >= 0:
                    self.memory.update_track_memory(track_id, mask, bbox, feature, confidence)
        
        return rgb_crops, depth_crops, depth_map, np.array(features), depth_stats
    
    def get_current_masks(self):
        """Get current frame's segmentation masks"""
        return self.current_masks
    
    def get_reassignment_count(self):
        """Get total reassignments"""
        return self.reassignment_count
    
    def get_tracking_info(self):
        """Get enhanced tracking information"""
        active_tracks = len(self.memory.position_memory)
        lost_tracks = len(self.lost_tracks)
        
        # Get motion predictions for active tracks
        motion_predictions = {}
        track_reliabilities = {}
        
        for track_id in self.memory.position_memory.keys():
            pred_pos = self.memory.predict_next_position(track_id)
            if pred_pos is not None:
                motion_predictions[track_id] = pred_pos
            
            reliability = self.memory.get_track_reliability(track_id)
            track_reliabilities[track_id] = reliability
        
        return {
            'active_tracks': active_tracks,
            'lost_tracks': lost_tracks,
            'motion_predictions': motion_predictions,
            'track_reliabilities': track_reliabilities,
            'total_reassignments': self.reassignment_count,
            'frame_count': self.frame_count,
            'memory_efficiency': len(self.memory.feature_memory),
            'consolidation_enabled': self.track_consolidation_enabled
        }