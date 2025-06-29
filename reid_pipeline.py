"""
Enhanced SAMURAI ReID Pipeline - MegaDescriptor RGB+Depth Re-identification
Uses object detection → SAM segmentation → MegaDescriptor embeddings for track assignment
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
    print("✓ Depth-Anything (transformers) available")
except ImportError:
    DEPTH_ANYTHING_AVAILABLE = False
    print("⚠️ Depth-Anything not available - install: pip install transformers")

try:
    from transformers import AutoModel, AutoProcessor
    MEGADESCRIPTOR_AVAILABLE = True
    print("✓ MegaDescriptor (transformers) available")
except ImportError:
    MEGADESCRIPTOR_AVAILABLE = False
    print("⚠️ MegaDescriptor not available - install: pip install transformers>=4.35.0")


class TrackStabilityManager:
    """Manage track assignment stability and prevent oscillation"""
    
    def __init__(self, cooling_period=10, oscillation_threshold=3):
        self.cooling_period = cooling_period
        self.oscillation_threshold = oscillation_threshold
        
        self.assignment_history = defaultdict(deque)
        self.cooling_until = defaultdict(int)
        self.oscillation_count = defaultdict(int)
        self.last_target_ids = defaultdict(deque)
        self.assignment_confidence = defaultdict(float)
        self.assignment_frame = defaultdict(int)
        
        self.current_frame = 0
        
    def is_track_locked(self, track_id: int) -> bool:
        return self.current_frame < self.cooling_until[track_id]
        
    def record_assignment(self, track_id: int, target_id: int, confidence: float) -> bool:
        if target_id in self.last_target_ids[track_id]:
            self.oscillation_count[track_id] += 1
        else:
            self.oscillation_count[track_id] = max(0, self.oscillation_count[track_id] - 1)
            
        self.assignment_confidence[track_id] = confidence
        self.assignment_frame[track_id] = self.current_frame
        self.last_target_ids[track_id].append(target_id)
        
        if len(self.last_target_ids[track_id]) > 5:
            self.last_target_ids[track_id].popleft()
            
        cooling_multiplier = 1 + self.oscillation_count[track_id]
        self.cooling_until[track_id] = self.current_frame + (self.cooling_period * cooling_multiplier)
        
        self.assignment_history[track_id].append({
            'frame': self.current_frame,
            'target_id': target_id,
            'confidence': confidence
        })
        if len(self.assignment_history[track_id]) > 10:
            self.assignment_history[track_id].popleft()
            
        return self.oscillation_count[track_id] > self.oscillation_threshold
        
    def get_stability_penalty(self, track_id: int) -> float:
        oscillation_penalty = min(0.5, self.oscillation_count[track_id] * 0.1)
        frames_since_assignment = self.current_frame - self.assignment_frame[track_id]
        recency_penalty = max(0, 0.3 - frames_since_assignment * 0.03)
        return oscillation_penalty + recency_penalty
        
    def get_stability_bonus(self, track_id: int) -> float:
        frames_stable = self.current_frame - self.assignment_frame[track_id]
        return min(0.2, frames_stable * 0.02)
        
    def advance_frame(self):
        self.current_frame += 1
        
    def cleanup_old_tracks(self, active_tracks: set):
        all_tracked = set(self.assignment_history.keys())
        inactive = all_tracked - active_tracks
        
        for track_id in inactive:
            self.assignment_history.pop(track_id, None)
            self.cooling_until.pop(track_id, None)
            self.oscillation_count.pop(track_id, None)
            self.last_target_ids.pop(track_id, None)
            self.assignment_confidence.pop(track_id, None)
            self.assignment_frame.pop(track_id, None)


class TrackQualityMonitor:
    """Enhanced track quality monitoring with temporal consistency"""
    
    def __init__(self, stability_window=10):
        self.stability_window = stability_window
        
        self.confidence_history = defaultdict(deque)
        self.position_variance = defaultdict(list)
        self.track_age = defaultdict(int)
        self.track_first_seen = {}
        
        self.stability_scores = defaultdict(float)
        self.last_update = defaultdict(int)
    
    def update_track_quality(self, track_id: int, confidence: float, position: np.ndarray, frame_num: int):
        if track_id not in self.track_first_seen:
            self.track_first_seen[track_id] = frame_num
        
        self.track_age[track_id] += 1
        self.last_update[track_id] = frame_num
        
        self.confidence_history[track_id].append(confidence)
        if len(self.confidence_history[track_id]) > self.stability_window:
            self.confidence_history[track_id].popleft()
        
        if track_id in self.position_variance:
            if len(self.position_variance[track_id]) >= 3:
                recent_positions = self.position_variance[track_id][-3:]
                variance = np.var([np.linalg.norm(p - position) for p in recent_positions])
                self.position_variance[track_id].append(position)
            else:
                self.position_variance[track_id].append(position)
                variance = 0.0
        else:
            self.position_variance[track_id] = [position]
            variance = 0.0
        
        if len(self.position_variance[track_id]) > self.stability_window:
            self.position_variance[track_id].pop(0)
        
        self._calculate_stability_score(track_id, variance)
    
    def _calculate_stability_score(self, track_id: int, position_variance: float):
        confidences = list(self.confidence_history[track_id])
        if len(confidences) > 1:
            conf_mean = np.mean(confidences)
            conf_stability = 1.0 - min(1.0, np.std(confidences))
        else:
            conf_mean = confidences[0] if confidences else 0.5
            conf_stability = 0.5
        
        motion_stability = 1.0 / (1.0 + position_variance / 100.0)
        age_factor = min(1.0, self.track_age[track_id] / 20.0)
        
        stability = conf_mean * 0.4 + conf_stability * 0.3 + motion_stability * 0.2 + age_factor * 0.1
        self.stability_scores[track_id] = stability
    
    def get_track_stability(self, track_id: int) -> float:
        return self.stability_scores.get(track_id, 0.0)
    
    def is_new_track(self, track_id: int, frame_num: int, newness_threshold: int = 5) -> bool:
        if track_id not in self.track_first_seen:
            return True
        return (frame_num - self.track_first_seen[track_id]) <= newness_threshold
    
    def is_unstable_track(self, track_id: int, stability_threshold: float = 0.4) -> bool:
        return self.get_track_stability(track_id) < stability_threshold
    
    def cleanup_old_tracks(self, active_track_ids: set, frame_num: int, max_age: int = 50):
        to_remove = []
        for track_id in list(self.last_update.keys()):
            if track_id not in active_track_ids:
                if frame_num - self.last_update[track_id] > max_age:
                    to_remove.append(track_id)
        
        for track_id in to_remove:
            self._remove_track(track_id)
    
    def _remove_track(self, track_id: int):
        self.confidence_history.pop(track_id, None)
        self.position_variance.pop(track_id, None)
        self.track_age.pop(track_id, None)
        self.track_first_seen.pop(track_id, None)
        self.stability_scores.pop(track_id, None)
        self.last_update.pop(track_id, None)


class MotionAwareMemory:
    """Enhanced memory with motion prediction"""
    
    def __init__(self, memory_size=15):
        self.memory_size = memory_size
        
        self.feature_memory = defaultdict(deque)
        self.position_memory = defaultdict(deque)
        self.bbox_memory = defaultdict(deque)
        self.mask_memory = defaultdict(deque)
        self.confidence_memory = defaultdict(deque)
    
    def update_track_memory(self, track_id: int, feature: np.ndarray, position: np.ndarray, 
                           bbox: np.ndarray, mask: np.ndarray, confidence: float):
        self.feature_memory[track_id].append(feature)
        self.position_memory[track_id].append(position)
        self.bbox_memory[track_id].append(bbox)
        self.mask_memory[track_id].append(mask)
        self.confidence_memory[track_id].append(confidence)
        
        for memory_bank in [self.feature_memory, self.position_memory, self.bbox_memory,
                           self.mask_memory, self.confidence_memory]:
            if len(memory_bank[track_id]) > self.memory_size:
                memory_bank[track_id].popleft()
    
    def predict_next_position(self, track_id: int) -> Optional[np.ndarray]:
        if track_id not in self.position_memory or len(self.position_memory[track_id]) < 2:
            return None
        
        positions = list(self.position_memory[track_id])
        
        if len(positions) >= 2:
            velocity = positions[-1] - positions[-2]
            max_speed = 30.0
            speed = np.linalg.norm(velocity)
            if speed > max_speed:
                velocity = velocity / speed * max_speed
            
            return positions[-1] + velocity
        
        return positions[-1]
    
    def get_recent_features(self, track_id: int, n_recent: int = 3) -> List[np.ndarray]:
        if track_id not in self.feature_memory:
            return []
        
        features = list(self.feature_memory[track_id])
        return features[-n_recent:] if len(features) >= n_recent else features
    
    def cleanup_track(self, track_id: int):
        for memory_bank in [self.feature_memory, self.position_memory, self.bbox_memory,
                           self.mask_memory, self.confidence_memory]:
            memory_bank.pop(track_id, None)


class EnhancedReIDPipeline:
    """Enhanced ReID Pipeline with MegaDescriptor RGB+Depth embeddings"""
    
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
        
        # Initialize MegaDescriptor
        self.megadescriptor_model = None
        self.megadescriptor_processor = None
        self.setup_megadescriptor()
        
        # Core components
        self.memory = MotionAwareMemory(
            memory_size=getattr(config, 'samurai_memory_size', 15)
        )
        self.quality_monitor = TrackQualityMonitor(
            stability_window=getattr(config, 'quality_stability_window', 10)
        )
        
        # Stability management
        self.stability_manager = TrackStabilityManager(
            cooling_period=getattr(config, 'cooling_period', 10),
            oscillation_threshold=getattr(config, 'oscillation_threshold', 3)
        )
        
        # Configuration parameters
        self.similarity_threshold = getattr(config, 'reid_similarity_threshold', 0.3)
        self.initial_assignment_threshold = getattr(config, 'initial_assignment_threshold', 0.5)
        self.reassignment_threshold = getattr(config, 'reassignment_threshold', 0.7)
        
        self.motion_distance_threshold = getattr(config, 'motion_distance_threshold', 150)
        self.stability_threshold = getattr(config, 'track_stability_threshold', 0.4)
        self.newness_threshold = getattr(config, 'track_newness_threshold', 5)
        
        # Tracking stats
        self.reassignment_count = 0
        self.oscillations_prevented = 0
        self.stability_locks = 0
        self.frame_count = 0
        
        # Store current frame data
        self.current_masks = []
        self._last_depth_map = None
        
        print(f"🎯 Enhanced ReID Pipeline initialized with MegaDescriptor RGB+Depth")
        print(f"   SAM model: {self.sam_model_type or 'disabled'}")
        print(f"   MegaDescriptor: {'enabled' if self.megadescriptor_model else 'disabled'}")
        print(f"   Embedding size: 64 (32 RGB + 32 Depth)")
    
    def setup_sam_model(self):
        """Initialize SAM model"""
        if not hasattr(self.config, 'sam_model') or self.config.sam_model == 'none':
            return
        
        if self.config.sam_model == 'sam2' and SAM2_AVAILABLE:
            try:
                self.sam_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-base-plus")
                self.sam_model_type = 'sam2'
                print("✅ SAM2 loaded")
            except Exception as e:
                print(f"❌ SAM2 failed: {e}")
        
        elif self.config.sam_model == 'mobilesam' and MOBILESAM_AVAILABLE:
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
                print("✅ MobileSAM loaded")
            except Exception as e:
                print(f"❌ MobileSAM failed: {e}")
    
    def setup_depth_anything(self):
        """Initialize Depth Anything"""
        if not DEPTH_ANYTHING_AVAILABLE or not getattr(self.config, 'enable_depth_anything', True):
            return
            
        try:
            self.depth_pipeline = pipeline(
                task="depth-estimation", 
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=0 if self.device == "cuda" else -1
            )
            print("✅ Depth Anything loaded")
        except Exception as e:
            print(f"❌ Depth Anything failed: {e}")
    
    def setup_megadescriptor(self):
        """Initialize MegaDescriptor model"""
        if not MEGADESCRIPTOR_AVAILABLE:
            print("❌ MegaDescriptor not available")
            print("   Install with: pip install transformers>=4.35.0")
            return
            
        try:
            self.megadescriptor_processor = AutoProcessor.from_pretrained("BVRA/MegaDescriptor-L-384")
            self.megadescriptor_model = AutoModel.from_pretrained("BVRA/MegaDescriptor-L-384")
            self.megadescriptor_model.to(self.device)
            self.megadescriptor_model.eval()
            print("✅ MegaDescriptor-L-384 loaded")
        except Exception as e:
            print(f"❌ MegaDescriptor failed: {e}")
            print("   Ensure you have internet connection for first download")
            print("   Model size: ~1.5GB - may take time to download")
    
    def estimate_depth_full_image(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth map for entire image"""
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
    
    def segment_with_sam(self, frame: np.ndarray, bbox: np.ndarray) -> Tuple[np.ndarray, float]:
        """Segment using SAM with bbox center as prompt"""
        if not self.sam_predictor:
            h, w = frame.shape[:2]
            return np.zeros((h, w), dtype=bool), 0.0
        
        try:
            self.sam_predictor.set_image(frame)
            
            # Use bbox center as point prompt
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            
            input_points = np.array([[center_x, center_y]])
            input_labels = np.array([1])
            
            if self.sam_model_type == 'sam2':
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    masks, scores, logits = self.sam_predictor.predict(
                        point_coords=input_points,
                        point_labels=input_labels,
                        box=bbox,
                        multimask_output=True,
                    )
            else:
                masks, scores, logits = self.sam_predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    box=bbox,
                    multimask_output=True,
                )
            
            best_mask = masks[np.argmax(scores)]
            best_score = np.max(scores)
            
            return best_mask.astype(bool), best_score
            
        except Exception as e:
            print(f"❌ SAM segmentation failed: {e}")
            h, w = frame.shape[:2]
            return np.zeros((h, w), dtype=bool), 0.0
    
    def extract_megadescriptor_features(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extract MegaDescriptor features from masked image"""
        if not self.megadescriptor_model or not self.megadescriptor_processor:
            return np.random.rand(32) * 0.01
        
        try:
            mask = mask.astype(bool)
            
            if not np.any(mask):
                return np.random.rand(32) * 0.01
            
            # Apply mask - ROBUST method that avoids boolean indexing issues
            masked_image = image.copy()
            
            if len(masked_image.shape) == 3:
                # RGB image - use where instead of boolean indexing
                background_mask = np.expand_dims(~mask, axis=2)  # Shape: (H, W, 1)
                background_mask = np.repeat(background_mask, 3, axis=2)  # Shape: (H, W, 3)
                masked_image = np.where(background_mask, 255, masked_image)
            else:
                # Grayscale image
                masked_image = np.where(~mask, 255, masked_image)
            
            # Convert to PIL Image
            if len(masked_image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(masked_image.astype(np.uint8), cv2.COLOR_BGR2RGB))
            else:
                # For depth map (grayscale) - convert to 3-channel
                depth_3channel = cv2.cvtColor(masked_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                pil_image = Image.fromarray(depth_3channel)
            
            # Process with MegaDescriptor
            inputs = self.megadescriptor_processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.megadescriptor_model(**inputs)
                # Get pooled output and take first 32 dimensions
                if hasattr(outputs, 'pooler_output'):
                    features = outputs.pooler_output.cpu().numpy().flatten()[:32]
                elif hasattr(outputs, 'last_hidden_state'):
                    # Alternative for models without pooler_output
                    features = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()[:32]
                else:
                    # Fallback
                    features = outputs[0].mean(dim=1).cpu().numpy().flatten()[:32]
            
            # Normalize (avoid division by zero)
            norm = np.linalg.norm(features)
            if norm > 1e-8:
                features = features / norm
            else:
                # If norm is too small, return small random features
                features = np.random.rand(32) * 0.01
            
            return features
            
        except Exception as e:
            print(f"❌ MegaDescriptor feature extraction failed: {e}")
            return np.random.rand(32) * 0.01
    
    def create_combined_embedding(self, rgb_image: np.ndarray, depth_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Create 64-dimensional embedding: 32 RGB + 32 Depth"""
        # Extract RGB features (32 dim)
        rgb_features = self.extract_megadescriptor_features(rgb_image, mask)
        
        # Extract Depth features (32 dim)
        depth_features = self.extract_megadescriptor_features(depth_image, mask)
        
        # Combine into 64-dimensional embedding
        combined_embedding = np.concatenate([rgb_features, depth_features])
        
        return combined_embedding
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between 64-dim embeddings with NaN protection"""
        try:
            # Check for NaN or invalid values
            if np.any(np.isnan(embedding1)) or np.any(np.isnan(embedding2)):
                return 0.0
            
            if np.any(np.isinf(embedding1)) or np.any(np.isinf(embedding2)):
                return 0.0
            
            # Calculate norms
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            # Check for zero norms
            if norm1 < 1e-8 or norm2 < 1e-8:
                return 0.0
            
            # Calculate cosine similarity
            cos_sim = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            # Ensure result is valid
            if np.isnan(cos_sim) or np.isinf(cos_sim):
                return 0.0
            
            # Clamp to valid range
            cos_sim = np.clip(cos_sim, -1.0, 1.0)
            
            return float(cos_sim)
            
        except Exception as e:
            print(f"❌ Similarity calculation failed: {e}")
            return 0.0
    
    def find_best_track_match(self, query_embedding: np.ndarray, query_position: np.ndarray, 
                             exclude_track_id: int, current_track_ids: set, max_racers: int = 9) -> Tuple[int, float]:
        """Find best memory track for assignment - limited to max_racers"""
        
        best_track_id = -1
        best_score = 0.0
        
        # Get all existing tracks from memory
        existing_tracks = list(self.memory.feature_memory.keys())
        
        # If we have reached max racers, only reassign to existing tracks
        if len(existing_tracks) >= max_racers:
            candidate_tracks = existing_tracks
        else:
            candidate_tracks = existing_tracks
        
        for track_id in candidate_tracks:
            if track_id == exclude_track_id:
                continue
                
            # Avoid conflicts with currently active tracks
            if track_id in current_track_ids:
                continue
            
            # Get recent features
            recent_features = self.memory.get_recent_features(track_id, n_recent=3)
            if not recent_features:
                continue
            
            # Calculate similarity with recent embeddings
            visual_similarities = [self.calculate_similarity(query_embedding, feat) for feat in recent_features]
            best_visual_sim = max(visual_similarities)
            
            # Calculate motion consistency
            predicted_pos = self.memory.predict_next_position(track_id)
            if predicted_pos is not None:
                distance = np.linalg.norm(query_position - predicted_pos)
                motion_score = 1.0 / (1.0 + distance / self.motion_distance_threshold)
            else:
                motion_score = 0.5

            if predicted_pos is not None:
                distance = np.linalg.norm(query_position - predicted_pos)
                print(f"Track {track_id}: predicted vs actual distance = {distance:.1f} pixels")
            
            # Get track stability
            stability = self.quality_monitor.get_track_stability(track_id)
            
            # Stability factors
            stability_bonus = self.stability_manager.get_stability_bonus(track_id)
            stability_penalty = self.stability_manager.get_stability_penalty(track_id)
            
            # Combined score
            combined_score = (best_visual_sim * 0.9 + motion_score * 0.05 + stability * 0.05 
                            + stability_bonus - stability_penalty)
            
            # Use reassignment threshold
            threshold = self.reassignment_threshold
            
            if combined_score > best_score and best_visual_sim > threshold:
                best_score = combined_score
                best_track_id = track_id
        
        return best_track_id, best_score
    
    def intelligent_track_assignment(self, detections, reid_features):
        """Core intelligent track assignment logic for limited racers"""
        
        if not sv or len(detections) == 0 or len(reid_features) == 0:
            return detections
        
        if not hasattr(detections, 'tracker_id'):
            return detections
        
        # Copy detections
        enhanced_detections = sv.Detections(
            xyxy=detections.xyxy.copy(),
            confidence=detections.confidence.copy() if hasattr(detections, 'confidence') else None,
            class_id=detections.class_id.copy() if hasattr(detections, 'class_id') else None,
            tracker_id=detections.tracker_id.copy()
        )
        
        reassignments_this_frame = 0
        max_racers = 9  # Fixed number of racers
        
        # Get set of currently active track IDs
        current_track_ids = set(enhanced_detections.tracker_id[enhanced_detections.tracker_id >= 0])
        
        # Find candidates for reassignment
        candidates = []
        
        for i, track_id in enumerate(detections.tracker_id):
            if track_id < 0:
                continue

            if i >= len(reid_features):
                continue
            
            # Skip if locked in cooling period
            if self.stability_manager.is_track_locked(track_id):
                self.stability_locks += 1
                continue
            
            bbox = detections.xyxy[i]
            center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            embedding = reid_features[i]
            
            should_check = False
            reason = ""
            
            # Check if new track
            if self.quality_monitor.is_new_track(track_id, self.frame_count, self.newness_threshold):
                should_check = True
                reason = "new_track"
            
            # Check if unstable track
            elif self.quality_monitor.is_unstable_track(track_id, self.stability_threshold):
                should_check = True
                reason = "unstable_track"
            
            if should_check:
                candidates.append({
                    'index': i,
                    'track_id': track_id,
                    'position': center,
                    'embedding': embedding,
                    'reason': reason
                })
        
        # Process reassignment candidates
        for candidate in candidates:
            i = candidate['index']
            current_track_id = candidate['track_id']
            position = candidate['position']
            embedding = candidate['embedding']
            reason = candidate['reason']
            
            # Find best memory match
            best_match_id, best_score = self.find_best_track_match(
                embedding, position, exclude_track_id=current_track_id, 
                current_track_ids=current_track_ids, max_racers=max_racers
            )
            
            # Decide on reassignment
            min_reassignment_score = 0.6
            
            if best_match_id >= 0 and best_score > min_reassignment_score:
                # Check for oscillation
                is_oscillating = self.stability_manager.record_assignment(
                    current_track_id, best_match_id, best_score
                )
                
                if is_oscillating:
                    self.oscillations_prevented += 1
                    continue
                
                # Safe to reassign
                enhanced_detections.tracker_id[i] = best_match_id
                current_track_ids.remove(current_track_id)
                current_track_ids.add(best_match_id)
                
                self.reassignment_count += 1
                reassignments_this_frame += 1
                
                print(f"✅ REID MATCH: Track #{current_track_id} → #{best_match_id} ({reason}, score: {best_score:.3f})")
                
                # Clean up old track if very new
                if self.quality_monitor.is_new_track(current_track_id, self.frame_count, 2):
                    self.cleanup_track(current_track_id)
        
        if reassignments_this_frame > 0:
            print(f"📊 Frame {self.frame_count}: {reassignments_this_frame} MegaDescriptor reassignments")
        
        return enhanced_detections
    
    def cleanup_track(self, track_id: int):
        """Remove track from all data structures"""
        self.memory.cleanup_track(track_id)
        self.quality_monitor._remove_track(track_id)
    
    def enhance_tracking(self, detections, reid_features, *args, **kwargs):
        """Main entry point for enhanced tracking - flexible interface for main.py compatibility"""
        
        # Handle optional arguments
        depth_stats = args[0] if len(args) > 0 else kwargs.get('depth_stats', None)
        tracked_horses = args[1] if len(args) > 1 else kwargs.get('tracked_horses', None)
        frame_info = args[2] if len(args) > 2 else kwargs.get('frame_info', None)
        
        if not sv or len(detections) == 0 or len(reid_features) == 0:
            return detections
        
        self.frame_count += 1
        self.stability_manager.advance_frame()
        
        # Update track quality monitoring
        if hasattr(detections, 'tracker_id'):
            for i, track_id in enumerate(detections.tracker_id):
                if track_id >= 0 and i < len(reid_features):
                    bbox = detections.xyxy[i]
                    center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                    confidence = detections.confidence[i] if hasattr(detections, 'confidence') else 0.8
                    
                    self.quality_monitor.update_track_quality(track_id, confidence, center, self.frame_count)
        
        # Apply intelligent track assignment
        enhanced_detections = self.intelligent_track_assignment(detections, reid_features)
        
        # Cleanup old tracks
        active_track_ids = set(enhanced_detections.tracker_id[enhanced_detections.tracker_id >= 0])
        self.quality_monitor.cleanup_old_tracks(active_track_ids, self.frame_count)
        self.stability_manager.cleanup_old_tracks(active_track_ids)
        
        return enhanced_detections
    
    def process_frame(self, frame: np.ndarray, detections) -> Tuple:
        """Process frame with SAM segmentation and MegaDescriptor embeddings"""
        
        # Estimate depth
        depth_map = self.estimate_depth_full_image(frame)
        self._last_depth_map = depth_map
        
        # Process each detection
        rgb_crops, depth_crops = [], []
        embeddings = []
        depth_stats = []
        self.current_masks = []
        
        if not sv or len(detections) == 0:
            return rgb_crops, depth_crops, depth_map, np.array([]), depth_stats
        
        for i, bbox in enumerate(detections.xyxy):
            track_id = detections.tracker_id[i] if hasattr(detections, 'tracker_id') and i < len(detections.tracker_id) else -1
            
            # Segment with SAM using bbox center
            mask, confidence = self.segment_with_sam(frame, bbox)
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
                    mask_crop = mask_crop.astype(bool)
                    rgb_crop[~mask_crop] = [240, 240, 240]
                    depth_crop[~mask_crop] = 0
                
                rgb_crops.append(rgb_crop)
                depth_crops.append(depth_crop)
                
                # Create combined 64-dim embedding (32 RGB + 32 Depth)
                embedding = self.create_combined_embedding(frame, depth_map, mask)
                embeddings.append(embedding)
                
                # Depth stats
                depth_stats.append({
                    'area': np.sum(mask),
                    'confidence': confidence,
                    'mask_quality': np.mean(mask.astype(float)),
                    'depth_variance': np.var(depth_crop[mask_crop]) if np.any(mask_crop) else 0
                })
                
                # Update memory if valid track
                if track_id >= 0:
                    center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                    self.memory.update_track_memory(track_id, embedding, center, bbox, mask, confidence)
        
        return rgb_crops, depth_crops, depth_map, np.array(embeddings), depth_stats
    
    def get_current_masks(self):
        """Get current segmentation masks"""
        return self.current_masks
    
    def get_reassignment_count(self):
        """Get total reassignments made"""
        return self.reassignment_count
    
    def get_tracking_info(self):
        """Get tracking information"""
        active_tracks = len(self.memory.feature_memory)
        
        motion_predictions = {}
        for track_id in self.memory.feature_memory.keys():
            pred_pos = self.memory.predict_next_position(track_id)
            if pred_pos is not None:
                motion_predictions[track_id] = pred_pos
        
        return {
            'active_tracks': active_tracks,
            'lost_tracks': 0,
            'motion_predictions': motion_predictions,
            'total_reassignments': self.reassignment_count,
            'oscillations_prevented': self.oscillations_prevented,
            'stability_locks': self.stability_locks,
            'frame_count': self.frame_count,
            'memory_efficiency': len(self.memory.feature_memory)
        }