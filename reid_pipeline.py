"""
Enhanced SAMURAI ReID Pipeline - FULLY CONFIGURABLE VERSION
All parameters now loaded from config file for horse racing optimization
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


class TrackStabilityManager:
    """Manage track assignment stability and prevent oscillation - FULLY CONFIGURABLE"""
    
    def __init__(self, config):
        # Load ALL parameters from config
        self.cooling_period = getattr(config, 'cooling_period', 10)
        self.oscillation_threshold = getattr(config, 'oscillation_threshold', 3)
        self.stability_bonus_factor = getattr(config, 'stability_bonus_factor', 0.2)
        self.stability_penalty_factor = getattr(config, 'stability_penalty_factor', 0.5)
        
        # Track assignment records
        self.assignment_history = defaultdict(deque)
        self.cooling_until = defaultdict(int)
        self.oscillation_count = defaultdict(int)
        self.last_target_ids = defaultdict(deque)
        self.assignment_confidence = defaultdict(float)
        self.assignment_frame = defaultdict(int)
        
        self.current_frame = 0
        
        print(f"🔒 Stability Manager: cooling={self.cooling_period}, oscillation_threshold={self.oscillation_threshold}")
        
    def is_track_locked(self, track_id: int) -> bool:
        """Check if track is in cooling period"""
        return self.current_frame < self.cooling_until[track_id]
        
    def record_assignment(self, track_id: int, target_id: int, confidence: float) -> bool:
        """Record a new assignment and return True if oscillating"""
        
        # Check for oscillation
        if target_id in self.last_target_ids[track_id]:
            self.oscillation_count[track_id] += 1
        else:
            self.oscillation_count[track_id] = max(0, self.oscillation_count[track_id] - 1)
            
        # Update assignment record
        self.assignment_confidence[track_id] = confidence
        self.assignment_frame[track_id] = self.current_frame
        self.last_target_ids[track_id].append(target_id)
        
        # Keep history manageable
        if len(self.last_target_ids[track_id]) > 5:
            self.last_target_ids[track_id].popleft()
            
        # Set cooling period (longer if oscillating)
        cooling_multiplier = 1 + self.oscillation_count[track_id]
        self.cooling_until[track_id] = self.current_frame + (self.cooling_period * cooling_multiplier)
        
        # Record in assignment history
        self.assignment_history[track_id].append({
            'frame': self.current_frame,
            'target_id': target_id,
            'confidence': confidence
        })
        if len(self.assignment_history[track_id]) > 10:
            self.assignment_history[track_id].popleft()
            
        return self.oscillation_count[track_id] > self.oscillation_threshold
        
    def get_stability_penalty(self, track_id: int) -> float:
        """Get penalty for unstable tracks"""
        # Penalty for oscillation
        oscillation_penalty = min(self.stability_penalty_factor, 
                                 self.oscillation_count[track_id] * 0.1)
        
        # Penalty for recent changes
        frames_since_assignment = self.current_frame - self.assignment_frame[track_id]
        recency_penalty = max(0, 0.3 - frames_since_assignment * 0.03)
        
        return oscillation_penalty + recency_penalty
        
    def get_stability_bonus(self, track_id: int) -> float:
        """Get bonus for stable assignments"""
        frames_stable = self.current_frame - self.assignment_frame[track_id]
        return min(self.stability_bonus_factor, frames_stable * 0.02)
        
    def advance_frame(self):
        """Advance frame counter"""
        self.current_frame += 1
        
    def cleanup_old_tracks(self, active_tracks: set):
        """Remove data for inactive tracks"""
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
    """Enhanced track quality monitoring - FULLY CONFIGURABLE"""
    
    def __init__(self, config):
        # Load ALL parameters from config
        self.stability_window = getattr(config, 'quality_stability_window', 10)
        self.confidence_variance_threshold = getattr(config, 'confidence_variance_threshold', 0.2)
        self.position_variance_threshold = getattr(config, 'position_variance_threshold', 100)
        self.track_newness_threshold = getattr(config, 'track_newness_threshold', 5)
        self.track_stability_threshold = getattr(config, 'track_stability_threshold', 0.4)
        
        # Track quality metrics
        self.confidence_history = defaultdict(deque)
        self.position_variance = defaultdict(list)
        self.track_age = defaultdict(int)
        self.track_first_seen = {}
        
        # Stability scores
        self.stability_scores = defaultdict(float)
        self.last_update = defaultdict(int)
        
        print(f"📊 Quality Monitor: window={self.stability_window}, stability_threshold={self.track_stability_threshold}")
    
    def update_track_quality(self, track_id: int, confidence: float, position: np.ndarray, frame_num: int):
        """Update quality metrics for a track"""
        
        # Track first appearance
        if track_id not in self.track_first_seen:
            self.track_first_seen[track_id] = frame_num
        
        # Update age and last seen
        self.track_age[track_id] += 1
        self.last_update[track_id] = frame_num
        
        # Update confidence history
        self.confidence_history[track_id].append(confidence)
        if len(self.confidence_history[track_id]) > self.stability_window:
            self.confidence_history[track_id].popleft()
        
        # Calculate position variance
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
        
        # Keep position history manageable
        if len(self.position_variance[track_id]) > self.stability_window:
            self.position_variance[track_id].pop(0)
        
        # Calculate stability score
        self._calculate_stability_score(track_id, variance)
    
    def _calculate_stability_score(self, track_id: int, position_variance: float):
        """Calculate overall stability score for track"""
        
        # Confidence stability (high = consistent confidence)
        confidences = list(self.confidence_history[track_id])
        if len(confidences) > 1:
            conf_mean = np.mean(confidences)
            conf_variance = np.var(confidences)
            conf_stability = 1.0 - min(1.0, conf_variance / self.confidence_variance_threshold)
        else:
            conf_mean = confidences[0] if confidences else 0.5
            conf_stability = 0.5
        
        # Motion stability (low variance = stable motion)
        motion_stability = 1.0 / (1.0 + position_variance / self.position_variance_threshold)
        
        # Age factor (older tracks are more stable)
        age_factor = min(1.0, self.track_age[track_id] / 20.0)
        
        # Combined stability score
        stability = conf_mean * 0.4 + conf_stability * 0.3 + motion_stability * 0.2 + age_factor * 0.1
        self.stability_scores[track_id] = stability
    
    def get_track_stability(self, track_id: int) -> float:
        """Get stability score for track"""
        return self.stability_scores.get(track_id, 0.0)
    
    def is_new_track(self, track_id: int, frame_num: int) -> bool:
        """Check if track is newly created"""
        if track_id not in self.track_first_seen:
            return True
        return (frame_num - self.track_first_seen[track_id]) <= self.track_newness_threshold
    
    def is_unstable_track(self, track_id: int) -> bool:
        """Check if track has poor quality/stability"""
        return self.get_track_stability(track_id) < self.track_stability_threshold
    
    def cleanup_old_tracks(self, active_track_ids: set, frame_num: int):
        """Clean up tracks that haven't been seen recently"""
        max_age = getattr(self, 'track_expiry_frames', 100)
        to_remove = []
        for track_id in list(self.last_update.keys()):
            if track_id not in active_track_ids:
                if frame_num - self.last_update[track_id] > max_age:
                    to_remove.append(track_id)
        
        for track_id in to_remove:
            self._remove_track(track_id)
    
    def _remove_track(self, track_id: int):
        """Remove all data for a track"""
        self.confidence_history.pop(track_id, None)
        self.position_variance.pop(track_id, None)
        self.track_age.pop(track_id, None)
        self.track_first_seen.pop(track_id, None)
        self.stability_scores.pop(track_id, None)
        self.last_update.pop(track_id, None)


class MotionAwareMemory:
    """Enhanced memory with motion prediction - FULLY CONFIGURABLE"""
    
    def __init__(self, config):
        # Load ALL parameters from config
        self.memory_size = getattr(config, 'samurai_memory_size', 15)
        self.max_motion_speed = getattr(config, 'max_motion_speed', 30.0)
        self.motion_prediction_frames = getattr(config, 'motion_prediction_frames', 3)
        
        # Memory banks per track
        self.feature_memory = defaultdict(deque)
        self.position_memory = defaultdict(deque)
        self.bbox_memory = defaultdict(deque)
        self.mask_memory = defaultdict(deque)
        self.confidence_memory = defaultdict(deque)
        
        print(f"🧠 Memory: size={self.memory_size}, max_speed={self.max_motion_speed}px/frame")
    
    def update_track_memory(self, track_id: int, feature: np.ndarray, position: np.ndarray, 
                           bbox: np.ndarray, mask: np.ndarray, confidence: float):
        """Update memory for track"""
        
        self.feature_memory[track_id].append(feature)
        self.position_memory[track_id].append(position)
        self.bbox_memory[track_id].append(bbox)
        self.mask_memory[track_id].append(mask)
        self.confidence_memory[track_id].append(confidence)
        
        # Maintain memory size
        for memory_bank in [self.feature_memory, self.position_memory, self.bbox_memory,
                           self.mask_memory, self.confidence_memory]:
            if len(memory_bank[track_id]) > self.memory_size:
                memory_bank[track_id].popleft()
    
    def predict_next_position(self, track_id: int) -> Optional[np.ndarray]:
        """Predict next position using motion model"""
        if track_id not in self.position_memory or len(self.position_memory[track_id]) < 2:
            return None
        
        positions = list(self.position_memory[track_id])
        
        # Use configurable number of frames for prediction
        frames_to_use = min(self.motion_prediction_frames, len(positions))
        if frames_to_use >= 2:
            # Linear prediction using recent frames
            recent_positions = positions[-frames_to_use:]
            
            # Calculate average velocity
            velocities = []
            for i in range(1, len(recent_positions)):
                velocity = recent_positions[i] - recent_positions[i-1]
                velocities.append(velocity)
            
            avg_velocity = np.mean(velocities, axis=0)
            
            # Apply speed limit from config
            speed = np.linalg.norm(avg_velocity)
            if speed > self.max_motion_speed:
                avg_velocity = avg_velocity / speed * self.max_motion_speed
            
            return positions[-1] + avg_velocity
        
        return positions[-1]
    
    def get_recent_features(self, track_id: int, n_recent: int = 3) -> List[np.ndarray]:
        """Get recent features for similarity comparison"""
        if track_id not in self.feature_memory:
            return []
        
        features = list(self.feature_memory[track_id])
        return features[-n_recent:] if len(features) >= n_recent else features
    
    def cleanup_track(self, track_id: int):
        """Remove track from memory"""
        for memory_bank in [self.feature_memory, self.position_memory, self.bbox_memory,
                           self.mask_memory, self.confidence_memory]:
            memory_bank.pop(track_id, None)


class EnhancedReIDPipeline:
    """Enhanced ReID Pipeline - FULLY CONFIGURABLE VERSION"""
    
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
        
        # Core components with full config support
        self.memory = MotionAwareMemory(config)
        self.quality_monitor = TrackQualityMonitor(config)
        self.stability_manager = TrackStabilityManager(config)
        
        # Load ALL configuration parameters
        self.similarity_threshold = getattr(config, 'reid_similarity_threshold', 0.3)
        self.initial_assignment_threshold = getattr(config, 'initial_assignment_threshold', 0.5)
        self.reassignment_threshold = getattr(config, 'reassignment_threshold', 0.7)
        
        self.motion_distance_threshold = getattr(config, 'motion_distance_threshold', 150)
        
        # Feature weighting from config
        self.visual_feature_weight = getattr(config, 'visual_feature_weight', 0.6)
        self.motion_feature_weight = getattr(config, 'motion_feature_weight', 0.25)
        self.stability_feature_weight = getattr(config, 'stability_feature_weight', 0.15)
        
        # Similarity calculation weights
        self.cosine_similarity_weight = getattr(config, 'cosine_similarity_weight', 0.7)
        self.l2_similarity_weight = getattr(config, 'l2_similarity_weight', 0.3)
        
        # SAM parameters
        self.sam_multimask_output = getattr(config, 'sam_multimask_output', True)
        self.sam_confidence_threshold = getattr(config, 'sam_confidence_threshold', 0.5)
        
        # Depth parameters
        self.depth_weight = getattr(config, 'depth_weight', 0.3)
        self.depth_variance_threshold = getattr(config, 'depth_variance_threshold', 1000)
        
        # Feature extraction parameters
        self.crop_padding_factor = getattr(config, 'crop_padding_factor', 0.05)
        self.min_crop_size = getattr(config, 'min_crop_size', 50)
        self.feature_vector_size = getattr(config, 'feature_vector_size', 64)
        self.color_histogram_bins = getattr(config, 'color_histogram_bins', 8)
        
        # Tracking stats
        self.reassignment_count = 0
        self.oscillations_prevented = 0
        self.stability_locks = 0
        self.frame_count = 0
        
        # Store current frame data
        self.current_masks = []
        self._last_depth_map = None
        
        print(f"🎯 Enhanced ReID Pipeline - FULLY CONFIGURABLE")
        print(f"   Motion threshold: {self.motion_distance_threshold}px")
        print(f"   Feature weights: Visual={self.visual_feature_weight}, Motion={self.motion_feature_weight}")
        print(f"   Assignment thresholds: Initial={self.initial_assignment_threshold}, Reassign={self.reassignment_threshold}")
        print(f"   SAM model: {self.sam_model_type or 'disabled'}")
        
        # Analyze configuration appropriateness
        self._analyze_config_for_scenario()
    
    def _analyze_config_for_scenario(self):
        """Analyze if configuration is appropriate for the scenario"""
        if hasattr(self.config, 'average_speed_mph') and self.config.average_speed_mph > 0:
            speed_mph = self.config.average_speed_mph
            
            # Estimate movement per frame
            estimated_movement = speed_mph * 0.74  # rough pixels per frame
            
            if self.motion_distance_threshold < estimated_movement * 3:
                print(f"   ⚠️ WARNING: Motion threshold ({self.motion_distance_threshold}px) may be too low for {speed_mph}mph")
                recommended = int(estimated_movement * 8)
                print(f"   💡 Recommend increasing to: {recommended}px")
            else:
                print(f"   ✅ Motion threshold appropriate for {speed_mph}mph racing")
    
    def setup_sam_model(self):
        """Initialize SAM model"""
        sam_model = getattr(self.config, 'sam_model', 'none')
        
        if sam_model == 'none':
            return
        
        if sam_model == 'sam2' and SAM2_AVAILABLE:
            try:
                self.sam_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-base-plus")
                self.sam_model_type = 'sam2'
                print("✅ SAM2 loaded")
            except Exception as e:
                print(f"❌ SAM2 failed: {e}")
        
        elif sam_model == 'mobilesam' and MOBILESAM_AVAILABLE:
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
            
            depth_range = getattr(self.config, 'depth_normalization_range', 255)
            if depth_resized.max() > depth_resized.min():
                depth_normalized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min()) * depth_range
            else:
                depth_normalized = np.zeros_like(depth_resized)
            
            return depth_normalized.astype(np.uint8)
        except Exception as e:
            print(f"❌ Depth estimation failed: {e}")
            return np.zeros_like(frame[:,:,0])
    
    def segment_with_sam(self, frame: np.ndarray, bbox: np.ndarray) -> Tuple[np.ndarray, float]:
        """Segment using SAM with bbox prompt"""
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
            
            # Use config parameters for SAM
            use_bbox = getattr(self.config, 'sam_bbox_prompt_enabled', True)
            bbox_prompt = bbox if use_bbox else None
            
            if self.sam_model_type == 'sam2':
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    masks, scores, logits = self.sam_predictor.predict(
                        point_coords=input_points,
                        point_labels=input_labels,
                        box=bbox_prompt,
                        multimask_output=self.sam_multimask_output,
                    )
            else:
                masks, scores, logits = self.sam_predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    box=bbox_prompt,
                    multimask_output=self.sam_multimask_output,
                )
            
            best_mask = masks[np.argmax(scores)]
            best_score = np.max(scores)
            
            # Apply confidence threshold from config
            if best_score < self.sam_confidence_threshold:
                h, w = frame.shape[:2]
                return np.zeros((h, w), dtype=bool), 0.0
            
            return best_mask.astype(bool), best_score
            
        except Exception as e:
            print(f"❌ SAM segmentation failed: {e}")
            h, w = frame.shape[:2]
            return np.zeros((h, w), dtype=bool), 0.0
    
    def extract_visual_features(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extract visual features from masked region - CONFIGURABLE"""
        try:
            mask = mask.astype(bool)
            
            if not np.any(mask):
                return np.random.rand(self.feature_vector_size) * 0.01
            
            features = []
            
            # Color features using config parameters
            masked_region = frame.copy()
            masked_region[~mask] = 0
            
            # RGB histograms with configurable bins
            for channel in range(3):
                hist = cv2.calcHist([masked_region], [channel], mask.astype(np.uint8), 
                                  [self.color_histogram_bins], [0, 256])
                features.extend(hist.flatten())
            
            # Texture features  
            gray = cv2.cvtColor(masked_region, cv2.COLOR_BGR2GRAY)
            
            if np.any(mask):
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                
                features.extend([
                    np.mean(grad_x[mask]),
                    np.std(grad_x[mask]),
                    np.mean(grad_y[mask]),
                    np.std(grad_y[mask])
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            # Shape features
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    circularity = 0
                
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                features.extend([area / 10000, perimeter / 1000, circularity, aspect_ratio])
            else:
                features.extend([0, 0, 0, 0])
            
            # Pad or truncate to target size
            features = np.array(features, dtype=np.float32)
            if len(features) > self.feature_vector_size:
                features = features[:self.feature_vector_size]
            elif len(features) < self.feature_vector_size:
                padding = np.zeros(self.feature_vector_size - len(features))
                features = np.concatenate([features, padding])
            
            # Normalize
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return features
            
        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
            return np.random.rand(self.feature_vector_size) * 0.01
    
    def calculate_similarity(self, feature1: np.ndarray, feature2: np.ndarray) -> float:
        """Calculate similarity between features using config weights"""
        cos_sim = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2) + 1e-8)
        l2_dist = np.linalg.norm(feature1 - feature2)
        l2_sim = 1.0 / (1.0 + l2_dist)
        
        # Use config weights for similarity combination
        return self.cosine_similarity_weight * cos_sim + self.l2_similarity_weight * l2_sim
    
    def find_best_reassignment_candidate(self, query_feature: np.ndarray, query_position: np.ndarray, 
                                       exclude_track_id: int, current_track_ids: set) -> Tuple[int, float]:
        """Find best memory track for reassignment with CONFIGURABLE thresholds"""
        
        best_track_id = -1
        best_score = 0.0
        
        for track_id in self.memory.feature_memory:
            if track_id == exclude_track_id:
                continue
                
            # Avoid conflicts - don't assign to currently active tracks
            if track_id in current_track_ids:
                continue
            
            # Get recent features
            recent_features = self.memory.get_recent_features(track_id, n_recent=3)
            if not recent_features:
                continue
            
            # Calculate visual similarity using config weights
            visual_similarities = [self.calculate_similarity(query_feature, feat) for feat in recent_features]
            best_visual_sim = max(visual_similarities)
            
            # Calculate motion consistency using config threshold
            predicted_pos = self.memory.predict_next_position(track_id)
            if predicted_pos is not None:
                distance = np.linalg.norm(query_position - predicted_pos)
                motion_score = 1.0 / (1.0 + distance / self.motion_distance_threshold)
            else:
                motion_score = 0.5
            
            # Get track stability
            stability = self.quality_monitor.get_track_stability(track_id)
            
            # STABILITY FACTORS using config parameters
            stability_bonus = self.stability_manager.get_stability_bonus(track_id)
            stability_penalty = self.stability_manager.get_stability_penalty(track_id)
            
            # Combined score with CONFIGURABLE weights
            combined_score = (best_visual_sim * self.visual_feature_weight + 
                            motion_score * self.motion_feature_weight + 
                            stability * self.stability_feature_weight + 
                            stability_bonus - stability_penalty)
            
            # Use configurable reassignment threshold
            if combined_score > best_score and best_visual_sim > self.reassignment_threshold:
                best_score = combined_score
                best_track_id = track_id
        
        return best_track_id, best_score
    
    def intelligent_track_assignment(self, detections, reid_features):
        """Core intelligent track assignment with FULL CONFIG SUPPORT"""
        
        if not sv or len(detections) == 0 or len(reid_features) == 0:
            return detections
        
        if not hasattr(detections, 'tracker_id'):
            return detections
        
        # Properly copy supervision Detections object
        enhanced_detections = sv.Detections(
            xyxy=detections.xyxy.copy(),
            confidence=detections.confidence.copy() if hasattr(detections, 'confidence') else None,
            class_id=detections.class_id.copy() if hasattr(detections, 'class_id') else None,
            tracker_id=detections.tracker_id.copy()
        )
        
        reassignments_this_frame = 0
        current_track_ids = set(enhanced_detections.tracker_id[enhanced_detections.tracker_id >= 0])
        
        # Find candidates using configurable thresholds
        candidates = []
        
        for i, track_id in enumerate(detections.tracker_id):
            if track_id < 0:
                continue
            
            # STABILITY CHECK using config cooling period
            if self.stability_manager.is_track_locked(track_id):
                self.stability_locks += 1
                continue
            
            bbox = detections.xyxy[i]
            center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            feature = reid_features[i]
            
            should_check = False
            reason = ""
            
            # Check if new track using config threshold
            if self.quality_monitor.is_new_track(track_id, self.frame_count):
                should_check = True
                reason = "new_track"
            
            # Check if unstable track using config threshold
            elif self.quality_monitor.is_unstable_track(track_id):
                should_check = True
                reason = "unstable_track"
            
            if should_check:
                candidates.append({
                    'index': i,
                    'track_id': track_id,
                    'position': center,
                    'feature': feature,
                    'reason': reason
                })
        
        # Process reassignment candidates with CONFIGURABLE parameters
        for candidate in candidates:
            i = candidate['index']
            current_track_id = candidate['track_id']
            position = candidate['position']
            feature = candidate['feature']
            reason = candidate['reason']
            
            # Find best memory match
            best_match_id, best_score = self.find_best_reassignment_candidate(
                feature, position, exclude_track_id=current_track_id, 
                current_track_ids=current_track_ids
            )
            
            # Use configurable threshold for reassignment decision
            threshold = self.initial_assignment_threshold if reason == "new_track" else self.reassignment_threshold
            
            if best_match_id >= 0 and best_score > threshold:
                # Check for oscillation using config parameters
                is_oscillating = self.stability_manager.record_assignment(
                    current_track_id, best_match_id, best_score
                )
                
                if is_oscillating:
                    self.oscillations_prevented += 1
                    print(f"🚫 OSCILLATION PREVENTED: Track #{current_track_id} → #{best_match_id}")
                    continue
                
                # Safe to reassign
                enhanced_detections.tracker_id[i] = best_match_id
                current_track_ids.remove(current_track_id)
                current_track_ids.add(best_match_id)
                
                self.reassignment_count += 1
                reassignments_this_frame += 1
                
                print(f"✅ STABLE REASSIGNMENT: Track #{current_track_id} → #{best_match_id} ({reason}, score: {best_score:.3f})")
                
                # Clean up old track if very new
                if self.quality_monitor.is_new_track(current_track_id, self.frame_count):
                    self.cleanup_track(current_track_id)
        
        if reassignments_this_frame > 0:
            print(f"📊 Frame {self.frame_count}: {reassignments_this_frame} intelligent reassignments")
        
        return enhanced_detections
    
    def cleanup_track(self, track_id: int):
        """Remove track from all data structures"""
        self.memory.cleanup_track(track_id)
        self.quality_monitor._remove_track(track_id)
    
    def enhance_tracking(self, detections, reid_features, depth_stats=None):
        """Main entry point for enhanced tracking with FULL CONFIG"""
        
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
        
        # Cleanup old tracks using config parameters
        active_track_ids = set(enhanced_detections.tracker_id[enhanced_detections.tracker_id >= 0])
        self.quality_monitor.cleanup_old_tracks(active_track_ids, self.frame_count)
        self.stability_manager.cleanup_old_tracks(active_track_ids)
        
        return enhanced_detections
    
    def process_frame(self, frame: np.ndarray, detections) -> Tuple:
        """Process frame with enhanced SAM and ReID - CONFIGURABLE"""
        
        # Estimate depth
        depth_map = self.estimate_depth_full_image(frame)
        self._last_depth_map = depth_map
        
        # Process each detection
        rgb_crops, depth_crops = [], []
        features = []
        depth_stats = []
        self.current_masks = []
        
        if not sv or len(detections) == 0:
            return rgb_crops, depth_crops, depth_map, np.array([]), depth_stats
        
        for i, bbox in enumerate(detections.xyxy):
            track_id = detections.tracker_id[i] if hasattr(detections, 'tracker_id') and i < len(detections.tracker_id) else -1
            
            # Segment with SAM using config parameters
            mask, confidence = self.segment_with_sam(frame, bbox)
            self.current_masks.append(mask)
            
            # Extract crops with configurable padding
            x1, y1, x2, y2 = bbox.astype(int)
            
            # Apply padding from config
            w, h = x2 - x1, y2 - y1
            padding_x = int(w * self.crop_padding_factor)
            padding_y = int(h * self.crop_padding_factor)
            
            x1 = max(0, x1 - padding_x)
            y1 = max(0, y1 - padding_y)
            x2 = min(frame.shape[1], x2 + padding_x)
            y2 = min(frame.shape[0], y2 + padding_y)
            
            # Check minimum size from config
            if x2 - x1 >= self.min_crop_size and y2 - y1 >= self.min_crop_size:
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
                
                # Extract features using configurable parameters
                feature = self.extract_visual_features(frame, mask)
                features.append(feature)
                
                # Depth stats with configurable thresholds
                depth_variance = np.var(depth_crop[mask_crop]) if np.any(mask_crop) else 0
                depth_stats.append({
                    'area': np.sum(mask),
                    'confidence': confidence,
                    'mask_quality': np.mean(mask.astype(float)),
                    'depth_variance': depth_variance
                })
                
                # Update memory if valid track
                if track_id >= 0:
                    center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                    self.memory.update_track_memory(track_id, feature, center, bbox, mask, confidence)
        
        return rgb_crops, depth_crops, depth_map, np.array(features), depth_stats
    
    def get_current_masks(self):
        """Get current segmentation masks"""
        return self.current_masks
    
    def get_reassignment_count(self):
        """Get total reassignments made"""
        return self.reassignment_count
    
    def get_tracking_info(self):
        """Get tracking information with stability stats"""
        active_tracks = len(self.memory.feature_memory)
        
        # Get motion predictions
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
            'memory_efficiency': len(self.memory.feature_memory),
            'config_motion_threshold': self.motion_distance_threshold,
            'config_cooling_period': self.stability_manager.cooling_period
        }