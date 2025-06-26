"""
Enhanced SAMURAI ReID Pipeline with 3D Pose Integration - FIXED VERSION
Implements intelligent track assignment with quality-based reassignment
FIXED: Feature dimension mismatch between 3D-enhanced and original features
"""

import torch
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch.nn.functional as F
from collections import deque, defaultdict
import os
from PIL import Image

# Import our 3D pose estimator
from pose_3d_estimator import Pose3DEstimator

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
    """NEW: Manage track assignment stability and prevent oscillation"""
    
    def __init__(self, cooling_period=10, oscillation_threshold=3):
        self.cooling_period = cooling_period  # Frames to wait after reassignment
        self.oscillation_threshold = oscillation_threshold  # Max oscillations before penalty
        
        # Track assignment records
        self.assignment_history = defaultdict(deque)  # Track recent assignments
        self.cooling_until = defaultdict(int)  # Frame when cooling ends
        self.oscillation_count = defaultdict(int)  # Count of oscillations
        self.last_target_ids = defaultdict(deque)  # Recent target IDs
        self.assignment_confidence = defaultdict(float)  # Confidence in current assignment
        self.assignment_frame = defaultdict(int)  # When assignment was made
        
        self.current_frame = 0
        
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
        oscillation_penalty = min(0.5, self.oscillation_count[track_id] * 0.1)
        
        # Penalty for recent changes
        frames_since_assignment = self.current_frame - self.assignment_frame[track_id]
        recency_penalty = max(0, 0.3 - frames_since_assignment * 0.03)
        
        return oscillation_penalty + recency_penalty
        
    def get_stability_bonus(self, track_id: int) -> float:
        """Get bonus for stable assignments"""
        frames_stable = self.current_frame - self.assignment_frame[track_id]
        return min(0.2, frames_stable * 0.02)
        
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
    """Enhanced track quality monitoring with temporal consistency"""
    
    def __init__(self, stability_window=10):
        self.stability_window = stability_window
        
        # Track quality metrics
        self.confidence_history = defaultdict(deque)
        self.position_variance = defaultdict(list)
        self.track_age = defaultdict(int)
        self.track_first_seen = {}
        
        # Stability scores
        self.stability_scores = defaultdict(float)
        self.last_update = defaultdict(int)
    
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
            conf_stability = 1.0 - min(1.0, np.std(confidences))
        else:
            conf_mean = confidences[0] if confidences else 0.5
            conf_stability = 0.5
        
        # Motion stability (low variance = stable motion)
        motion_stability = 1.0 / (1.0 + position_variance / 100.0)
        
        # Age factor (older tracks are more stable)
        age_factor = min(1.0, self.track_age[track_id] / 20.0)
        
        # Combined stability score
        stability = conf_mean * 0.4 + conf_stability * 0.3 + motion_stability * 0.2 + age_factor * 0.1
        self.stability_scores[track_id] = stability
    
    def get_track_stability(self, track_id: int) -> float:
        """Get stability score for track (0.0 = unstable, 1.0 = very stable)"""
        return self.stability_scores.get(track_id, 0.0)
    
    def is_new_track(self, track_id: int, frame_num: int, newness_threshold: int = 5) -> bool:
        """Check if track is newly created"""
        if track_id not in self.track_first_seen:
            return True
        return (frame_num - self.track_first_seen[track_id]) <= newness_threshold
    
    def is_unstable_track(self, track_id: int, stability_threshold: float = 0.4) -> bool:
        """Check if track has poor quality/stability"""
        return self.get_track_stability(track_id) < stability_threshold
    
    def cleanup_old_tracks(self, active_track_ids: set, frame_num: int, max_age: int = 50):
        """Clean up tracks that haven't been seen recently"""
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
    """Enhanced memory with motion prediction and consistent feature dimensions"""
    
    def __init__(self, memory_size=15):
        self.memory_size = memory_size
        
        # Memory banks per track
        self.feature_memory = defaultdict(deque)
        self.position_memory = defaultdict(deque)
        self.bbox_memory = defaultdict(deque)
        self.mask_memory = defaultdict(deque)
        self.confidence_memory = defaultdict(deque)
        
        # NEW: Track feature dimensions for consistency
        self.feature_dimensions = defaultdict(int)
    
    def update_track_memory(self, track_id: int, feature: np.ndarray, position: np.ndarray, 
                           bbox: np.ndarray, mask: np.ndarray, confidence: float):
        """Update memory for track with consistent feature dimensions"""
        
        # Store feature dimension for this track
        self.feature_dimensions[track_id] = len(feature)
        
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
        
        # Simple linear prediction
        if len(positions) >= 2:
            velocity = positions[-1] - positions[-2]
            # Apply reasonable speed limit
            max_speed = 30.0  # pixels per frame
            speed = np.linalg.norm(velocity)
            if speed > max_speed:
                velocity = velocity / speed * max_speed
            
            return positions[-1] + velocity
        
        return positions[-1]
    
    def get_recent_features(self, track_id: int, n_recent: int = 3) -> List[np.ndarray]:
        """Get recent features for similarity comparison"""
        if track_id not in self.feature_memory:
            return []
        
        features = list(self.feature_memory[track_id])
        return features[-n_recent:] if len(features) >= n_recent else features
    
    def get_feature_dimension(self, track_id: int) -> int:
        """Get stored feature dimension for track"""
        return self.feature_dimensions.get(track_id, 0)
    
    def cleanup_track(self, track_id: int):
        """Remove track from memory"""
        for memory_bank in [self.feature_memory, self.position_memory, self.bbox_memory,
                           self.mask_memory, self.confidence_memory]:
            memory_bank.pop(track_id, None)
        self.feature_dimensions.pop(track_id, None)


class EnhancedReIDPipeline:
    """Enhanced ReID Pipeline with Intelligent Track Assignment + STABILITY CONTROLS + 3D POSE INTEGRATION"""
    
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
        
        # NEW: Initialize 3D Pose Estimator
        self.pose_3d_estimator = None
        if getattr(config, 'enable_pose_prediction', False) or getattr(config, 'enable_multi_scale_depth', False):
            try:
                self.pose_3d_estimator = Pose3DEstimator(config)
                print("✅ 3D Pose Estimator integrated")
            except Exception as e:
                print(f"⚠️ 3D Pose Estimator failed to initialize: {e}")
        
        # Core components
        self.memory = MotionAwareMemory(
            memory_size=getattr(config, 'samurai_memory_size', 15)
        )
        self.quality_monitor = TrackQualityMonitor(
            stability_window=getattr(config, 'quality_stability_window', 10)
        )
        
        # NEW: Stability management
        self.stability_manager = TrackStabilityManager(
            cooling_period=getattr(config, 'cooling_period', 10),
            oscillation_threshold=getattr(config, 'oscillation_threshold', 3)
        )
        
        # Configuration parameters with HYSTERESIS
        self.similarity_threshold = getattr(config, 'reid_similarity_threshold', 0.3)
        self.initial_assignment_threshold = getattr(config, 'initial_assignment_threshold', 0.5)
        self.reassignment_threshold = getattr(config, 'reassignment_threshold', 0.7)  # Higher bar for stealing
        
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
        self._current_poses_3d = []  # NEW: Store 3D poses
        
        # NEW: Feature dimension management
        self.base_visual_feature_dim = 32  # Standard visual features
        self.geometric_feature_dim = 20    # 3D geometric features
        self.enhanced_feature_dim = self.base_visual_feature_dim + self.geometric_feature_dim  # 52
        
        print(f"🎯 Enhanced ReID Pipeline initialized with STABILITY CONTROLS + 3D INTEGRATION")
        print(f"   SAM model: {self.sam_model_type or 'disabled'}")
        print(f"   3D Pose Integration: {'ENABLED' if self.pose_3d_estimator else 'DISABLED'}")
        print(f"   Feature dimensions: Visual({self.base_visual_feature_dim}) + Geometric({self.geometric_feature_dim}) = Enhanced({self.enhanced_feature_dim})")
        print(f"   Initial assignment threshold: {self.initial_assignment_threshold}")
        print(f"   Reassignment threshold: {self.reassignment_threshold}")
        print(f"   Cooling period: {getattr(config, 'cooling_period', 10)} frames")
        print(f"   Oscillation prevention: ENABLED")
    
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
    
    def extract_visual_features(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extract visual features from masked region - always returns 32-dimensional features"""
        try:
            mask = mask.astype(bool)
            
            if not np.any(mask):
                return np.random.rand(self.base_visual_feature_dim) * 0.01
            
            features = []
            
            # Color features (24 dimensions: 8 bins × 3 channels)
            masked_region = frame.copy()
            masked_region[~mask] = 0
            
            # RGB histograms
            for channel in range(3):
                hist = cv2.calcHist([masked_region], [channel], mask.astype(np.uint8), [8], [0, 256])
                features.extend(hist.flatten())
            
            # Texture features (4 dimensions)
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
            
            # Shape features (4 dimensions)
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
            
            # Ensure exactly 32 dimensions
            features = features[:self.base_visual_feature_dim] + [0] * max(0, self.base_visual_feature_dim - len(features))
            
            # Normalize
            features = np.array(features, dtype=np.float32)
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return features
            
        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
            return np.random.rand(self.base_visual_feature_dim) * 0.01
    
    def normalize_feature_dimensions(self, feature1: np.ndarray, feature2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        CRITICAL FIX: Normalize features to same dimensions for comparison
        Handles mismatch between 32-dim visual and 52-dim enhanced features
        """
        dim1, dim2 = len(feature1), len(feature2)
        
        if dim1 == dim2:
            return feature1, feature2
        
        # Determine target dimension (use the larger one for better information preservation)
        target_dim = max(dim1, dim2)
        
        def pad_or_truncate(feat, target_size):
            if len(feat) < target_size:
                # Pad with zeros
                padded = np.zeros(target_size)
                padded[:len(feat)] = feat
                return padded
            elif len(feat) > target_size:
                # If enhanced features (52-dim) need to be compared with visual (32-dim),
                # extract only the visual part (first 32 dimensions)
                return feat[:target_size]
            else:
                return feat
        
        normalized_feat1 = pad_or_truncate(feature1, target_dim)
        normalized_feat2 = pad_or_truncate(feature2, target_dim)
        
        return normalized_feat1, normalized_feat2
    
    def calculate_similarity(self, feature1: np.ndarray, feature2: np.ndarray) -> float:
        """
        FIXED: Calculate similarity between features with dimension normalization
        """
        try:
            # Normalize feature dimensions to handle mismatch
            norm_feat1, norm_feat2 = self.normalize_feature_dimensions(feature1, feature2)
            
            # Calculate cosine similarity
            dot_product = np.dot(norm_feat1, norm_feat2)
            norm_product = np.linalg.norm(norm_feat1) * np.linalg.norm(norm_feat2)
            cos_sim = dot_product / (norm_product + 1e-8)
            
            # Calculate L2 similarity
            l2_dist = np.linalg.norm(norm_feat1 - norm_feat2)
            l2_sim = 1.0 / (1.0 + l2_dist)
            
            # Combined similarity
            similarity = 0.7 * cos_sim + 0.3 * l2_sim
            
            return float(np.clip(similarity, 0.0, 1.0))
            
        except Exception as e:
            print(f"❌ Similarity calculation failed: {e}")
            return 0.0
    
    def find_best_reassignment_candidate(self, query_feature: np.ndarray, query_position: np.ndarray, 
                                       exclude_track_id: int, current_track_ids: set) -> Tuple[int, float]:
        """Find best memory track for reassignment with STABILITY CONTROLS and dimension handling"""
        
        best_track_id = -1
        best_score = 0.0
        
        for track_id in self.memory.feature_memory:
            if track_id == exclude_track_id:
                continue
                
            # CRITICAL: Avoid conflicts - don't assign to currently active tracks
            if track_id in current_track_ids:
                continue
            
            # Get recent features
            recent_features = self.memory.get_recent_features(track_id, n_recent=3)
            if not recent_features:
                continue
            
            # Calculate visual similarity with dimension normalization
            visual_similarities = []
            for feat in recent_features:
                try:
                    sim = self.calculate_similarity(query_feature, feat)
                    visual_similarities.append(sim)
                except Exception as e:
                    print(f"⚠️ Similarity calculation failed for track {track_id}: {e}")
                    visual_similarities.append(0.0)
            
            if not visual_similarities:
                continue
                
            best_visual_sim = max(visual_similarities)
            
            # Calculate motion consistency
            predicted_pos = self.memory.predict_next_position(track_id)
            if predicted_pos is not None:
                distance = np.linalg.norm(query_position - predicted_pos)
                motion_score = 1.0 / (1.0 + distance / self.motion_distance_threshold)
            else:
                motion_score = 0.5  # Neutral score if no motion prediction
            
            # Get track stability
            stability = self.quality_monitor.get_track_stability(track_id)
            
            # STABILITY FACTORS
            stability_bonus = self.stability_manager.get_stability_bonus(track_id)
            stability_penalty = self.stability_manager.get_stability_penalty(track_id)
            
            # Combined score with stability
            combined_score = (best_visual_sim * 0.6 + motion_score * 0.25 + stability * 0.15 
                            + stability_bonus - stability_penalty)
            
            # Use appropriate threshold
            threshold = self.reassignment_threshold  # Higher bar for reassignment
            
            if combined_score > best_score and best_visual_sim > threshold:
                best_score = combined_score
                best_track_id = track_id
        
        return best_track_id, best_score
    
    def intelligent_track_assignment(self, detections, reid_features):
        """Core intelligent track assignment logic WITH STABILITY CONTROLS"""
        
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
        
        # Get set of currently active track IDs to avoid conflicts
        current_track_ids = set(enhanced_detections.tracker_id[enhanced_detections.tracker_id >= 0])
        
        # Find candidates for reassignment
        candidates = []
        
        for i, track_id in enumerate(detections.tracker_id):
            if track_id < 0:
                continue

            if i >= len(reid_features):
                print(f"⚠️ Missing ReID feature for detection index {i}, skipping")
                continue
            
            # STABILITY CHECK: Skip if track is locked in cooling period
            if self.stability_manager.is_track_locked(track_id):
                self.stability_locks += 1
                continue
            
            bbox = detections.xyxy[i]
            center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            feature = reid_features[i]
            
            should_check = False
            reason = ""
            
            # Check if new track
            if self.quality_monitor.is_new_track(track_id, self.frame_count, self.newness_threshold):
                should_check = True
                reason = "new_track"
            
            # Check if unstable track (but only if not locked)
            elif self.quality_monitor.is_unstable_track(track_id, self.stability_threshold):
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
        
        # Process reassignment candidates with STABILITY CONTROLS
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
            
            # Decide on reassignment with higher threshold
            min_reassignment_score = 0.6  # Higher threshold for reassignment
            
            if best_match_id >= 0 and best_score > min_reassignment_score:
                # Check for oscillation
                is_oscillating = self.stability_manager.record_assignment(
                    current_track_id, best_match_id, best_score
                )
                
                if is_oscillating:
                    self.oscillations_prevented += 1
                    print(f"🚫 OSCILLATION PREVENTED: Track #{current_track_id} → #{best_match_id}")
                    continue
                
                # Safe to reassign
                enhanced_detections.tracker_id[i] = best_match_id
                current_track_ids.remove(current_track_id)  # Remove old
                current_track_ids.add(best_match_id)  # Add new
                
                self.reassignment_count += 1
                reassignments_this_frame += 1
                
                print(f"✅ STABLE REASSIGNMENT: Track #{current_track_id} → #{best_match_id} ({reason}, score: {best_score:.3f})")
                
                # Clean up old track if it was very new
                if self.quality_monitor.is_new_track(current_track_id, self.frame_count, 2):
                    self.cleanup_track(current_track_id)
        
        if reassignments_this_frame > 0:
            print(f"📊 Frame {self.frame_count}: {reassignments_this_frame} intelligent reassignments")
        
        return enhanced_detections
    
    def cleanup_track(self, track_id: int):
        """Remove track from all data structures"""
        self.memory.cleanup_track(track_id)
        self.quality_monitor._remove_track(track_id)
        
        # NEW: Cleanup 3D pose history
        if self.pose_3d_estimator:
            self.pose_3d_estimator.cleanup_track_history(track_id)
    
    def enhance_tracking(self, detections, reid_features, depth_stats=None, poses_2d=None):
        """Main entry point for enhanced tracking WITH 3D POSE INTEGRATION"""
        
        if not sv or len(detections) == 0 or len(reid_features) == 0:
            return detections
        
        self.frame_count += 1
        self.stability_manager.advance_frame()  # NEW: Advance stability manager
        
        # NEW: 3D Pose Integration with dimension consistency
        enhanced_reid_features = reid_features
        if self.pose_3d_estimator and poses_2d and self._last_depth_map is not None:
            try:
                # Estimate 3D poses from 2D poses and depth
                poses_3d = self.pose_3d_estimator.estimate_3d_poses(poses_2d, self._last_depth_map)
                self._current_poses_3d = poses_3d
                
                # Enhance ReID features with 3D geometric features
                enhanced_reid_features = []
                for i, visual_feat in enumerate(reid_features):
                    pose_3d = poses_3d[i] if i < len(poses_3d) else None
                    enhanced_feat = self.pose_3d_estimator.combine_features_with_3d(visual_feat, pose_3d)
                    enhanced_reid_features.append(enhanced_feat)
                
                enhanced_reid_features = np.array(enhanced_reid_features) if enhanced_reid_features else reid_features
                
                print(f"🎯 3D Pose Integration: Enhanced {len(enhanced_reid_features)} features "
                      f"from {reid_features[0].shape if len(reid_features) > 0 else 0}-D to "
                      f"{enhanced_reid_features[0].shape if len(enhanced_reid_features) > 0 else 0}-D")
                
            except Exception as e:
                print(f"⚠️ 3D Pose integration failed: {e}")
                enhanced_reid_features = reid_features
                self._current_poses_3d = []
        else:
            self._current_poses_3d = []
        
        # Update track quality monitoring
        if hasattr(detections, 'tracker_id'):
            for i, track_id in enumerate(detections.tracker_id):
                if track_id >= 0 and i < len(enhanced_reid_features):
                    bbox = detections.xyxy[i]
                    center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                    confidence = detections.confidence[i] if hasattr(detections, 'confidence') else 0.8
                    
                    self.quality_monitor.update_track_quality(track_id, confidence, center, self.frame_count)
        
        # Apply intelligent track assignment with STABILITY CONTROLS and enhanced features
        enhanced_detections = self.intelligent_track_assignment(detections, enhanced_reid_features)
        
        # Cleanup old tracks
        active_track_ids = set(enhanced_detections.tracker_id[enhanced_detections.tracker_id >= 0])
        self.quality_monitor.cleanup_old_tracks(active_track_ids, self.frame_count)
        self.stability_manager.cleanup_old_tracks(active_track_ids)  # NEW: Cleanup stability data
        
        return enhanced_detections
    
    def process_frame(self, frame: np.ndarray, detections) -> Tuple:
        """Process frame with enhanced SAM and ReID"""
        
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
            
            # Segment with SAM
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
                
                # Extract visual features (always 32-dimensional)
                feature = self.extract_visual_features(frame, mask)
                features.append(feature)
                
                # Depth stats
                depth_stats.append({
                    'area': np.sum(mask),
                    'confidence': confidence,
                    'mask_quality': np.mean(mask.astype(float)),
                    'depth_variance': np.var(depth_crop[mask_crop]) if np.any(mask_crop) else 0
                })
                
                # Update memory if valid track (store enhanced features if available)
                if track_id >= 0:
                    center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                    # Note: We store visual features in memory and enhance them during comparison
                    # This ensures consistency across frames regardless of 3D availability
                    self.memory.update_track_memory(track_id, feature, center, bbox, mask, confidence)
        
        return rgb_crops, depth_crops, depth_map, np.array(features), depth_stats
    
    def get_current_masks(self):
        """Get current segmentation masks"""
        return self.current_masks
    
    def get_current_poses_3d(self):
        """Get current 3D poses"""
        return self._current_poses_3d
    
    def get_reassignment_count(self):
        """Get total reassignments made"""
        return self.reassignment_count
    
    def get_tracking_info(self):
        """Get tracking information with stability stats and 3D pose info"""
        active_tracks = len(self.memory.feature_memory)
        
        # Get motion predictions
        motion_predictions = {}
        for track_id in self.memory.feature_memory.keys():
            pred_pos = self.memory.predict_next_position(track_id)
            if pred_pos is not None:
                motion_predictions[track_id] = pred_pos
        
        # Get 3D pose stats
        pose_3d_stats = {}
        if self.pose_3d_estimator:
            pose_3d_stats = self.pose_3d_estimator.get_3d_stats()
        
        return {
            'active_tracks': active_tracks,
            'lost_tracks': 0,
            'motion_predictions': motion_predictions,
            'total_reassignments': self.reassignment_count,
            'oscillations_prevented': self.oscillations_prevented,
            'stability_locks': self.stability_locks,
            'frame_count': self.frame_count,
            'memory_efficiency': len(self.memory.feature_memory),
            '3d_pose_stats': pose_3d_stats,
            'current_3d_poses': len(self._current_poses_3d)
        }