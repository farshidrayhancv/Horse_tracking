"""
Simplified ReID Pipeline for Compound Racer Entities (Horse+Jockey)
Uses MegaDescriptor embeddings on compound crops with segmentation masks
"""

import torch
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch.nn.functional as F
from collections import deque, defaultdict
from PIL import Image

try:
    import supervision as sv
except ImportError:
    sv = None

try:
    from transformers import AutoModel, AutoProcessor
    MEGADESCRIPTOR_AVAILABLE = True
except ImportError:
    MEGADESCRIPTOR_AVAILABLE = False
    print("⚠️ MegaDescriptor not available - install: pip install transformers>=4.35.0")


class RacerMemory:
    """Memory system for compound racer entities"""
    
    def __init__(self, memory_size=15):
        self.memory_size = memory_size
        
        self.embeddings = defaultdict(deque)
        self.positions = defaultdict(deque)
        self.confidences = defaultdict(deque)
        self.frame_history = defaultdict(deque)
        
        # Racer statistics
        self.first_seen = {}
        self.last_seen = {}
        self.track_stability = defaultdict(float)
    
    def update_racer(self, track_id: int, embedding: np.ndarray, position: np.ndarray, 
                    confidence: float, frame_num: int):
        """Update memory for a racer"""
        if track_id not in self.first_seen:
            self.first_seen[track_id] = frame_num
        
        self.last_seen[track_id] = frame_num
        
        # Add to memory
        self.embeddings[track_id].append(embedding)
        self.positions[track_id].append(position)
        self.confidences[track_id].append(confidence)
        self.frame_history[track_id].append(frame_num)
        
        # Maintain memory size
        for memory_deque in [self.embeddings[track_id], self.positions[track_id],
                           self.confidences[track_id], self.frame_history[track_id]]:
            if len(memory_deque) > self.memory_size:
                memory_deque.popleft()
        
        # Update stability
        self._update_stability(track_id)
    
    def _update_stability(self, track_id: int):
        """Update track stability score"""
        if len(self.positions[track_id]) < 2:
            return
        
        # Calculate position variance
        positions = np.array(list(self.positions[track_id]))
        position_variance = np.var(np.diff(positions, axis=0))
        
        # Calculate confidence stability
        confidences = list(self.confidences[track_id])
        conf_stability = 1.0 - np.std(confidences) if len(confidences) > 1 else 0.5
        
        # Combine metrics
        motion_stability = 1.0 / (1.0 + position_variance / 1000.0)
        self.track_stability[track_id] = (motion_stability * 0.6 + conf_stability * 0.4)
    
    def get_recent_embeddings(self, track_id: int, n_recent: int = 3) -> List[np.ndarray]:
        """Get recent embeddings for a track"""
        if track_id not in self.embeddings:
            return []
        
        embeddings = list(self.embeddings[track_id])
        return embeddings[-n_recent:] if len(embeddings) >= n_recent else embeddings
    
    def predict_position(self, track_id: int) -> Optional[np.ndarray]:
        """Predict next position for a track"""
        if track_id not in self.positions or len(self.positions[track_id]) < 2:
            return None
        
        positions = list(self.positions[track_id])
        if len(positions) >= 2:
            velocity = positions[-1] - positions[-2]
            # Limit velocity to reasonable range
            speed = np.linalg.norm(velocity)
            if speed > 50.0:  # Max 50 pixels per frame
                velocity = velocity / speed * 50.0
            return positions[-1] + velocity
        
        return positions[-1]
    
    def cleanup_old_tracks(self, active_track_ids: set, max_age: int = 60):
        """Remove old inactive tracks"""
        to_remove = []
        current_frame = max(self.last_seen.values()) if self.last_seen else 0
        
        for track_id in list(self.embeddings.keys()):
            if track_id not in active_track_ids:
                if current_frame - self.last_seen.get(track_id, 0) > max_age:
                    to_remove.append(track_id)
        
        for track_id in to_remove:
            self._remove_track(track_id)
    
    def _remove_track(self, track_id: int):
        """Remove all data for a track"""
        for memory_dict in [self.embeddings, self.positions, self.confidences, self.frame_history]:
            memory_dict.pop(track_id, None)
        
        for info_dict in [self.first_seen, self.last_seen, self.track_stability]:
            info_dict.pop(track_id, None)


class SimplifiedReIDPipeline:
    """Simplified ReID Pipeline for Compound Racer Entities"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # Initialize MegaDescriptor
        self.megadescriptor_model = None
        self.megadescriptor_processor = None
        self.setup_megadescriptor()
        
        # Memory system
        self.memory = RacerMemory(
            memory_size=getattr(config, 'reid_memory_size', 15)
        )
        
        # Configuration
        self.similarity_threshold = getattr(config, 'reid_similarity_threshold', 0.4)
        self.reassignment_threshold = 0.6
        
        # Statistics
        self.reassignment_count = 0
        self.frame_count = 0
        
        print(f"🎯 Simplified ReID Pipeline initialized for compound racers")
        print(f"   MegaDescriptor: {'enabled' if self.megadescriptor_model else 'disabled'}")
    
    def setup_megadescriptor(self):
        """Initialize MegaDescriptor model"""
        if not MEGADESCRIPTOR_AVAILABLE:
            return
            
        try:
            self.megadescriptor_processor = AutoProcessor.from_pretrained("BVRA/MegaDescriptor-L-384")
            self.megadescriptor_model = AutoModel.from_pretrained("BVRA/MegaDescriptor-L-384")
            self.megadescriptor_model.to(self.device)
            self.megadescriptor_model.eval()
            print("✅ MegaDescriptor loaded for racer ReID")
        except Exception as e:
            print(f"❌ MegaDescriptor failed: {e}")
    
    def extract_racer_embedding(self, frame: np.ndarray, bbox: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Extract MegaDescriptor embedding from racer crop"""
        if not self.megadescriptor_model or not self.megadescriptor_processor:
            return np.random.rand(768) * 0.01  # Default embedding size
        
        try:
            # Extract crop
            x1, y1, x2, y2 = bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return np.random.rand(768) * 0.01
            
            crop = frame[y1:y2, x1:x2].copy()
            
            # Apply mask if available
            if mask is not None:
                mask_crop = mask[y1:y2, x1:x2]
                if mask_crop.shape == crop.shape[:2]:
                    # Set background to neutral gray
                    crop[~mask_crop] = [128, 128, 128]
            
            # Convert to PIL Image
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(crop_rgb)
            
            # Process with MegaDescriptor
            inputs = self.megadescriptor_processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.megadescriptor_model(**inputs)
                # Get pooled output
                if hasattr(outputs, 'pooler_output'):
                    features = outputs.pooler_output.cpu().numpy().flatten()
                elif hasattr(outputs, 'last_hidden_state'):
                    features = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
                else:
                    features = outputs[0].mean(dim=1).cpu().numpy().flatten()
            
            # Normalize
            norm = np.linalg.norm(features)
            if norm > 1e-8:
                features = features / norm
            else:
                features = np.random.rand(len(features)) * 0.01
            
            return features
            
        except Exception as e:
            print(f"❌ MegaDescriptor embedding extraction failed: {e}")
            return np.random.rand(768) * 0.01
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            # Check for invalid values
            if np.any(np.isnan(embedding1)) or np.any(np.isnan(embedding2)):
                return 0.0
            
            # Calculate norms
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 < 1e-8 or norm2 < 1e-8:
                return 0.0
            
            # Cosine similarity
            cos_sim = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            # Ensure valid range
            cos_sim = np.clip(cos_sim, -1.0, 1.0)
            
            return float(cos_sim)
            
        except Exception as e:
            print(f"❌ Similarity calculation failed: {e}")
            return 0.0
    
    def find_best_match(self, query_embedding: np.ndarray, query_position: np.ndarray, 
                       exclude_track_id: int, active_track_ids: set) -> Tuple[int, float]:
        """Find best matching track from memory"""
        best_track_id = -1
        best_score = 0.0
        
        for track_id in self.memory.embeddings.keys():
            if track_id == exclude_track_id or track_id in active_track_ids:
                continue
            
            # Get recent embeddings
            recent_embeddings = self.memory.get_recent_embeddings(track_id, n_recent=3)
            if not recent_embeddings:
                continue
            
            # Calculate visual similarity
            visual_similarities = [self.calculate_similarity(query_embedding, emb) 
                                 for emb in recent_embeddings]
            best_visual_sim = max(visual_similarities)
            
            # Calculate motion consistency
            predicted_pos = self.memory.predict_position(track_id)
            if predicted_pos is not None:
                distance = np.linalg.norm(query_position - predicted_pos)
                motion_score = 1.0 / (1.0 + distance / 100.0)  # 100 pixel threshold
            else:
                motion_score = 0.5
            
            # Track stability bonus
            stability = self.memory.track_stability.get(track_id, 0.0)
            
            # Combined score
            combined_score = (best_visual_sim * 0.7 + motion_score * 0.2 + stability * 0.1)
            
            if combined_score > best_score and best_visual_sim > self.reassignment_threshold:
                best_score = combined_score
                best_track_id = track_id
        
        return best_track_id, best_score
    
    def enhance_tracking(self, detections, masks: List[np.ndarray] = None, frame: np.ndarray = None) -> 'sv.Detections':
        """Main ReID enhancement for compound racer tracking"""
        if not sv or len(detections) == 0:
            return detections
        
        if not hasattr(detections, 'tracker_id'):
            return detections
        
        self.frame_count += 1
        
        # Extract embeddings for all detections
        embeddings = []
        for i, bbox in enumerate(detections.xyxy):
            mask = masks[i] if masks and i < len(masks) else None
            embedding = self.extract_racer_embedding(frame, bbox, mask)
            embeddings.append(embedding)
        
        # Copy detections for modification
        enhanced_detections = sv.Detections(
            xyxy=detections.xyxy.copy(),
            confidence=detections.confidence.copy() if hasattr(detections, 'confidence') else None,
            class_id=detections.class_id.copy() if hasattr(detections, 'class_id') else None,
            tracker_id=detections.tracker_id.copy()
        )
        
        # Update memory and perform reassignments
        active_track_ids = set()
        reassignments_this_frame = 0
        
        for i, (bbox, track_id, embedding) in enumerate(zip(detections.xyxy, detections.tracker_id, embeddings)):
            if track_id < 0:
                continue
            
            center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            confidence = detections.confidence[i] if hasattr(detections, 'confidence') else 0.8
            
            # Check if this is a new or unstable track that should be reassigned
            should_reassign = False
            
            # New track check (first 5 frames)
            if track_id not in self.memory.first_seen:
                should_reassign = True
            elif self.frame_count - self.memory.first_seen[track_id] <= 5:
                should_reassign = True
            # Unstable track check
            elif self.memory.track_stability.get(track_id, 0.0) < 0.3:
                should_reassign = True
            
            if should_reassign:
                # Find best match from memory
                best_match_id, best_score = self.find_best_match(
                    embedding, center, exclude_track_id=track_id, 
                    active_track_ids=active_track_ids
                )
                
                if best_match_id >= 0:
                    enhanced_detections.tracker_id[i] = best_match_id
                    active_track_ids.add(best_match_id)
                    reassignments_this_frame += 1
                    self.reassignment_count += 1
                    print(f"🔄 ReID: Racer #{track_id} → #{best_match_id} (score: {best_score:.3f})")
                else:
                    active_track_ids.add(track_id)
            else:
                active_track_ids.add(track_id)
            
            # Update memory
            final_track_id = enhanced_detections.tracker_id[i]
            self.memory.update_racer(final_track_id, embedding, center, confidence, self.frame_count)
        
        # Cleanup old tracks
        self.memory.cleanup_old_tracks(active_track_ids)
        
        if reassignments_this_frame > 0:
            print(f"📊 Frame {self.frame_count}: {reassignments_this_frame} racer reassignments")
        
        return enhanced_detections
    
    def get_tracking_info(self) -> Dict:
        """Get ReID tracking statistics"""
        return {
            'active_tracks': len(self.memory.embeddings),
            'total_reassignments': self.reassignment_count,
            'frame_count': self.frame_count,
            'memory_tracks': list(self.memory.embeddings.keys())
        }