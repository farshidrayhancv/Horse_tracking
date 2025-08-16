#!/usr/bin/env python3

import cv2
import numpy as np
import torch
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import supervision as sv
from collections import deque
import gc

# Fix torch.load issue
import torch.serialization
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

# Memory optimization
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAMURAI_AVAILABLE = True
except ImportError:
    SAMURAI_AVAILABLE = False
    print("❌ SAMURAI not available")

try:
    from boxmot import DeepOcSort
    DEEPOCSORT_AVAILABLE = True
except ImportError:
    DEEPOCSORT_AVAILABLE = False
    print("❌ DeepOCSORT not available")

from config import Config

class SAMURAIUnifiedTracker:
    def __init__(self, device=0, min_detections=3, iou_threshold=0.6, max_tracks=6, 
                 centroid_update_interval=30, use_mask_centroids=False):
        self.device = device
        self.min_detections = min_detections
        self.iou_threshold = iou_threshold
        self.max_tracks = max_tracks
        self.centroid_update_interval = centroid_update_interval
        self.use_mask_centroids = use_mask_centroids
        
        self.samurai_active = False
        self.active_tracks = {}
        self.detection_buffer = deque(maxlen=10)
        self.next_track_id = 0
        self.frames_since_activation = 0
        
        self.predictor = None
        
    def _get_predictor(self):
        """Lazy init single shared predictor"""
        if self.predictor is None:
            print("🔧 Initializing shared SAM2 predictor...")
            self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
            if torch.cuda.is_available():
                self.predictor.model.to("cuda")
            torch.cuda.empty_cache()
        return self.predictor
        
    def process_frame(self, frame_idx, frame, detections):
        self.detection_buffer.append({
            'frame_idx': frame_idx,
            'detections': detections,
            'centroids': self._compute_centroids(detections)
        })
        
        # Activate when threshold met
        if not self.samurai_active and len(detections) > self.min_detections:
            print(f"🚀 Activating SAMURAI at frame {frame_idx} with {len(detections)} detections")
            self.samurai_active = True
            self.frames_since_activation = 0
            self._initialize_tracks(frame, detections)
            
        if not self.samurai_active:
            return detections, {}
        
        self.frames_since_activation += 1
        
        should_update_centroids = (self.frames_since_activation % self.centroid_update_interval == 0)
        if should_update_centroids and len(detections) > 0:
            centroid_type = "mask" if self.use_mask_centroids else "bbox"
            print(f"🔄 Correcting SAMURAI centroids ({centroid_type}) at frame {frame_idx}")
            self._update_tracking_centroids(frame, detections)
            
        # Update existing tracks
        track_associations = {}
        lost_tracks = []
        
        predictor = self._get_predictor()
        predictor.set_image(frame)
        
        for track_id, track_data in self.active_tracks.items():
            # Use previous mask centroid as point prompt
            mask_point = track_data['last_point']
            
            masks, scores, _ = predictor.predict(
                point_coords=np.array([mask_point]),
                point_labels=np.array([1]),
                multimask_output=False
            )
            
            if len(masks) > 0 and scores[0] > 0.3:
                mask = masks[0]
                track_data['mask'] = mask
                track_data['last_point'] = self._get_mask_centroid(mask)
                track_data['lost_frames'] = 0
                
                # Find merged detections
                merged_indices = []
                for det_idx in range(len(detections)):
                    det_mask = self._bbox_to_mask(detections.xyxy[det_idx], frame.shape)
                    iou = self._compute_mask_iou(mask, det_mask)
                    if iou > self.iou_threshold:
                        merged_indices.append(det_idx)
                        
                if merged_indices:
                    track_associations[track_id] = merged_indices
            else:
                track_data['lost_frames'] += 1
                if track_data['lost_frames'] > 5:
                    lost_tracks.append(track_id)
                    
        # Remove lost tracks
        for track_id in lost_tracks:
            del self.active_tracks[track_id]
            print(f"❌ Lost track {track_id}")
            
        # Add new tracks if under limit
        if len(self.active_tracks) < self.max_tracks:
            all_associated = set()
            for indices in track_associations.values():
                all_associated.update(indices)
                
            unassociated = [i for i in range(len(detections)) if i not in all_associated]
            if unassociated:
                self._add_limited_tracks(frame, detections, unassociated)
                
        return self._create_unified_output(detections, track_associations), track_associations
    
    def _initialize_tracks(self, frame, detections):
        """Initialize up to max_tracks horses"""
        stable_points = self._find_stable_points()
        
        if not stable_points:
            stable_points = self._compute_centroids(detections)
            
        predictor = self._get_predictor()
        predictor.set_image(frame)
        
        # Only track first N horses
        for point in stable_points[:self.max_tracks]:
            masks, scores, _ = predictor.predict(
                point_coords=np.array([point]),
                point_labels=np.array([1]),
                multimask_output=False
            )
            
            if len(masks) > 0 and scores[0] > 0.5:
                self.active_tracks[self.next_track_id] = {
                    'mask': masks[0],
                    'last_point': point,
                    'lost_frames': 0,
                    'init_frame': frame.shape
                }
                self.next_track_id += 1
                print(f"✅ Initialized track {self.next_track_id-1}")
                
        torch.cuda.empty_cache()
                
    def _add_limited_tracks(self, frame, detections, unassociated_indices):
        """Add new tracks only if under limit"""
        slots_available = self.max_tracks - len(self.active_tracks)
        if slots_available <= 0:
            return
            
        predictor = self._get_predictor()
        predictor.set_image(frame)
        
        for idx in unassociated_indices[:slots_available]:
            centroid = self._get_detection_centroid(detections, idx)
            
            masks, scores, _ = predictor.predict(
                point_coords=np.array([centroid]),
                point_labels=np.array([1]),
                multimask_output=False
            )
            
            if len(masks) > 0 and scores[0] > 0.6:
                self.active_tracks[self.next_track_id] = {
                    'mask': masks[0],
                    'last_point': centroid,
                    'lost_frames': 0,
                    'init_frame': frame.shape
                }
                self.next_track_id += 1
                print(f"✅ Added new track {self.next_track_id-1}")
                
    def _update_tracking_centroids(self, frame, detections):
        """Correct tracking drift using bbox or mask centroids"""
        predictor = self._get_predictor()
        predictor.set_image(frame)
        
        if self.use_mask_centroids:
            correction_centroids = self._get_mask_centroids_from_detections(frame, detections)
        else:
            correction_centroids = self._compute_centroids(detections)
        
        updated_tracks = 0
        
        for track_id, track_data in self.active_tracks.items():
            current_point = track_data['last_point']
            
            closest_centroid = None
            min_distance = float('inf')
            
            for centroid in correction_centroids:
                distance = np.linalg.norm(current_point - centroid)
                if distance < min_distance and distance < 100:
                    min_distance = distance
                    closest_centroid = centroid
            
            if closest_centroid is not None:
                masks, scores, _ = predictor.predict(
                    point_coords=np.array([closest_centroid]),
                    point_labels=np.array([1]),
                    multimask_output=False
                )
                
                if len(masks) > 0 and scores[0] > 0.4:
                    track_data['last_point'] = closest_centroid
                    track_data['mask'] = masks[0]
                    updated_tracks += 1
        
        if updated_tracks > 0:
            print(f"✅ Corrected {updated_tracks} tracking centroids")
        else:
            print(f"⚠️  No centroids corrected - may indicate tracking degradation")
    def _get_mask_centroids_from_detections(self, frame, detections):
        """Generate masks from detections and extract precise centroids"""
        predictor = self._get_predictor()
        predictor.set_image(frame)
        
        mask_centroids = []
        
        for i in range(len(detections)):
            bbox_centroid = self._get_detection_centroid(detections, i)
            
            masks, scores, _ = predictor.predict(
                point_coords=np.array([bbox_centroid]),
                point_labels=np.array([1]),
                multimask_output=False
            )
            
            if len(masks) > 0 and scores[0] > 0.3:
                mask_centroid = self._get_mask_centroid(masks[0])
                mask_centroids.append(mask_centroid)
            else:
                mask_centroids.append(bbox_centroid)
        
        return mask_centroids
                
    def _get_mask_centroid(self, mask):
        """CRITICAL: Get centroid of mask - RESTORED METHOD"""
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0:
            return np.array([np.mean(x_indices), np.mean(y_indices)])
        return np.array([0, 0])
        
    def _find_stable_points(self):
        if len(self.detection_buffer) < 3:
            return []
            
        recent_frames = list(self.detection_buffer)[-3:]
        stable_points = []
        base_centroids = recent_frames[0]['centroids']
        
        for base_point in base_centroids:
            is_stable = True
            for frame_data in recent_frames[1:]:
                found_match = False
                for point in frame_data['centroids']:
                    if np.linalg.norm(base_point - point) < 50:
                        found_match = True
                        break
                if not found_match:
                    is_stable = False
                    break
                    
            if is_stable:
                stable_points.append(base_point)
                
        return stable_points
        
    def _compute_centroids(self, detections):
        centroids = []
        for i in range(len(detections)):
            box = detections.xyxy[i]
            centroid = np.array([
                (box[0] + box[2]) / 2,
                (box[1] + box[3]) / 2
            ])
            centroids.append(centroid)
        return centroids
        
    def _get_detection_centroid(self, detections, idx):
        box = detections.xyxy[idx]
        return np.array([
            (box[0] + box[2]) / 2,
            (box[1] + box[3]) / 2
        ])
        
    def _compute_mask_iou(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0
        return intersection / union
        
    def _bbox_to_mask(self, bbox, shape):
        mask = np.zeros(shape[:2], dtype=bool)
        x1, y1, x2, y2 = map(int, bbox)
        mask[y1:y2, x1:x2] = True
        return mask
        
    def _create_unified_output(self, original_detections, associations):
        if not associations:
            return original_detections
            
        unified_boxes = []
        unified_confidences = []
        unified_track_ids = []
        
        for track_id, detection_indices in associations.items():
            if not detection_indices:
                continue
                
            boxes = [original_detections.xyxy[i] for i in detection_indices]
            boxes_array = np.array(boxes)
            
            merged_box = [
                np.min(boxes_array[:, 0]),
                np.min(boxes_array[:, 1]),
                np.max(boxes_array[:, 2]),
                np.max(boxes_array[:, 3])
            ]
            
            unified_boxes.append(merged_box)
            unified_confidences.append(np.mean([original_detections.confidence[i] for i in detection_indices]))
            unified_track_ids.append(track_id)
            
        # Add untracked detections
        all_tracked = set()
        for indices in associations.values():
            all_tracked.update(indices)
            
        for i in range(len(original_detections)):
            if i not in all_tracked:
                unified_boxes.append(original_detections.xyxy[i])
                unified_confidences.append(original_detections.confidence[i])
                unified_track_ids.append(-1)  # Untracked
                
        if unified_boxes:
            return sv.Detections(
                xyxy=np.array(unified_boxes),
                confidence=np.array(unified_confidences),
                class_id=np.zeros(len(unified_boxes), dtype=np.int32),
                tracker_id=np.array(unified_track_ids)
            )
        return original_detections

class SupervisionVideoAnnotator:
    """SUPERVISION-CENTRIC ANNOTATION SYSTEM"""
    
    def __init__(self):
        # Supervision annotators - PROPER ABSTRACTION LAYER
        self.box_annotator = sv.BoundingBoxAnnotator(
            thickness=3,
        )
        
        self.label_annotator = sv.LabelAnnotator(
            text_thickness=2,
            text_scale=0.8,
            text_padding=5
        )
        
        self.trace_annotator = sv.TraceAnnotator(
            thickness=3,
            trace_length=30,
            position=sv.Position.CENTER
        )
        
        self.mask_annotator = sv.MaskAnnotator(
            opacity=0.4
        )
        
        # Track colors using supervision color palette
        self.color_palette = sv.ColorPalette.DEFAULT
        
        # Enhanced trail system
        self.track_trails = {}
        
    def annotate_frame(self, frame, detections, unified_tracker, frame_idx, 
                      samurai_active, samurai_merges):
        """SUPERVISION-POWERED COMPREHENSIVE ANNOTATION"""
        annotated_frame = frame.copy()
        
        # STEP 1: Draw SAMURAI masks using supervision
        if samurai_active and unified_tracker.active_tracks:
            annotated_frame = self._draw_samurai_masks_supervision(
                annotated_frame, unified_tracker
            )
        
        # STEP 2: Supervision trail annotation
        if hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
            annotated_frame = self.trace_annotator.annotate(
                scene=annotated_frame,
                detections=detections
            )
        
        # STEP 3: Supervision bounding box annotation
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        
        # STEP 4: Supervision label annotation
        labels = self._generate_labels(detections)
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )
        
        # STEP 5: Status overlay (OpenCV only for final display)
        annotated_frame = self._draw_status_overlay(
            annotated_frame, frame_idx, len(detections), 
            len(unified_tracker.active_tracks), samurai_active, samurai_merges
        )
        
        return annotated_frame
    
    def _draw_samurai_masks_supervision(self, frame, unified_tracker):
        """SUPERVISION MASK RENDERING - PROPER MULTI-MASK SUPPORT"""
        if not unified_tracker.active_tracks:
            return frame
        
        # Prepare masks for supervision batch processing
        masks = []
        colors = []
        
        for track_id, track_data in unified_tracker.active_tracks.items():
            raw_mask = track_data['mask']
            
            # Ensure proper boolean mask
            if raw_mask.dtype != bool:
                mask = raw_mask.astype(bool)
            else:
                mask = raw_mask
                
            if mask.shape[:2] != frame.shape[:2]:
                print(f"⚠️  Mask shape mismatch Track {track_id}: {mask.shape} vs {frame.shape[:2]}")
                continue
            
            masks.append(mask)
            colors.append(self.color_palette.by_idx(track_id))
        
        if not masks:
            return frame
        
        # Create supervision Detections for masks
        mask_detections = sv.Detections(
            xyxy=np.array([[0, 0, frame.shape[1], frame.shape[0]]] * len(masks)),
            mask=np.array(masks),
            class_id=np.arange(len(masks))
        )
        
        # Use supervision mask annotator for proper multi-mask rendering
        annotated_frame = self.mask_annotator.annotate(
            scene=frame,
            detections=mask_detections
        )
        
        # Add SAMURAI-specific centroids and labels
        for track_id, track_data in unified_tracker.active_tracks.items():
            centroid = track_data['last_point']
            color_bgr = self.color_palette.by_idx(track_id).as_bgr()
            
            # Draw track centroid
            cv2.circle(annotated_frame, tuple(map(int, centroid)), 8, color_bgr, -1)
            cv2.circle(annotated_frame, tuple(map(int, centroid)), 12, (255, 255, 255), 2)
            
            # Add SAMURAI track ID label
            cv2.putText(annotated_frame, f"S{track_id}", 
                       (int(centroid[0]) + 20, int(centroid[1]) - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return annotated_frame
    
    def _generate_labels(self, detections):
        """Generate supervision-compatible labels"""
        labels = []
        for i in range(len(detections)):
            confidence = detections.confidence[i]
            
            # Get track ID if available
            track_id = -1
            if hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
                track_id = detections.tracker_id[i]
            
            if track_id != -1:
                label = f"Horse ID:{track_id} ({confidence:.2f})"
            else:
                label = f"Horse ({confidence:.2f})"
                
            labels.append(label)
        
        return labels
    
    def _draw_status_overlay(self, frame, frame_idx, num_detections, num_samurai_tracks, 
                           samurai_active, total_merges):
        """Status information overlay (OpenCV for final display only)"""
        height, width = frame.shape[:2]
        
        # Status background
        cv2.rectangle(frame, (10, 10), (500, 140), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (500, 140), (255, 255, 255), 2)
        
        # Status text
        status_lines = [
            f"Frame: {frame_idx}",
            f"Detections: {num_detections}",
            f"SAMURAI: {'ACTIVE' if samurai_active else 'INACTIVE'} ({num_samurai_tracks}/8)",
            f"Total Merges: {total_merges}",
            f"Supervision: ENABLED"
        ]
        
        for i, line in enumerate(status_lines):
            y_pos = 40 + (i * 22)
            cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
        
        return frame

def _validate_detection_data(dets_np, frame_idx):
    """CRITICAL: Validate detection data integrity before tracker update"""
    if dets_np is None or len(dets_np) == 0:
        return True
    
    # Check for NaN or infinite values
    if np.any(np.isnan(dets_np)) or np.any(np.isinf(dets_np)):
        print(f"🚨 NaN/Inf values detected in detection data at frame {frame_idx}")
        return False
    
    # Validate bounding box coordinates
    for i, det in enumerate(dets_np):
        x1, y1, x2, y2 = det[:4]
        
        # Check for invalid box dimensions
        if x2 <= x1 or y2 <= y1:
            print(f"🚨 Invalid bounding box dimensions at frame {frame_idx}, detection {i}")
            return False
        
        # Check for reasonable coordinate ranges
        if x1 < 0 or y1 < 0 or x2 > 10000 or y2 > 10000:
            print(f"🚨 Extreme coordinate values at frame {frame_idx}, detection {i}")
            return False
    
    # Check confidence values
    confidences = dets_np[:, 4]
    if np.any(confidences < 0) or np.any(confidences > 1):
        print(f"🚨 Invalid confidence values at frame {frame_idx}")
        return False
    
    return True

def process_video_with_unified_tracking(config: Config, max_frames: int = 2000, 
                                      save_video: bool = True, output_video_path: str = None,
                                      use_mask_centroids: bool = False):
    cache_file = "detection_cache/horse_9_del_mar_pan_seg_mdxam_2_0.7_2100.json"
    
    print(f"Loading cached detections from {cache_file}")
    with open(cache_file, 'r') as f:
        cache_data = json.load(f)
    
    cached_detections = []
    for frame_data in cache_data['detections']:
        detection_info = frame_data['detections']
        
        if not detection_info['xyxy']:
            detections = sv.Detections.empty()
        else:
            detections = sv.Detections(
                xyxy=np.array(detection_info['xyxy'], dtype=np.float32),
                confidence=np.array(detection_info['confidence'], dtype=np.float32),
                class_id=np.array(detection_info['class_id'], dtype=np.int32)
            )
        cached_detections.append(detections)
    
    print(f"Loaded {len(cached_detections)} frames")
    
    unified_tracker = SAMURAIUnifiedTracker(
        device=config.device,
        max_tracks=6,
        centroid_update_interval=50,
        use_mask_centroids=use_mask_centroids
    )
    
    deepsort_tracker = DeepOcSort(
        reid_weights=Path('osnet_x0_25_msmt17.pt'),
        device='cuda:0',
        half=False,
        max_age=180,
        min_hits=5,
        det_thresh=0.5,
        iou_threshold=0.15,
        w_association_emb=0.98,
        embedding_off=False,
    )
    
    # Video input/output setup
    cap = cv2.VideoCapture(config.video_path)
    
    # Video writer setup
    video_writer = None
    if save_video:
        if output_video_path is None:
            output_video_path = f"supervision_samurai_tracking_{int(time.time())}.mp4"
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print(f"📹 Saving SUPERVISION-ENHANCED video to: {output_video_path}")
        print(f"   Resolution: {width}x{height} @ {fps}fps")
    
    annotator = SupervisionVideoAnnotator()
    
    track_history = {}
    samurai_merges = 0
    tracker_reinit_count = 0
    
    for frame_idx, detections in enumerate(cached_detections[:max_frames]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        # STEP 1: SAMURAI processing
        unified_detections, associations = unified_tracker.process_frame(frame_idx, frame, detections)
        
        if associations:
            samurai_merges += sum(len(v) > 1 for v in associations.values())
        
        if len(unified_detections) > 0:
            dets_np = np.column_stack((
                unified_detections.xyxy,
                unified_detections.confidence,
                np.zeros(len(unified_detections))
            )).astype(np.float64)
            
            # CRITICAL: Data validation before tracker update
            if not _validate_detection_data(dets_np, frame_idx):
                print(f"⚠️  Invalid detection data at frame {frame_idx}, skipping tracker update")
                unified_detections.tracker_id = np.array([-1] * len(unified_detections))
            else:
                try:
                    tracks = deepsort_tracker.update(dets_np, frame)
                    
                    if tracks is not None:
                        track_ids = []
                        for i, detection in enumerate(unified_detections.xyxy):
                            matched_track_id = -1
                            for track in tracks:
                                track_bbox = track[:4]
                                if _bbox_overlap(detection, track_bbox) > 0.5:
                                    matched_track_id = int(track[4])
                                    break
                            track_ids.append(matched_track_id)
                        
                        unified_detections.tracker_id = np.array(track_ids)
                        
                        for track in tracks:
                            track_id = int(track[4])
                            if track_id not in track_history:
                                track_history[track_id] = {'start': frame_idx, 'end': frame_idx}
                            track_history[track_id]['end'] = frame_idx
                    else:
                        unified_detections.tracker_id = np.array([-1] * len(unified_detections))
                        
                except Exception as e:
                    print(f"🚨 TRACKER STATE CORRUPTION at frame {frame_idx}: {str(e)}")
                    print(f"   Reinitializing DeepOCSORT tracker...")
                    tracker_reinit_count += 1
                    
                    deepsort_tracker = DeepOcSort(
                        reid_weights=Path('osnet_x0_25_msmt17.pt'),
                        device='cuda:0',
                        half=True
                    )
                    
                    unified_detections.tracker_id = np.array([-1] * len(unified_detections))
        else:
            unified_detections.tracker_id = np.array([])
        
        # STEP 3: SUPERVISION ANNOTATION
        if save_video:
            annotated_frame = annotator.annotate_frame(
                frame, unified_detections, unified_tracker, frame_idx,
                unified_tracker.samurai_active, samurai_merges
            )
            video_writer.write(annotated_frame)
                    
        if frame_idx % 100 == 0:
            active_samurai = len(unified_tracker.active_tracks)
            print(f"Frame {frame_idx}: {len(detections)}→{len(unified_detections)} detections, "
                  f"SAMURAI: {active_samurai}/8, Merges: {samurai_merges}")
            
            # Memory cleanup
            if frame_idx % 500 == 0:
                torch.cuda.empty_cache()
                gc.collect()
    
    cap.release()
    if video_writer:
        video_writer.release()
        print(f"✅ SUPERVISION VIDEO SAVED: {output_video_path}")
    
    long_tracks = {tid: data for tid, data in track_history.items() 
                   if data['end'] - data['start'] >= 1500}
    
    print(f"\n🏁 SUPERVISION-ENHANCED RESULTS:")
    print(f"Total tracks: {len(track_history)} → Long tracks: {len(long_tracks)}")
    print(f"SAMURAI merges: {samurai_merges}")
    print(f"Max concurrent SAMURAI: 8")
    
    if tracker_reinit_count > 0:
        print(f"🚨 TRACKER RECOVERY STATISTICS:")
        print(f"   State corruptions recovered: {tracker_reinit_count}")
        print(f"   System maintained stability through {tracker_reinit_count} tracker failures")
    else:
        print(f"✅ TRACKER STABILITY: Zero state corruptions detected")
    
    return track_history

def _bbox_overlap(bbox1, bbox2):
    """Calculate overlap between two bounding boxes"""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python sam_test.py config.yaml [max_frames] [--no-video] [--output-video path] [--mask-centroids]")
        sys.exit(1)
    
    config_file = sys.argv[1]
    max_frames = 2000
    save_video = True
    output_video_path = None
    use_mask_centroids = False
    
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg.isdigit():
            max_frames = int(arg)
        elif arg == '--no-video':
            save_video = False
        elif arg == '--output-video' and i + 1 < len(sys.argv):
            output_video_path = sys.argv[i + 1]
        elif arg == '--mask-centroids':
            use_mask_centroids = True
    
    config = Config(config_file)
    
    centroid_mode = "MASK" if use_mask_centroids else "BBOX"
    print(f"🔧 SUPERVISION-ENHANCED SAMURAI TRACKING")
    print(f"   Video: {config.video_path}")
    print(f"   Max Frames: {max_frames}")
    print(f"   Centroid Mode: {centroid_mode}")
    print(f"   Save Video: {save_video}")
    if save_video and output_video_path:
        print(f"   Output Video: {output_video_path}")
    
    track_data = process_video_with_unified_tracking(
        config, max_frames, save_video, output_video_path, use_mask_centroids
    )
    
    with open('supervision_unified_tracks.json', 'w') as f:
        json.dump(track_data, f, indent=2, default=str)
    
    print(f"✅ Saved SUPERVISION tracking results to supervision_unified_tracks.json")

if __name__ == "__main__":
    main()