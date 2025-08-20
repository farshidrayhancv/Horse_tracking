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

from config import Config

class SAMURAIPrimaryTracker:
    """ARCHITECTURE 1: SAMURAI-CENTRIC TRACKING SYSTEM"""
    
    def __init__(self, device=0, min_detections=4, max_tracks=6, 
                 detection_reinit_interval=100, use_mask_centroids=False):
        self.device = device
        self.min_detections = min_detections
        self.max_tracks = max_tracks  # Fixed 6 horses for racing
        self.detection_reinit_interval = detection_reinit_interval
        self.use_mask_centroids = use_mask_centroids
        
        self.samurai_active = False
        self.active_tracks = {}  # track_id -> track_data
        self.detection_buffer = deque(maxlen=10)
        self.next_track_id = 0
        self.frames_since_activation = 0
        
        self.predictor = None
        
        # ARCHITECTURE 1 SPECIFIC
        self.lost_track_positions = {}  # Store positions of recently lost tracks
        self.track_confidence_history = {}  # Track confidence over time
        
    def _get_predictor(self):
        """Lazy init single shared predictor"""
        if self.predictor is None:
            print("🔧 Initializing SAMURAI predictor for PRIMARY tracking...")
            self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
            if torch.cuda.is_available():
                self.predictor.model.to("cuda")
            torch.cuda.empty_cache()
        return self.predictor
        
    def process_frame(self, frame_idx, frame, detections):
        """ARCHITECTURE 1: SAMURAI PRIMARY with detection backup"""
        self.detection_buffer.append({
            'frame_idx': frame_idx,
            'detections': detections,
            'centroids': self._compute_centroids(detections)
        })
        
        # COLD START: Bootstrap SAMURAI with detections
        if not self.samurai_active and len(detections) >= self.min_detections:
            print(f"🚀 COLD START: Bootstrapping SAMURAI PRIMARY at frame {frame_idx}")
            self.samurai_active = True
            self.frames_since_activation = 0
            self._bootstrap_primary_tracks(frame, detections)
            
        if not self.samurai_active:
            # Before activation, return detections as-is
            return self._create_detection_output(detections), {}
            
        self.frames_since_activation += 1
        
        # SAMURAI PRIMARY TRACKING
        samurai_detections = self._run_primary_tracking(frame, frame_idx)
        
        # BACKUP REINITIALIZATION: Only when tracks are lost
        if len(self.active_tracks) < self.max_tracks and len(detections) > 0:
            self._reinitialize_lost_tracks(frame, detections, frame_idx)
            
        # OPPORTUNISTIC ENHANCEMENT: Optional detection-based improvements
        should_opportunistic_update = (self.frames_since_activation % self.detection_reinit_interval == 0)
        if should_opportunistic_update and len(detections) > 2:
            self._opportunistic_track_enhancement(frame, detections, frame_idx)
            
        return samurai_detections, {"samurai_primary": len(self.active_tracks)}
    
    def _bootstrap_primary_tracks(self, frame, detections):
        """COLD START: Initialize exactly max_tracks from detections"""
        predictor = self._get_predictor()
        predictor.set_image(frame)
        
        # Use most stable detection centroids for bootstrapping
        stable_points = self._find_stable_points()
        if not stable_points:
            stable_points = self._compute_centroids(detections)
        
        bootstrap_points = stable_points[:self.max_tracks]
        
        for i, point in enumerate(bootstrap_points):
            masks, scores, _ = predictor.predict(
                point_coords=np.array([point]),
                point_labels=np.array([1]),
                multimask_output=False
            )
            
            if len(masks) > 0 and scores[0] > 0.4:
                self.active_tracks[self.next_track_id] = {
                    'mask': masks[0],
                    'last_point': point,
                    'lost_frames': 0,
                    'confidence': scores[0],
                    'birth_frame': self.frames_since_activation
                }
                self.track_confidence_history[self.next_track_id] = [scores[0]]
                self.next_track_id += 1
                print(f"✅ BOOTSTRAP: Primary track {self.next_track_id-1} initialized")
                
        print(f"🎯 SAMURAI PRIMARY: {len(self.active_tracks)}/{self.max_tracks} tracks active")
        torch.cuda.empty_cache()
    
    def _run_primary_tracking(self, frame, frame_idx):
        """SAMURAI PRIMARY: Autonomous tracking without detection dependency"""
        predictor = self._get_predictor()
        predictor.set_image(frame)
        
        active_detections = []
        lost_tracks = []
        
        for track_id, track_data in self.active_tracks.items():
            mask_point = track_data['last_point']
            
            masks, scores, _ = predictor.predict(
                point_coords=np.array([mask_point]),
                point_labels=np.array([1]),
                multimask_output=False
            )
            
            if len(masks) > 0 and scores[0] > 0.2:  # Lower threshold for primary tracking
                mask = masks[0]
                track_data['mask'] = mask
                track_data['last_point'] = self._get_mask_centroid(mask)
                track_data['confidence'] = scores[0]
                track_data['lost_frames'] = 0
                
                # Update confidence history
                if track_id in self.track_confidence_history:
                    self.track_confidence_history[track_id].append(scores[0])
                    if len(self.track_confidence_history[track_id]) > 10:
                        self.track_confidence_history[track_id].pop(0)
                
                # Create detection from SAMURAI track
                bbox = self._mask_to_bbox(mask)
                active_detections.append({
                    'bbox': bbox,
                    'confidence': scores[0],
                    'track_id': track_id,
                    'mask': mask
                })
                
            else:
                track_data['lost_frames'] += 1
                if track_data['lost_frames'] > 8:  # Longer patience for primary tracking
                    self.lost_track_positions[track_id] = track_data['last_point']
                    lost_tracks.append(track_id)
                    print(f"💀 LOST: Primary track {track_id} after {track_data['lost_frames']} frames")
                    
        # Remove definitively lost tracks
        for track_id in lost_tracks:
            del self.active_tracks[track_id]
            
        return self._create_samurai_detections(active_detections)
    
    def _reinitialize_lost_tracks(self, frame, detections, frame_idx):
        """BACKUP REINITIALIZATION: Use detections only to replace lost tracks"""
        if len(self.active_tracks) >= self.max_tracks:
            return
            
        slots_needed = self.max_tracks - len(self.active_tracks)
        print(f"🔧 REINIT: Need {slots_needed} tracks, have {len(detections)} detections")
        
        predictor = self._get_predictor()
        predictor.set_image(frame)
        
        # Try to reinitialize near lost track positions first
        reinitialized = 0
        
        if self.use_mask_centroids and len(detections) > 0:
            candidate_centroids = self._get_mask_centroids_from_detections(frame, detections)
        else:
            candidate_centroids = self._compute_centroids(detections)
        
        # Prioritize positions near recently lost tracks
        reinit_candidates = []
        
        for centroid in candidate_centroids[:slots_needed * 2]:  # More candidates than slots
            too_close_to_existing = False
            
            # Don't reinitialize too close to existing tracks
            for track_data in self.active_tracks.values():
                if np.linalg.norm(centroid - track_data['last_point']) < 80:
                    too_close_to_existing = True
                    break
                    
            if not too_close_to_existing:
                reinit_candidates.append(centroid)
        
        for centroid in reinit_candidates[:slots_needed]:
            masks, scores, _ = predictor.predict(
                point_coords=np.array([centroid]),
                point_labels=np.array([1]),
                multimask_output=False
            )
            
            if len(masks) > 0 and scores[0] > 0.5:  # Higher threshold for reinitialization
                self.active_tracks[self.next_track_id] = {
                    'mask': masks[0],
                    'last_point': centroid,
                    'lost_frames': 0,
                    'confidence': scores[0],
                    'birth_frame': self.frames_since_activation
                }
                self.track_confidence_history[self.next_track_id] = [scores[0]]
                self.next_track_id += 1
                reinitialized += 1
                print(f"🔄 REINIT: Track {self.next_track_id-1} reinitialized")
                
        if reinitialized > 0:
            print(f"✅ REINIT SUCCESS: {reinitialized} tracks restored, {len(self.active_tracks)}/{self.max_tracks} active")
    
    def _opportunistic_track_enhancement(self, frame, detections, frame_idx):
        """OPPORTUNISTIC: Improve existing tracks when high-quality detections available"""
        if len(detections) < 3:  # Only enhance when we have good detection coverage
            return
            
        enhanced_count = 0
        enhancement_type = "mask" if self.use_mask_centroids else "bbox"
        
        print(f"🔍 OPPORTUNISTIC: Enhancing tracks with {enhancement_type} centroids")
        
        predictor = self._get_predictor()
        predictor.set_image(frame)
        
        if self.use_mask_centroids:
            detection_centroids = self._get_mask_centroids_from_detections(frame, detections)
        else:
            detection_centroids = self._compute_centroids(detections)
        
        for track_id, track_data in self.active_tracks.items():
            current_point = track_data['last_point']
            
            # Find nearby detection centroid (not mandatory)
            closest_centroid = None
            min_distance = float('inf')
            
            for centroid in detection_centroids:
                distance = np.linalg.norm(current_point - centroid)
                if distance < min_distance and distance < 120:  # Relaxed threshold
                    min_distance = distance
                    closest_centroid = centroid
            
            if closest_centroid is not None:
                # Only enhance if it improves confidence
                masks, scores, _ = predictor.predict(
                    point_coords=np.array([closest_centroid]),
                    point_labels=np.array([1]),
                    multimask_output=False
                )
                
                if len(masks) > 0 and scores[0] > track_data['confidence']:
                    track_data['last_point'] = closest_centroid
                    track_data['mask'] = masks[0]
                    track_data['confidence'] = scores[0]
                    enhanced_count += 1
        
        if enhanced_count > 0:
            print(f"✨ ENHANCED: {enhanced_count} tracks improved opportunistically")
        else:
            print(f"➡️  ENHANCEMENT: No improvements needed - SAMURAI tracking well")
    
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
        """Get centroid of mask"""
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0:
            return np.array([np.mean(x_indices), np.mean(y_indices)])
        return np.array([0, 0])
        
    def _mask_to_bbox(self, mask):
        """Convert mask to bounding box"""
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0:
            return [np.min(x_indices), np.min(y_indices), np.max(x_indices), np.max(y_indices)]
        return [0, 0, 10, 10]
        
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
    
    def _create_samurai_detections(self, active_detections):
        """Convert SAMURAI tracks to supervision Detections format"""
        if not active_detections:
            return sv.Detections.empty()
        
        boxes = []
        confidences = []
        track_ids = []
        masks = []
        
        for det in active_detections:
            boxes.append(det['bbox'])
            confidences.append(det['confidence'])
            track_ids.append(det['track_id'])
            masks.append(det['mask'])
        
        return sv.Detections(
            xyxy=np.array(boxes),
            confidence=np.array(confidences),
            class_id=np.zeros(len(boxes), dtype=np.int32),
            tracker_id=np.array(track_ids),
            mask=np.array(masks) if masks else None
        )
    
    def _create_detection_output(self, detections):
        """Before SAMURAI activation, pass through detections with no tracker_id"""
        if len(detections) == 0:
            detections.tracker_id = np.array([])
        else:
            detections.tracker_id = np.array([-1] * len(detections))
        return detections

class SupervisionVideoAnnotator:
    """SUPERVISION-CENTRIC ANNOTATION FOR ARCHITECTURE 1"""
    
    def __init__(self):
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
            trace_length=50,  # Longer trails for primary tracking
            position=sv.Position.CENTER
        )
        
        self.mask_annotator = sv.MaskAnnotator(
            opacity=0.3
        )
        
        self.color_palette = sv.ColorPalette.DEFAULT
        
    def annotate_frame(self, frame, detections, samurai_tracker, frame_idx, stats):
        """ARCHITECTURE 1 ANNOTATION: Emphasize SAMURAI primary tracking"""
        annotated_frame = frame.copy()
        
        # SAMURAI masks (primary visualization)
        if samurai_tracker.samurai_active and samurai_tracker.active_tracks:
            annotated_frame = self._draw_primary_samurai_masks(
                annotated_frame, samurai_tracker
            )
        
        # Supervision trail annotation
        if hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
            annotated_frame = self.trace_annotator.annotate(
                scene=annotated_frame,
                detections=detections
            )
        
        # Supervision bounding box annotation
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        
        # Enhanced labels for Architecture 1
        labels = self._generate_primary_labels(detections)
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )
        
        # Architecture 1 status overlay
        annotated_frame = self._draw_architecture1_status(
            annotated_frame, frame_idx, len(detections), 
            len(samurai_tracker.active_tracks), samurai_tracker.samurai_active, stats
        )
        
        return annotated_frame
    
    def _draw_primary_samurai_masks(self, frame, samurai_tracker):
        """Render SAMURAI masks as primary tracking indicators"""
        if not samurai_tracker.active_tracks:
            return frame
        
        masks = []
        colors = []
        
        for track_id, track_data in samurai_tracker.active_tracks.items():
            raw_mask = track_data['mask']
            
            if raw_mask.dtype != bool:
                mask = raw_mask.astype(bool)
            else:
                mask = raw_mask
                
            if mask.shape[:2] != frame.shape[:2]:
                continue
            
            masks.append(mask)
            colors.append(self.color_palette.by_idx(track_id))
        
        if not masks:
            return frame
        
        mask_detections = sv.Detections(
            xyxy=np.array([[0, 0, frame.shape[1], frame.shape[0]]] * len(masks)),
            mask=np.array(masks),
            class_id=np.arange(len(masks))
        )
        
        annotated_frame = self.mask_annotator.annotate(
            scene=frame,
            detections=mask_detections
        )
        
        # Primary track indicators
        for track_id, track_data in samurai_tracker.active_tracks.items():
            centroid = track_data['last_point']
            confidence = track_data['confidence']
            color_bgr = self.color_palette.by_idx(track_id).as_bgr()
            
            # Larger indicators for primary tracks
            cv2.circle(annotated_frame, tuple(map(int, centroid)), 12, color_bgr, -1)
            cv2.circle(annotated_frame, tuple(map(int, centroid)), 16, (255, 255, 255), 3)
            
            # Primary track label
            cv2.putText(annotated_frame, f"P{track_id}", 
                       (int(centroid[0]) + 25, int(centroid[1]) - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            
            # Confidence indicator
            cv2.putText(annotated_frame, f"{confidence:.2f}", 
                       (int(centroid[0]) + 25, int(centroid[1]) + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def _generate_primary_labels(self, detections):
        """Generate labels emphasizing primary tracking"""
        labels = []
        for i in range(len(detections)):
            confidence = detections.confidence[i]
            
            track_id = -1
            if hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
                track_id = detections.tracker_id[i]
            
            if track_id != -1:
                label = f"PRIMARY-{track_id} ({confidence:.2f})"
            else:
                label = f"DETECTION ({confidence:.2f})"
                
            labels.append(label)
        
        return labels
    
    def _draw_architecture1_status(self, frame, frame_idx, num_detections, 
                                  num_primary_tracks, samurai_active, stats):
        """Architecture 1 specific status display"""
        height, width = frame.shape[:2]
        
        cv2.rectangle(frame, (10, 10), (600, 160), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (600, 160), (255, 255, 255), 2)
        
        status_lines = [
            f"ARCHITECTURE 1: SAMURAI PRIMARY",
            f"Frame: {frame_idx}",
            f"Detections Available: {num_detections}",
            f"SAMURAI PRIMARY: {'ACTIVE' if samurai_active else 'BOOTSTRAPPING'} ({num_primary_tracks}/6)",
            f"System: Detection-Independent Tracking",
            f"Status: {'TRACKING' if samurai_active else 'WAITING FOR BOOTSTRAP'}"
        ]
        
        for i, line in enumerate(status_lines):
            y_pos = 35 + (i * 20)
            color = (0, 255, 0) if samurai_active else (255, 255, 0)
            cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
        
        return frame

def process_video_architecture1(config: Config, max_frames: int = 2000, 
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
    
    # ARCHITECTURE 1: SAMURAI PRIMARY TRACKER
    samurai_tracker = SAMURAIPrimaryTracker(
        device=config.device,
        max_tracks=6,  # Racing: exactly 6 horses
        detection_reinit_interval=100,  # Opportunistic enhancement every 100 frames
        use_mask_centroids=use_mask_centroids
    )
    
    # Video setup
    cap = cv2.VideoCapture(config.video_path)
    
    video_writer = None
    if save_video:
        if output_video_path is None:
            output_video_path = f"architecture1_samurai_primary_{int(time.time())}.mp4"
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print(f"📹 Saving ARCHITECTURE 1 video to: {output_video_path}")
        print(f"   Resolution: {width}x{height} @ {fps}fps")
    
    annotator = SupervisionVideoAnnotator()
    
    track_history = {}
    reinitialization_events = 0
    enhancement_events = 0
    
    for frame_idx, detections in enumerate(cached_detections[:max_frames]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        # ARCHITECTURE 1: SAMURAI PRIMARY PROCESSING
        samurai_detections, stats = samurai_tracker.process_frame(frame_idx, frame, detections)
        
        # Track history for analysis
        if hasattr(samurai_detections, 'tracker_id') and samurai_detections.tracker_id is not None:
            for i, track_id in enumerate(samurai_detections.tracker_id):
                if track_id != -1:
                    if track_id not in track_history:
                        track_history[track_id] = {'start': frame_idx, 'end': frame_idx}
                    track_history[track_id]['end'] = frame_idx
        
        # Video annotation
        if save_video:
            annotated_frame = annotator.annotate_frame(
                frame, samurai_detections, samurai_tracker, frame_idx, stats
            )
            video_writer.write(annotated_frame)
                    
        if frame_idx % 100 == 0:
            active_primary = len(samurai_tracker.active_tracks)
            print(f"Frame {frame_idx}: {len(detections)} detections → "
                  f"SAMURAI PRIMARY: {active_primary}/6 tracks")
            
            if frame_idx % 500 == 0:
                torch.cuda.empty_cache()
                gc.collect()
    
    cap.release()
    if video_writer:
        video_writer.release()
        print(f"✅ ARCHITECTURE 1 VIDEO SAVED: {output_video_path}")
    
    long_tracks = {tid: data for tid, data in track_history.items() 
                   if data['end'] - data['start'] >= 1500}
    
    print(f"\n🏁 ARCHITECTURE 1 RESULTS:")
    print(f"Total tracks: {len(track_history)} → Long tracks: {len(long_tracks)}")
    print(f"SAMURAI PRIMARY: Detection-independent tracking")
    print(f"System reliability: {'HIGH' if len(long_tracks) >= 4 else 'NEEDS TUNING'}")
    
    return track_history

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
    print(f"🏎️  ARCHITECTURE 1: SAMURAI PRIMARY TRACKING")
    print(f"   Video: {config.video_path}")
    print(f"   Max Frames: {max_frames}")
    print(f"   Centroid Mode: {centroid_mode}")
    print(f"   Racing Mode: 6 horses fixed count")
    print(f"   Save Video: {save_video}")
    
    track_data = process_video_architecture1(
        config, max_frames, save_video, output_video_path, use_mask_centroids
    )
    
    with open('architecture1_tracks.json', 'w') as f:
        json.dump(track_data, f, indent=2, default=str)
    
    print(f"✅ Saved ARCHITECTURE 1 results to architecture1_tracks.json")

if __name__ == "__main__":
    main()