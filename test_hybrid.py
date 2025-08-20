#!/usr/bin/env python3

import cv2
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import supervision as sv
from collections import deque
import gc

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("❌ YOLO not available")

try:
    from boxmot import DeepOcSort
    DEEPOCSORT_AVAILABLE = True
except ImportError:
    DEEPOCSORT_AVAILABLE = False
    print("❌ DeepOCSORT not available")

from config import Config

class SegmentationObjectDetectionTracker:
    def __init__(self, yolo_model_path="yolo11s.pt", confidence_threshold=0.5, max_horses=6):
        self.confidence_threshold = confidence_threshold
        self.max_horses = max_horses
        
        if YOLO_AVAILABLE:
            print(f"🔧 Loading YOLO11s model: {yolo_model_path}")
            self.yolo_model = YOLO(yolo_model_path)
            print("✅ YOLO11s model loaded successfully")
        else:
            raise ImportError("YOLO not available - install ultralytics")
        
        self.grouping_stats = {
            'total_frames': 0,
            'successful_groupings': 0,
            'yolo_detections': 0,
            'segmentation_blobs': 0
        }
        
    def process_frame(self, frame, cached_segmentation_detections):
        self.grouping_stats['total_frames'] += 1
        
        yolo_detections = self._run_yolo_detection(frame)
        self.grouping_stats['yolo_detections'] += len(yolo_detections)
        
        segmentation_detections = cached_segmentation_detections
        self.grouping_stats['segmentation_blobs'] += len(segmentation_detections)
        
        grouped_detections = self._group_segmentation_to_yolo(
            yolo_detections, segmentation_detections
        )
        
        if len(grouped_detections) > 0:
            self.grouping_stats['successful_groupings'] += 1
        
        return grouped_detections
    
    def _run_yolo_detection(self, frame):
        results = self.yolo_model(frame, verbose=False, conf=self.confidence_threshold)
        
        if not results or len(results) == 0:
            return sv.Detections.empty()
        
        result = results[0]
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            detections = sv.Detections(
                xyxy=boxes,
                confidence=confidences,
                class_id=class_ids
            )
            
            if len(detections) > self.max_horses:
                top_indices = np.argsort(detections.confidence)[-self.max_horses:]
                detections = detections[top_indices]
            
            return detections
        
        return sv.Detections.empty()
    
    def _group_segmentation_to_yolo(self, yolo_detections, segmentation_detections):
        if len(yolo_detections) == 0 or len(segmentation_detections) == 0:
            return yolo_detections
        
        grouped_detections = []
        
        for i, yolo_bbox in enumerate(yolo_detections.xyxy):
            overlapping_blobs = []
            overlap_confidences = []
            
            for j, seg_bbox in enumerate(segmentation_detections.xyxy):
                overlap_ratio = self._calculate_bbox_overlap(yolo_bbox, seg_bbox)
                
                if overlap_ratio > 0.1:
                    overlapping_blobs.append(seg_bbox)
                    overlap_confidences.append(segmentation_detections.confidence[j])
            
            if overlapping_blobs:
                combined_confidence = np.mean(overlap_confidences) if overlap_confidences else yolo_detections.confidence[i]
                
                grouped_detections.append({
                    'bbox': yolo_bbox,
                    'confidence': combined_confidence,
                    'yolo_confidence': yolo_detections.confidence[i],
                    'segmentation_blobs': len(overlapping_blobs),
                    'combined_centroid': self._calculate_grouped_centroid(yolo_bbox, overlapping_blobs)
                })
            else:
                grouped_detections.append({
                    'bbox': yolo_bbox,
                    'confidence': yolo_detections.confidence[i],
                    'yolo_confidence': yolo_detections.confidence[i],
                    'segmentation_blobs': 0,
                    'combined_centroid': self._bbox_centroid(yolo_bbox)
                })
        
        if grouped_detections:
            boxes = np.array([det['bbox'] for det in grouped_detections])
            confidences = np.array([det['confidence'] for det in grouped_detections])
            class_ids = np.zeros(len(grouped_detections), dtype=np.int32)
            
            return sv.Detections(
                xyxy=boxes,
                confidence=confidences,
                class_id=class_ids
            )
        
        return sv.Detections.empty()
    
    def _calculate_bbox_overlap(self, bbox1, bbox2):
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
    
    def _calculate_grouped_centroid(self, yolo_bbox, segmentation_blobs):
        if not segmentation_blobs:
            return self._bbox_centroid(yolo_bbox)
        
        total_area = 0
        weighted_x = 0
        weighted_y = 0
        
        for blob_bbox in segmentation_blobs:
            area = (blob_bbox[2] - blob_bbox[0]) * (blob_bbox[3] - blob_bbox[1])
            centroid_x = (blob_bbox[0] + blob_bbox[2]) / 2
            centroid_y = (blob_bbox[1] + blob_bbox[3]) / 2
            
            weighted_x += centroid_x * area
            weighted_y += centroid_y * area
            total_area += area
        
        if total_area > 0:
            return np.array([weighted_x / total_area, weighted_y / total_area])
        else:
            return self._bbox_centroid(yolo_bbox)
    
    def _bbox_centroid(self, bbox):
        return np.array([
            (bbox[0] + bbox[2]) / 2,
            (bbox[1] + bbox[3]) / 2
        ])
    
    def get_statistics(self):
        stats = self.grouping_stats.copy()
        if stats['total_frames'] > 0:
            stats['grouping_success_rate'] = stats['successful_groupings'] / stats['total_frames']
            stats['avg_yolo_per_frame'] = stats['yolo_detections'] / stats['total_frames']
            stats['avg_segmentation_per_frame'] = stats['segmentation_blobs'] / stats['total_frames']
        return stats

class SupervisionVideoAnnotator:
    def __init__(self):
        self.box_annotator = sv.BoundingBoxAnnotator(
            thickness=3,
        )
        
        self.label_annotator = sv.LabelAnnotator(
            text_thickness=2,
            text_scale=0.8,
            text_padding=5
        )
        
        self.color_palette = sv.ColorPalette.DEFAULT
        
        self.detection_smoother = sv.DetectionsSmoother()
        
    def annotate_frame(self, frame, detections, cached_segmentation_detections, frame_idx, stats):
        annotated_frame = frame.copy()
        
        smoothed_detections = self.detection_smoother.update_with_detections(
            detections=detections
        )
        
        annotated_frame = self._draw_segmentation_masks(
            annotated_frame, smoothed_detections, cached_segmentation_detections
        )
        
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame,
            detections=smoothed_detections
        )
        
        labels = self._generate_labels(smoothed_detections)
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=smoothed_detections,
            labels=labels
        )
        
        annotated_frame = self._draw_status(
            annotated_frame, frame_idx, len(smoothed_detections), stats
        )
        
        return annotated_frame
    
    def _draw_segmentation_masks(self, frame, detections, cached_segmentation_detections):
        if len(detections) == 0 or len(cached_segmentation_detections) == 0:
            return frame
        
        overlay = frame.copy()
        
        for i, detection_bbox in enumerate(detections.xyxy):
            track_id = -1
            if hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
                track_id = detections.tracker_id[i]
            
            if track_id == -1:
                continue
                
            color_bgr = self.color_palette.by_idx(track_id).as_bgr()
            
            for j, seg_bbox in enumerate(cached_segmentation_detections.xyxy):
                overlap_ratio = self._calculate_bbox_overlap(detection_bbox, seg_bbox)
                
                if overlap_ratio > 0.1:
                    x1, y1, x2, y2 = map(int, seg_bbox)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)
            
            centroid = np.array([
                (detection_bbox[0] + detection_bbox[2]) / 2,
                (detection_bbox[1] + detection_bbox[3]) / 2
            ])
            
            cv2.circle(overlay, tuple(map(int, centroid)), 12, color_bgr, -1)
            cv2.circle(overlay, tuple(map(int, centroid)), 16, (255, 255, 255), 3)
            
            cv2.putText(overlay, f"{track_id}", 
                       (int(centroid[0]) + 25, int(centroid[1]) - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_bgr, 3)
        
        result = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        return result
    
    def _calculate_bbox_overlap(self, bbox1, bbox2):
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
    
    def _generate_labels(self, detections):
        labels = []
        for i in range(len(detections)):
            confidence = detections.confidence[i]
            
            track_id = -1
            if hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
                track_id = detections.tracker_id[i]
            
            if track_id != -1:
                label = f"Horse-{track_id} ({confidence:.2f})"
            else:
                label = f"Horse ({confidence:.2f})"
                
            labels.append(label)
        
        return labels
    
    def _draw_status(self, frame, frame_idx, num_detections, stats):
        height, width = frame.shape[:2]
        
        cv2.rectangle(frame, (10, 10), (550, 180), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (550, 180), (255, 255, 255), 2)
        
        grouping_rate = stats.get('grouping_success_rate', 0) * 100
        avg_yolo = stats.get('avg_yolo_per_frame', 0)
        avg_seg = stats.get('avg_segmentation_per_frame', 0)
        
        status_lines = [
            f"SOLUTION 2: YOLO + Segmentation + MOT",
            f"Frame: {frame_idx}",
            f"Final Detections: {num_detections}",
            f"Grouping Success: {grouping_rate:.1f}%",
            f"Avg YOLO/frame: {avg_yolo:.1f}",
            f"Avg Segmentation/frame: {avg_seg:.1f}",
            f"Pipeline: Clean + Precise + Traditional"
        ]
        
        for i, line in enumerate(status_lines):
            y_pos = 35 + (i * 20)
            color = (0, 255, 0) if num_detections > 0 else (255, 255, 0)
            cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
        
        return frame

def _validate_detection_data(dets_np, frame_idx):
    if dets_np is None or len(dets_np) == 0:
        return True
    
    if np.any(np.isnan(dets_np)) or np.any(np.isinf(dets_np)):
        print(f"🚨 NaN/Inf values detected in detection data at frame {frame_idx}")
        return False
    
    for i, det in enumerate(dets_np):
        x1, y1, x2, y2 = det[:4]
        
        if x2 <= x1 or y2 <= y1:
            print(f"🚨 Invalid bounding box dimensions at frame {frame_idx}, detection {i}")
            return False
        
        if x1 < 0 or y1 < 0 or x2 > 10000 or y2 > 10000:
            print(f"🚨 Extreme coordinate values at frame {frame_idx}, detection {i}")
            return False
    
    confidences = dets_np[:, 4]
    if np.any(confidences < 0) or np.any(confidences > 1):
        print(f"🚨 Invalid confidence values at frame {frame_idx}")
        return False
    
    return True

def _bbox_overlap(bbox1, bbox2):
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

def process_video_solution2(config: Config, max_frames: int = 2000, 
                           save_video: bool = True, output_video_path: str = None,
                           yolo_model: str = "yolo11s.pt", confidence_threshold: float = 0.5):
    cache_file = "detection_cache/horse_9_del_mar_pan_seg_mdxam_2_0.7_2100.json"
    
    print(f"Loading cached segmentation detections from {cache_file}")
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
    
    print(f"Loaded {len(cached_detections)} cached segmentation frames")
    
    tracker_system = SegmentationObjectDetectionTracker(
        yolo_model_path=yolo_model,
        confidence_threshold=confidence_threshold,
        max_horses=6
    )
    
    deepsort_tracker = DeepOcSort(
        reid_weights=Path('osnet_x0_25_msmt17.pt'),
        device='cuda:0',
        half=True
    )
    
    cap = cv2.VideoCapture(config.video_path)
    
    video_writer = None
    if save_video:
        if output_video_path is None:
            output_video_path = f"solution2_yolo_segmentation_tracking_{int(time.time())}.mp4"
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print(f"📹 Saving SOLUTION 2 video to: {output_video_path}")
        print(f"   Resolution: {width}x{height} @ {fps}fps")
    
    annotator = SupervisionVideoAnnotator()
    
    track_history = {}
    tracker_reinit_count = 0
    
    for frame_idx, segmentation_detections in enumerate(cached_detections[:max_frames]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        grouped_detections = tracker_system.process_frame(frame, segmentation_detections)
        
        if len(grouped_detections) > 0:
            dets_np = np.column_stack((
                grouped_detections.xyxy,
                grouped_detections.confidence,
                np.zeros(len(grouped_detections))
            )).astype(np.float64)
            
            if not _validate_detection_data(dets_np, frame_idx):
                print(f"⚠️  Invalid detection data at frame {frame_idx}, skipping tracker update")
                grouped_detections.tracker_id = np.array([-1] * len(grouped_detections))
            else:
                try:
                    tracks = deepsort_tracker.update(dets_np, frame)
                    
                    if tracks is not None:
                        track_ids = []
                        for i, detection in enumerate(grouped_detections.xyxy):
                            matched_track_id = -1
                            for track in tracks:
                                track_bbox = track[:4]
                                if _bbox_overlap(detection, track_bbox) > 0.5:
                                    matched_track_id = int(track[4])
                                    break
                            track_ids.append(matched_track_id)
                        
                        grouped_detections.tracker_id = np.array(track_ids)
                        
                        for track in tracks:
                            track_id = int(track[4])
                            if track_id not in track_history:
                                track_history[track_id] = {'start': frame_idx, 'end': frame_idx}
                            track_history[track_id]['end'] = frame_idx
                    else:
                        grouped_detections.tracker_id = np.array([-1] * len(grouped_detections))
                        
                except Exception as e:
                    print(f"🚨 TRACKER STATE CORRUPTION at frame {frame_idx}: {str(e)}")
                    print(f"   Reinitializing DeepOCSORT tracker...")
                    tracker_reinit_count += 1
                    
                    deepsort_tracker = DeepOcSort(
                        reid_weights=Path('osnet_x0_25_msmt17.pt'),
                        device='cuda:0',
                        half=True
                    )
                    
                    grouped_detections.tracker_id = np.array([-1] * len(grouped_detections))
        else:
            grouped_detections.tracker_id = np.array([])
        
        if save_video:
            stats = tracker_system.get_statistics()
            annotated_frame = annotator.annotate_frame(
                frame, grouped_detections, segmentation_detections, frame_idx, stats
            )
            video_writer.write(annotated_frame)
                    
        if frame_idx % 100 == 0:
            stats = tracker_system.get_statistics()
            print(f"Frame {frame_idx}: YOLO+Seg → {len(grouped_detections)} final detections, "
                  f"Grouping: {stats.get('grouping_success_rate', 0)*100:.1f}%")
            
            if frame_idx % 500 == 0:
                gc.collect()
    
    cap.release()
    if video_writer:
        video_writer.release()
        print(f"✅ SOLUTION 2 VIDEO SAVED: {output_video_path}")
    
    final_stats = tracker_system.get_statistics()
    long_tracks = {tid: data for tid, data in track_history.items() 
                   if data['end'] - data['start'] >= 1500}
    
    print(f"\n🏁 SOLUTION 2 RESULTS:")
    print(f"Total tracks: {len(track_history)} → Long tracks: {len(long_tracks)}")
    print(f"Grouping success rate: {final_stats.get('grouping_success_rate', 0)*100:.1f}%")
    print(f"Average YOLO detections/frame: {final_stats.get('avg_yolo_per_frame', 0):.1f}")
    print(f"Average segmentation blobs/frame: {final_stats.get('avg_segmentation_per_frame', 0):.1f}")
    
    if tracker_reinit_count > 0:
        print(f"🚨 TRACKER RECOVERY STATISTICS:")
        print(f"   State corruptions recovered: {tracker_reinit_count}")
    else:
        print(f"✅ TRACKER STABILITY: Zero state corruptions detected")
    
    return track_history

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python sam_test.py config.yaml [max_frames] [--no-video] [--output-video path] [--yolo-model model] [--confidence threshold]")
        sys.exit(1)
    
    config_file = sys.argv[1]
    max_frames = 2000
    save_video = True
    output_video_path = None
    yolo_model = "yolo11s.pt"
    confidence_threshold = 0.5
    
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg.isdigit():
            max_frames = int(arg)
        elif arg == '--no-video':
            save_video = False
        elif arg == '--output-video' and i + 1 < len(sys.argv):
            output_video_path = sys.argv[i + 1]
        elif arg == '--yolo-model' and i + 1 < len(sys.argv):
            yolo_model = sys.argv[i + 1]
        elif arg == '--confidence' and i + 1 < len(sys.argv):
            confidence_threshold = float(sys.argv[i + 1])
    
    config = Config(config_file)
    
    print(f"🏇 SOLUTION 2: YOLO + SEGMENTATION + TRADITIONAL MOT")
    print(f"   Video: {config.video_path}")
    print(f"   Max Frames: {max_frames}")
    print(f"   YOLO Model: {yolo_model}")
    print(f"   Confidence: {confidence_threshold}")
    print(f"   Save Video: {save_video}")
    print(f"   Pipeline: Clean Detection + Precise Segmentation + Reliable MOT")
    
    track_data = process_video_solution2(
        config, max_frames, save_video, output_video_path, yolo_model, confidence_threshold
    )
    
    with open('solution2_tracks.json', 'w') as f:
        json.dump(track_data, f, indent=2, default=str)
    
    print(f"✅ Saved SOLUTION 2 results to solution2_tracks.json")

if __name__ == "__main__":
    main()