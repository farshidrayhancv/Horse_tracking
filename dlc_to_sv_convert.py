#!/usr/bin/env python3
"""
Hybrid horse detection system:
- YOLOv11 for horse detection (object detection)
- DeepLabCut keypoints when available
- SigLIP for matching YOLO detections to DLC individuals
"""

import numpy as np
import pandas as pd
import cv2
import supervision as sv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from transformers import AutoProcessor, SiglipVisionModel
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore")


class HybridHorseSystem:
    """Hybrid system: YOLO detection + DLC keypoints + SigLIP identification"""
    
    def __init__(self, h5_path: str, video_path: str, yolo_model: str = "yolo11n.pt", device: str = 'auto', save_crops: bool = True):
        self.h5_path = Path(h5_path)
        self.video_path = Path(video_path)
        
        # Smart device selection
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Crop saving setup
        self.save_crops = save_crops
        if self.save_crops:
            self.crop_save_dir = Path("hybrid_horse_crops")
            self.crop_save_dir.mkdir(exist_ok=True)
            print(f"📁 Saving crops to: {self.crop_save_dir}")
        
        # Load YOLO model for horse detection
        self.load_yolo_model(yolo_model)
        
        # Load DLC data
        self.df = pd.read_hdf(self.h5_path)
        self.setup_horse_info()
        
        # Setup video
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Load SigLIP model for horse identification
        self.load_siglip_model()
        
        # Horse matching system
        self.num_horses = 8  # We know there are 8 horses in DLC data
        self.horse_matcher = None
        self.training_crops = []
        self.crop_size = (224, 224)
        self.crop_counter = 0
        
        # Detection matching parameters
        self.iou_threshold = 0.3  # For matching YOLO boxes with DLC regions
        
        print(f"🐴 Hybrid horse system ready: {len(self.individuals)} DLC horses, {self.total_frames} frames")
        print(f"🧠 Using device: {self.device}")
    
    def load_yolo_model(self, model_path: str):
        """Load YOLO model for horse detection"""
        try:
            self.yolo_model = YOLO(model_path)
            # COCO class for horse is 17
            self.horse_class_id = 17
            print(f"✓ YOLO model loaded: {model_path}")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            self.yolo_model = None
    
    def setup_horse_info(self):
        """Setup DLC horse info"""
        self.scorer = self.df.columns.levels[0][0]
        self.individuals = list(self.df.columns.levels[1])
        self.bodyparts = list(self.df.columns.levels[2])
        
        # Skeleton connections
        self.skeleton_connections = [
            ('nose', 'upper_jaw'), ('nose', 'lower_jaw'),
            ('left_eye', 'right_eye'), ('left_eye', 'nose'), ('right_eye', 'nose'),
            ('left_earbase', 'left_earend'), ('right_earbase', 'right_earend'),
            ('throat_base', 'throat_end'), ('neck_base', 'neck_end'),
            ('neck_base', 'body_middle_left'), ('neck_base', 'body_middle_right'),
            ('body_middle_left', 'back_base'), ('body_middle_right', 'back_base'),
            ('back_base', 'back_middle'), ('back_middle', 'back_end'),
            ('body_middle_left', 'front_left_thai'), ('front_left_thai', 'front_left_knee'),
            ('front_left_knee', 'front_left_paw'),
            ('body_middle_right', 'front_right_thai'), ('front_right_thai', 'front_right_knee'),
            ('front_right_knee', 'front_right_paw'),
            ('back_base', 'back_left_thai'), ('back_left_thai', 'back_left_knee'),
            ('back_left_knee', 'back_left_paw'),
            ('back_base', 'back_right_thai'), ('back_right_thai', 'back_right_knee'),
            ('back_right_knee', 'back_right_paw'),
            ('back_end', 'tail_base'), ('tail_base', 'tail_end'),
        ]
        
        # Colors for 8 horses
        self.colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
    
    def load_siglip_model(self):
        """Load SigLIP model for horse identification"""
        try:
            model_path = "google/siglip-base-patch16-224"
            print(f"Loading SigLIP model...")
            self.siglip_processor = AutoProcessor.from_pretrained(model_path)
            self.siglip_model = SiglipVisionModel.from_pretrained(model_path)
            
            if self.device == 'cuda':
                try:
                    self.siglip_model.to('cuda')
                    test_tensor = torch.randn(1, 3, 224, 224).to('cuda')
                    with torch.no_grad():
                        _ = self.siglip_model.vision_model.embeddings(test_tensor)
                    print(f"✓ SigLIP model loaded on GPU")
                except (RuntimeError, torch.cuda.OutOfMemoryError):
                    print(f"⚠️ GPU failed, switching to CPU...")
                    torch.cuda.empty_cache()
                    self.device = 'cpu'
                    self.siglip_model.to('cpu')
                    print(f"✓ SigLIP model loaded on CPU")
            else:
                self.siglip_model.to('cpu')
                print(f"✓ SigLIP model loaded on CPU")
            
            self.siglip_model.eval()
            
        except Exception as e:
            print(f"❌ Error loading SigLIP model: {e}")
            self.siglip_model = None
            self.siglip_processor = None
    
    def detect_horses_yolo(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        """Detect horses using YOLO"""
        if self.yolo_model is None:
            return []
        
        try:
            results = self.yolo_model(frame, verbose=False)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # Check if it's a horse (class 17 in COCO)
                        if int(box.cls) == self.horse_class_id and float(box.conf) > confidence_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            confidence = float(box.conf)
                            
                            detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence,
                                'yolo_id': i
                            })
            
            return detections
            
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return []
    
    def get_keypoints_for_frame(self, frame_idx: int) -> Dict:
        """Get DLC keypoints for a frame"""
        frame_data = {}
        
        try:
            for individual in self.individuals:
                keypoints = {}
                for bodypart in self.bodyparts:
                    try:
                        x = self.df.iloc[frame_idx][(self.scorer, individual, bodypart, 'x')]
                        y = self.df.iloc[frame_idx][(self.scorer, individual, bodypart, 'y')]
                        conf = self.df.iloc[frame_idx][(self.scorer, individual, bodypart, 'likelihood')]
                        
                        if x == -1.0 or y == -1.0 or conf == -1.0:
                            keypoints[bodypart] = (0.0, 0.0, 0.0)
                        else:
                            keypoints[bodypart] = (float(x), float(y), float(conf))
                    except:
                        keypoints[bodypart] = (0.0, 0.0, 0.0)
                
                frame_data[individual] = keypoints
        except:
            for individual in self.individuals:
                keypoints = {bodypart: (0.0, 0.0, 0.0) for bodypart in self.bodyparts}
                frame_data[individual] = keypoints
        
        return frame_data
    
    def get_dlc_bounding_box(self, keypoints: Dict, confidence_threshold: float = 0.1):
        """Get bounding box from DLC keypoints"""
        confident_points = []
        for x, y, conf in keypoints.values():
            if conf > confidence_threshold and x > 0 and y > 0:
                confident_points.append((x, y))
        
        if len(confident_points) >= 3:
            xs, ys = zip(*confident_points)
            padding = 20
            x1 = max(0, int(min(xs)) - padding)
            y1 = max(0, int(min(ys)) - padding)
            x2 = min(self.width, int(max(xs)) + padding)
            y2 = min(self.height, int(max(ys)) + padding)
            return x1, y1, x2, y2
        return None
    
    def calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def match_yolo_to_dlc(self, yolo_detections: List[Dict], dlc_keypoints: Dict, 
                          confidence_threshold: float = 0.1) -> List[Dict]:
        """Match YOLO detections to DLC individuals"""
        matched_horses = []
        
        # Get DLC bounding boxes
        dlc_boxes = {}
        for individual, keypoints in dlc_keypoints.items():
            bbox = self.get_dlc_bounding_box(keypoints, confidence_threshold)
            if bbox:
                dlc_boxes[individual] = bbox
        
        # Match each YOLO detection to best DLC individual
        for yolo_det in yolo_detections:
            yolo_bbox = yolo_det['bbox']
            best_match = None
            best_iou = 0.0
            
            for individual, dlc_bbox in dlc_boxes.items():
                iou = self.calculate_iou(yolo_bbox, dlc_bbox)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_match = individual
            
            matched_horses.append({
                'yolo_detection': yolo_det,
                'dlc_individual': best_match,
                'dlc_keypoints': dlc_keypoints.get(best_match, {}) if best_match else {},
                'iou': best_iou,
                'has_keypoints': best_match is not None
            })
        
        return matched_horses
    
    def extract_horse_crop(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extract horse+jockey crop optimized for helmet identification"""
        x1, y1, x2, y2 = bbox
        
        box_width = x2 - x1
        box_height = y2 - y1
        
        if box_width < 32 or box_height < 32:
            return None
        
        # Racing-optimized padding (more space above for jockey helmet)
        pad_x = max(30, int(box_width * 0.4))
        pad_y_bottom = max(20, int(box_height * 0.3))
        pad_y_top = max(40, int(box_height * 0.7))
        
        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y_top)
        x2_pad = min(frame.shape[1], x2 + pad_x)
        y2_pad = min(frame.shape[0], y2 + pad_y_bottom)
        
        crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        
        if crop.size == 0:
            return None
        
        return cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_AREA)
    
    def get_siglip_embedding(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """Get SigLIP embedding from horse crop"""
        if self.siglip_model is None or crop is None:
            return None
        
        try:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            inputs = self.siglip_processor(images=crop_rgb, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.siglip_model(**inputs)
                embedding = outputs.pooler_output.cpu().numpy().flatten()
            
            return embedding
            
        except Exception as e:
            print(f"Error getting SigLIP embedding: {e}")
            return None
    
    def save_crop_image(self, crop: np.ndarray, frame_idx: int, detection_id: int, 
                       horse_id: int = None, phase: str = "training"):
        """Save crop image to disk"""
        if not self.save_crops or crop is None:
            return
        
        try:
            if horse_id is not None:
                filename = f"{phase}_frame{frame_idx:06d}_det{detection_id}_horse{horse_id}_{self.crop_counter:04d}.jpg"
            else:
                filename = f"{phase}_frame{frame_idx:06d}_det{detection_id}_{self.crop_counter:04d}.jpg"
            
            filepath = self.crop_save_dir / filename
            cv2.imwrite(str(filepath), crop)
            self.crop_counter += 1
            
        except Exception as e:
            print(f"Error saving crop: {e}")
    
    def collect_training_data(self, samples_per_100_frames: int = 5, confidence_threshold: float = 0.3):
        """Collect training data from YOLO detections matched to DLC individuals"""
        print(f"📚 Collecting training data: {samples_per_100_frames} samples per 100 frames...")
        
        segments = self.total_frames // 100
        frame_indices = []
        
        for segment in range(segments):
            segment_start = segment * 100
            segment_end = min((segment + 1) * 100, self.total_frames)
            segment_frames = np.linspace(segment_start, segment_end - 1, samples_per_100_frames).astype(int)
            frame_indices.extend(segment_frames)
        
        print(f"🎬 Sampling {len(frame_indices)} frames across {segments} video segments")
        
        for i, frame_idx in enumerate(frame_indices):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Detect horses with YOLO
            yolo_detections = self.detect_horses_yolo(frame, confidence_threshold)
            
            # Get DLC keypoints
            dlc_keypoints = self.get_keypoints_for_frame(frame_idx)
            
            # Match YOLO detections to DLC individuals
            matched_horses = self.match_yolo_to_dlc(yolo_detections, dlc_keypoints, confidence_threshold)
            
            # Collect training crops from matched horses
            for match in matched_horses:
                if match['has_keypoints']:  # Only use horses with DLC keypoints for training
                    yolo_bbox = match['yolo_detection']['bbox']
                    
                    # CRITICAL: Extract crop from CLEAN frame BEFORE any annotations
                    crop = self.extract_horse_crop(frame, yolo_bbox)
                    
                    if crop is not None:
                        self.save_crop_image(crop, frame_idx, match['yolo_detection']['yolo_id'], phase="training")
                        
                        embedding = self.get_siglip_embedding(crop)
                        if embedding is not None:
                            self.training_crops.append({
                                'crop': crop,
                                'embedding': embedding,
                                'frame': frame_idx,
                                'dlc_individual': match['dlc_individual'],
                                'yolo_confidence': match['yolo_detection']['confidence'],
                                'iou': match['iou']
                            })
            
            if i % 25 == 0:
                segment_progress = (i // samples_per_100_frames) + 1
                print(f"  Segment {segment_progress}/{segments} | Frame {frame_idx} | Crops: {len(self.training_crops)}")
        
        print(f"✓ Collected {len(self.training_crops)} training crops from matched detections")
    
    def train_horse_matcher(self):
        """Train horse matcher using SigLIP embeddings"""
        if len(self.training_crops) < 16:
            print(f"❌ Not enough training crops: {len(self.training_crops)} (need at least 16)")
            return False
        
        print(f"🧠 Training horse matcher on {len(self.training_crops)} crops...")
        
        # Analyze DLC individual distribution in training data
        dlc_distribution = {}
        for crop in self.training_crops:
            individual = crop['dlc_individual']
            dlc_distribution[individual] = dlc_distribution.get(individual, 0) + 1
        
        print(f"📊 DLC individual distribution in training:")
        for individual, count in sorted(dlc_distribution.items()):
            print(f"  {individual}: {count} crops")
        
        embeddings = np.array([crop['embedding'] for crop in self.training_crops], dtype=np.float64)
        
        # Check embedding diversity
        print(f"📈 Embedding stats: shape={embeddings.shape}, mean={embeddings.mean():.3f}, std={embeddings.std():.3f}")
        
        scaler = StandardScaler()
        embeddings_normalized = scaler.fit_transform(embeddings)
        
        # Try different cluster numbers to see what works best
        from sklearn.metrics import silhouette_score
        
        best_k = self.num_horses
        best_score = -1
        
        for k in range(2, min(self.num_horses + 3, len(self.training_crops) // 2)):
            kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_test = kmeans_test.fit_predict(embeddings_normalized)
            
            if len(np.unique(labels_test)) > 1:  # Need at least 2 clusters for silhouette score
                score = silhouette_score(embeddings_normalized, labels_test)
                print(f"  K={k}: silhouette_score={score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_k = k
        
        print(f"🎯 Using K={best_k} clusters (silhouette score: {best_score:.3f})")
        
        # Final clustering
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        horse_labels = kmeans.fit_predict(embeddings_normalized)
        
        # Create mapping from cluster to most common DLC individual
        cluster_to_dlc = {}
        for cluster_id in range(best_k):
            cluster_mask = horse_labels == cluster_id
            cluster_crops = [self.training_crops[i] for i in range(len(self.training_crops)) if cluster_mask[i]]
            
            if cluster_crops:
                # Find most common DLC individual in this cluster
                dlc_counts = {}
                for crop in cluster_crops:
                    dlc_ind = crop['dlc_individual']
                    dlc_counts[dlc_ind] = dlc_counts.get(dlc_ind, 0) + 1
                
                most_common_dlc = max(dlc_counts, key=dlc_counts.get)
                cluster_to_dlc[cluster_id] = most_common_dlc
                
                print(f"  Cluster {cluster_id} → {most_common_dlc} ({dlc_counts[most_common_dlc]}/{len(cluster_crops)} crops)")
        
        self.horse_matcher = {
            'kmeans': kmeans,
            'scaler': scaler,
            'cluster_to_dlc': cluster_to_dlc,
            'trained': True,
            'num_clusters': best_k
        }
        
        unique_labels = np.unique(horse_labels)
        print(f"✓ Horse matcher trained: {len(unique_labels)} clusters found")
        
        return True
    
    def identify_horse(self, crop: np.ndarray) -> Optional[int]:
        """Identify horse using trained matcher"""
        if self.horse_matcher is None or not self.horse_matcher.get('trained', False):
            return None
        
        embedding = self.get_siglip_embedding(crop)
        if embedding is None:
            return None
        
        try:
            embedding = embedding.astype(np.float64)
            embedding_normalized = self.horse_matcher['scaler'].transform([embedding])
            cluster_id = self.horse_matcher['kmeans'].predict(embedding_normalized)[0]
            
            # Map cluster to DLC individual, then to a consistent horse ID
            dlc_individual = self.horse_matcher['cluster_to_dlc'].get(cluster_id, f"unknown_{cluster_id}")
            
            # Create consistent mapping from DLC individual names to horse IDs
            if not hasattr(self, 'dlc_to_horse_id'):
                self.dlc_to_horse_id = {}
                for i, individual in enumerate(sorted(self.individuals)):
                    self.dlc_to_horse_id[individual] = i
            
            # Get horse ID from DLC individual
            if dlc_individual in self.dlc_to_horse_id:
                horse_id = self.dlc_to_horse_id[dlc_individual]
            else:
                # For unknown individuals, use cluster_id as fallback
                horse_id = cluster_id % len(self.colors)
            
            return int(horse_id)
            
        except Exception as e:
            print(f"Error identifying horse: {e}")
            return None
    
    def draw_horse_skeleton(self, frame: np.ndarray, keypoints: Dict, color: Tuple[int, int, int], 
                           confidence_threshold: float = 0.1) -> np.ndarray:
        """Draw horse skeleton from DLC keypoints"""
        
        # Draw skeleton connections
        for connection in self.skeleton_connections:
            pt1_name, pt2_name = connection
            if pt1_name in keypoints and pt2_name in keypoints:
                x1, y1, conf1 = keypoints[pt1_name]
                x2, y2, conf2 = keypoints[pt2_name]
                
                if (conf1 > confidence_threshold and conf2 > confidence_threshold and
                    x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0):
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        
        # Draw keypoints
        for bodypart, (x, y, conf) in keypoints.items():
            if conf > confidence_threshold and x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), 5, color, -1)
                cv2.circle(frame, (int(x), int(y)), 6, (255, 255, 255), 1)
        
        return frame
    
    def process_video(self, output_path: str = None, max_frames: int = None, confidence_threshold: float = 0.3):
        """Process video with hybrid YOLO + DLC + SigLIP system"""
        
        if output_path is None:
            output_path = str(self.video_path.with_suffix('.hybrid_horses.mp4'))
        
        # Step 1: Collect training data
        self.collect_training_data(samples_per_100_frames=5, confidence_threshold=confidence_threshold)
        
        # Step 2: Train horse matcher
        if not self.train_horse_matcher():
            print("❌ Failed to train horse matcher")
            return None
        
        # Step 3: Process video
        print(f"🎬 Processing video with hybrid system...")
        print(f"📹 Output: {output_path}")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 24, (self.width, self.height))
        
        frame_count = 0
        max_frames = max_frames or self.total_frames
        stats = {'yolo_detections': 0, 'matched_with_keypoints': 0, 'identified': 0}
        
        while frame_count < max_frames:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # YOLO horse detection
            yolo_detections = self.detect_horses_yolo(frame, confidence_threshold)
            stats['yolo_detections'] += len(yolo_detections)
            
            # DLC keypoints
            dlc_keypoints = self.get_keypoints_for_frame(frame_count)
            
            # Match YOLO to DLC
            matched_horses = self.match_yolo_to_dlc(yolo_detections, dlc_keypoints, confidence_threshold)
            
            detected_horse_ids = set()
            
            for match in matched_horses:
                yolo_bbox = match['yolo_detection']['bbox']
                x1, y1, x2, y2 = yolo_bbox
                
                # CRITICAL: Extract crop from CLEAN frame BEFORE any drawing
                crop = self.extract_horse_crop(frame, yolo_bbox)
                horse_id = self.identify_horse(crop) if crop is not None else None
                
                # Save crop from clean frame (first 500 frames)
                if crop is not None and frame_count < 500:
                    self.save_crop_image(crop, frame_count, match['yolo_detection']['yolo_id'], 
                                       horse_id, "processing")
                
                if horse_id is not None:
                    stats['identified'] += 1
                    detected_horse_ids.add(horse_id)
                    color = self.colors[horse_id % len(self.colors)]
                else:
                    horse_id = 0
                    color = (128, 128, 128)
                
                # NOW draw annotations on frame (AFTER crop extraction)
                # Draw YOLO bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw horse ID with more info
                if dlc_individual := (match.get('dlc_individual') if match['has_keypoints'] else None):
                    label = f"Horse {horse_id} ({dlc_individual})"
                else:
                    label = f"Horse {horse_id} (YOLO only)"
                
                # Different label colors for different confidence levels
                label_color = color if horse_id != 0 else (128, 128, 128)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
                
                # Draw skeleton ONLY if we have matched keypoints (AFTER crop extraction)
                if match['has_keypoints']:
                    stats['matched_with_keypoints'] += 1
                    frame = self.draw_horse_skeleton(frame, match['dlc_keypoints'], color, confidence_threshold)
            
            # Add frame info
            info_lines = [
                f"Frame: {frame_count}/{max_frames}",
                f"YOLO: {len(yolo_detections)} | Keypoints: {len([m for m in matched_horses if m['has_keypoints']])}",
                f"Horses: {len(detected_horse_ids)}"
            ]
            
            for i, line in enumerate(info_lines):
                y_pos = 30 + i * 30
                cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            out.write(frame)
            
            if frame_count % 100 == 0:
                print(f"  Frame {frame_count}/{max_frames} | YOLO: {len(yolo_detections)} | With keypoints: {len([m for m in matched_horses if m['has_keypoints']])}")
            
            frame_count += 1
        
        self.cap.release()
        out.release()
        
        print(f"✅ Video processed successfully!")
        print(f"📊 YOLO detections: {stats['yolo_detections']}")
        print(f"📊 Matched with keypoints: {stats['matched_with_keypoints']}")
        print(f"📊 Successfully identified: {stats['identified']}")
        print(f"🎯 Output saved to: {output_path}")
        
        return output_path


def process_hybrid_horses(h5_path: str, video_path: str, output_path: str = None, 
                         max_frames: int = None, confidence_threshold: float = 0.3):
    """Process horses using hybrid YOLO + DLC + SigLIP system"""
    
    system = HybridHorseSystem(h5_path, video_path, save_crops=True)
    return system.process_video(output_path, max_frames, confidence_threshold)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python hybrid_horse_system.py <h5_file> <video_file> [output_file] [max_frames] [confidence]")
        print("Example: python hybrid_horse_system.py poses.h5 horses.mp4 output.mp4 1000 0.3")
        sys.exit(1)
    
    h5_path = sys.argv[1]
    video_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    max_frames = int(sys.argv[4]) if len(sys.argv) > 4 else None
    confidence = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
    
    process_hybrid_horses(h5_path, video_path, output_path, max_frames, confidence)
