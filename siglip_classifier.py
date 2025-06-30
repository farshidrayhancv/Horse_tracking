"""
Enhanced SigLIP-based Horse/Jockey Classification
Multi-image training with clustering and classifier
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

try:
    import supervision as sv
except ImportError:
    sv = None

try:
    from transformers import AutoProcessor, SiglipVisionModel
    SIGLIP_AVAILABLE = True
except ImportError:
    SIGLIP_AVAILABLE = False

class SigLIPClassifier:
    """Enhanced SigLIP-based classification with clustering and training"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.n_races = getattr(config, 'n_races', 9)  # 9 horses, 9 jockeys
        
        # Initialize SigLIP
        self.siglip_model = None
        self.siglip_processor = None
        self.setup_siglip()
        
        # Classifiers
        self.horse_classifier = None
        self.jockey_classifier = None
        
        # Templates and embeddings
        self.horse_embeddings = []
        self.jockey_embeddings = []
        self.horse_labels = []
        self.jockey_labels = []
        
        # Dynamic thresholds
        self.horse_threshold = getattr(config, 'siglip_confidence_threshold', 0.7)
        self.jockey_threshold = getattr(config, 'siglip_confidence_threshold', 0.7)
        
        # Multi-scale settings
        self.scales = getattr(config, 'crop_scales', [1.0, 1.2, 0.8])
        
        # Build training data and train classifiers
        self.build_training_pipeline()
        
        print(f"✅ Enhanced SigLIP Classifier initialized")
        print(f"   🐴 Horse templates: {len([t for t in self.horse_embeddings if t is not None])}")
        print(f"   🏇 Jockey templates: {len([t for t in self.jockey_embeddings if t is not None])}")
    
    def setup_siglip(self):
        """Initialize SigLIP model"""
        if not SIGLIP_AVAILABLE:
            print("❌ SigLIP not available - install transformers")
            return
        
        try:
            model_name = "google/siglip-base-patch16-224"
            self.siglip_processor = AutoProcessor.from_pretrained(model_name)
            self.siglip_model = SiglipVisionModel.from_pretrained(model_name)
            self.siglip_model.to(self.device)
            self.siglip_model.eval()
            print("✅ SigLIP model loaded")
        except Exception as e:
            print(f"❌ SigLIP setup failed: {e}")
    
    def build_training_pipeline(self):
        """Complete training pipeline: detection → clustering → classifier training"""
        folder_path = Path(self.config.reference_folder_path)
        if not folder_path.exists():
            print(f"❌ Reference folder not found: {folder_path}")
            return
        
        print(f"📁 Processing images from: {folder_path}")
        
        # Step 1: Process all images and extract embeddings
        all_horse_crops, all_jockey_crops = self.process_reference_images(folder_path)
        
        if not all_horse_crops or not all_jockey_crops:
            print("❌ Insufficient data for training")
            return
        
        # Step 2: Extract embeddings for all crops
        print(f"   🔄 Extracting embeddings for {len(all_horse_crops)} horse crops...")
        horse_embeddings = self.extract_batch_embeddings(all_horse_crops)
        print(f"   🔄 Extracting embeddings for {len(all_jockey_crops)} jockey crops...")
        jockey_embeddings = self.extract_batch_embeddings(all_jockey_crops)
        
        print(f"   📊 Horse embeddings shape: {horse_embeddings.shape}")
        print(f"   📊 Jockey embeddings shape: {jockey_embeddings.shape}")
        
        # Step 3: Cluster into race groups (clusters 0-8, background=9)
        horse_labels = self.cluster_entities_with_background(horse_embeddings, all_horse_crops, "horses")
        jockey_labels = self.cluster_entities_with_background(jockey_embeddings, all_jockey_crops, "jockeys")
        
        print(f"   🏷️  Horse labels: {len(horse_labels)} assigned")
        print(f"   🏷️  Jockey labels: {len(jockey_labels)} assigned")
        
        # Step 4: Save crops regardless of clustering success
        if len(horse_labels) > 0:
            self.save_clustered_crops(all_horse_crops, horse_labels, "horses")
            self.create_cluster_visualizations(all_horse_crops, horse_labels, "horses")
        
        if len(jockey_labels) > 0:
            self.save_clustered_crops(all_jockey_crops, jockey_labels, "jockeys")
            self.create_cluster_visualizations(all_jockey_crops, jockey_labels, "jockeys")
        
        # Step 5: Store cluster embeddings as templates
        self.store_cluster_templates(horse_embeddings, horse_labels, "horses")
        self.store_cluster_templates(jockey_embeddings, jockey_labels, "jockeys")
    
    def process_reference_images(self, folder_path):
        """Process all images in folder and extract crops"""
        from detectors import DetectionManager
        from models import SuperAnimalQuadruped
        
        # Setup detection
        superanimal = SuperAnimalQuadruped(device=self.device, config=self.config)
        detection_manager = DetectionManager(self.config, superanimal)
        
        all_horse_crops = []
        all_jockey_crops = []
        
        image_files = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png"))
        print(f"   📷 Processing {len(image_files)} images")
        
        for img_path in image_files:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Detect entities
            human_detections = detection_manager.detect_humans(image)
            horse_detections = detection_manager.detect_horses(image)
            
            # Extract crops with multi-scale
            if sv and len(horse_detections) > 0:
                for bbox in horse_detections.xyxy:
                    crops = self.extract_multiscale_crops(image, bbox)
                    all_horse_crops.extend(crops)
            
            if sv and len(human_detections) > 0:
                for bbox in human_detections.xyxy:
                    crops = self.extract_multiscale_crops(image, bbox)
                    all_jockey_crops.extend(crops)
        
        print(f"   🐴 Extracted {len(all_horse_crops)} horse crops")
        print(f"   🏇 Extracted {len(all_jockey_crops)} jockey crops")
        
        return all_horse_crops, all_jockey_crops
    
    def extract_multiscale_crops(self, image, bbox, padding=0.1):
        """Extract crops at multiple scales"""
        crops = []
        
        for scale in self.scales:
            crop = self.extract_crop_with_scale(image, bbox, padding, scale)
            if crop is not None:
                crops.append(crop)
        
        return crops
    
    def extract_crop_with_scale(self, frame, bbox, padding=0.1, scale=1.0):
        """Extract crop with specific scale"""
        try:
            x1, y1, x2, y2 = bbox.astype(int)
            
            # Calculate center and scaled dimensions
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            w, h = int((x2 - x1) * scale), int((y2 - y1) * scale)
            
            # Add padding
            pad_x = int(w * padding)
            pad_y = int(h * padding)
            
            # Calculate new bounds
            x1 = max(0, cx - w//2 - pad_x)
            y1 = max(0, cy - h//2 - pad_y)
            x2 = min(frame.shape[1], cx + w//2 + pad_x)
            y2 = min(frame.shape[0], cy + h//2 + pad_y)
            
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                return None
            
            # Resize to standard size
            crop = cv2.resize(crop, (224, 224))
            return crop
            
        except Exception as e:
            return None
    
    def extract_batch_embeddings(self, crops):
        """Extract embeddings for batch of crops"""
        if not crops or self.siglip_model is None:
            return np.array([])
        
        embeddings = []
        batch_size = 32
        
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i+batch_size]
            batch_embeddings = []
            
            for crop in batch_crops:
                embedding = self.extract_embedding(crop)
                if embedding is not None:
                    batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings) if embeddings else np.array([])
    
    def extract_embedding(self, crop):
        """Extract SigLIP embedding from crop"""
        if self.siglip_model is None or crop is None:
            return None
        
        try:
            # Convert BGR to RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Process with SigLIP
            inputs = self.siglip_processor(images=crop_rgb, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.siglip_model(**inputs)
                embedding = outputs.pooler_output.cpu().numpy().flatten()
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            return None
    
    def cluster_entities(self, embeddings, n_clusters, entity_type):
        """Cluster embeddings into race groups"""
        if len(embeddings) < n_clusters:
            print(f"❌ Insufficient {entity_type} for clustering: {len(embeddings)} < {n_clusters}")
            return np.arange(len(embeddings)) % n_clusters
    
    def save_clustered_crops(self, crops, labels, entity_type):
        """Save crops organized by cluster"""
        output_dir = Path("clustered_ref_images") / entity_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create cluster subdirectories
        for cluster_id in range(self.n_races):
            cluster_dir = output_dir / f"cluster_{cluster_id}"
            cluster_dir.mkdir(exist_ok=True)
        
        # Save crops to respective clusters
        for i, (crop, label) in enumerate(zip(crops, labels)):
            cluster_dir = output_dir / f"cluster_{label}"
            crop_path = cluster_dir / f"{entity_type}_{i:04d}.jpg"
            cv2.imwrite(str(crop_path), crop)
        
        print(f"   💾 Saved {len(crops)} {entity_type} crops to clustered_ref_images/{entity_type}/")
    
    def create_cluster_visualizations(self, crops, labels, entity_type):
        """Create visualization PNG showing all 9 clusters"""
        grid_size = 3
        crop_size = 224
        grid_img = np.zeros((grid_size * crop_size, grid_size * crop_size, 3), dtype=np.uint8)
        
        # Get representative image for each cluster
        for cluster_id in range(self.n_races):
            cluster_crops = [crop for crop, label in zip(crops, labels) if label == cluster_id]
            
            if not cluster_crops:
                continue
            
            repr_crop = cluster_crops[0]
            
            row = cluster_id // grid_size
            col = cluster_id % grid_size
            
            y_start = row * crop_size
            y_end = y_start + crop_size
            x_start = col * crop_size
            x_end = x_start + crop_size
            
            grid_img[y_start:y_end, x_start:x_end] = repr_crop
            
            cv2.putText(grid_img, f"Cluster {cluster_id}", 
                       (x_start + 10, y_start + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        output_path = f"clustered_ref_images/{entity_type}_clusters.png"
        cv2.imwrite(output_path, grid_img)
        print(f"   🖼️  Created {entity_type} cluster visualization: {output_path}")
    
    def cluster_entities_with_background(self, embeddings, crops, entity_type):
        """Cluster with background class (0-8=races, 9=background)"""
        n_crops = len(crops)
        
        # If no embeddings, assign sequential labels
        if len(embeddings) == 0 or embeddings.size == 0:
            print(f"❌ No embeddings for {entity_type} - using sequential assignment")
            return np.arange(n_crops) % self.n_races
        
        # If embeddings don't match crops, assign sequential
        if len(embeddings) != n_crops:
            print(f"❌ Embedding/crop mismatch for {entity_type}: {len(embeddings)} vs {n_crops}")
            return np.arange(n_crops) % self.n_races
        
        try:
            # Cluster into n_races groups
            kmeans = KMeans(n_clusters=self.n_races, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            # Calculate silhouette scores to identify poor fits
            from sklearn.metrics import silhouette_samples
            silhouette_scores = silhouette_samples(embeddings, labels)
            
            # Assign poor fits to background class (9)
            background_threshold = 0.1  # Low silhouette = poor fit
            poor_fit_mask = silhouette_scores < background_threshold
            labels[poor_fit_mask] = 9  # Background class
            
            # Stats
            n_background = np.sum(poor_fit_mask)
            unique, counts = np.unique(labels, return_counts=True)
            print(f"   ✅ {entity_type}: {n_crops-n_background} clustered, {n_background} background")
            print(f"   📊 Distribution: {dict(zip(unique, counts))}")
            
            return labels
            
        except Exception as e:
            print(f"❌ Clustering failed for {entity_type}: {e}")
            print(f"   🔄 Using sequential assignment")
            return np.arange(n_crops) % self.n_races
            
        except Exception as e:
            print(f"❌ Clustering failed for {entity_type}: {e}")
            return np.arange(len(embeddings)) % n_clusters
    
    def store_cluster_templates(self, embeddings, labels, entity_type):
        """Store cluster embeddings as templates for SigLIP similarity"""
        cluster_embeddings = []
        
        for cluster_id in range(self.n_races):
            cluster_mask = labels == cluster_id
            if np.any(cluster_mask):
                # Use mean embedding for cluster
                cluster_embedding = np.mean(embeddings[cluster_mask], axis=0)
                cluster_embeddings.append(cluster_embedding)
            else:
                cluster_embeddings.append(None)
        
        if entity_type == "horses":
            self.horse_embeddings = cluster_embeddings
        else:
            self.jockey_embeddings = cluster_embeddings
        
        valid_clusters = sum(1 for emb in cluster_embeddings if emb is not None)
        print(f"   🎯 Stored {valid_clusters} {entity_type} cluster templates")
    
    def classify_detections(self, frame, detections, detection_type='horse'):
        """Classify detections using SigLIP cosine similarity"""
        if not sv or len(detections) == 0:
            return np.arange(len(detections))
        
        templates = self.horse_embeddings if detection_type == 'horse' else self.jockey_embeddings
        threshold = self.horse_threshold if detection_type == 'horse' else self.jockey_threshold
        
        if not templates:
            return np.arange(len(detections))
        
        class_ids = []
        
        for bbox in detections.xyxy:
            # Extract multi-scale crops
            crops = self.extract_multiscale_crops(frame, bbox)
            if not crops:
                class_ids.append(-1)
                continue
            
            # Get embeddings for all scales
            embeddings = []
            for crop in crops:
                embedding = self.extract_embedding(crop)
                if embedding is not None:
                    embeddings.append(embedding)
            
            if not embeddings:
                class_ids.append(-1)
                continue
            
            # Average embeddings across scales
            avg_embedding = np.mean(embeddings, axis=0)
            
            # Find best match using cosine similarity
            best_class = -1
            best_similarity = 0
            
            for i, template_embedding in enumerate(templates):
                if template_embedding is not None:
                    similarity = self.calculate_similarity(avg_embedding, template_embedding)
                    if similarity > best_similarity and similarity > threshold:
                        best_similarity = similarity
                        best_class = i
            
            class_ids.append(best_class)
        
        return np.array(class_ids)
    
    def calculate_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity"""
        try:
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 < 1e-8 or norm2 < 1e-8:
                return 0.0
            
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(np.clip(similarity, -1.0, 1.0))
            
        except Exception:
            return 0.0
    
    def update_tracker_ids(self, detections, class_ids, detection_type='horse'):
        """Update tracker IDs based on classification"""
        if not sv or not hasattr(detections, 'tracker_id') or detections.tracker_id is None:
            return detections
        
        updated_detections = sv.Detections(
            xyxy=detections.xyxy.copy(),
            confidence=detections.confidence.copy() if hasattr(detections, 'confidence') else None,
            class_id=detections.class_id.copy() if hasattr(detections, 'class_id') else None,
            tracker_id=detections.tracker_id.copy()
        )
        
        # Update tracker IDs
        for i, class_id in enumerate(class_ids):
            if class_id >= 0:
                if detection_type == 'horse':
                    updated_detections.tracker_id[i] = 100 + class_id
                else:
                    updated_detections.tracker_id[i] = 200 + class_id
        
        return updated_detections
    
    def get_classification_stats(self):
        """Get classification statistics"""
        return {
            'n_races': self.n_races,
            'horse_accuracy': getattr(self, 'horse_accuracy', None),
            'jockey_accuracy': getattr(self, 'jockey_accuracy', None),
            'horse_threshold': self.horse_threshold,
            'jockey_threshold': self.jockey_threshold,
            'scales': self.scales,
            'siglip_available': self.siglip_model is not None
        }