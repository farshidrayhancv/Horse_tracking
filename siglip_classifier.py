"""
Enhanced SigLIP-based Horse Classification with OCR Number Detection
Uses OCR to detect horse numbers (0-9) instead of clustering
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

try:
    import supervision as sv
except ImportError:
    sv = None

try:
    from transformers import AutoProcessor, SiglipVisionModel, AutoProcessor, AutoModelForImageClassification
    SIGLIP_AVAILABLE = True
except ImportError:
    SIGLIP_AVAILABLE = False

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False
    print("❌ TrOCR not available - install with: pip install transformers")

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

class SigLIPClassifier:
    """SigLIP-based classification with OCR number detection for horses only"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # Initialize SigLIP
        self.siglip_model = None
        self.siglip_processor = None
        self.setup_siglip()
        
        # Initialize TrOCR
        self.trocr_processor = None
        self.trocr_model = None
        self.setup_trocr()
        
        # Horse classifier only
        self.horse_classifier = None
        
        # Horse embeddings and labels (dynamic based on OCR detection)
        self.horse_embeddings = []
        self.horse_labels = []
        
        # Dynamic class mapping for detected horses
        self.class_mapping = {}      # horse_number -> classifier_index
        self.reverse_mapping = {}    # classifier_index -> horse_number  
        self.valid_classes = set()   # set of horse numbers with sufficient data
        self.accuracy = None         # classification accuracy
        
        # Classification threshold
        self.horse_threshold = getattr(config, 'siglip_confidence_threshold', 0.8)
        
        # Multi-scale settings
        self.scales = getattr(config, 'crop_scales', [1.0, 1.2, 0.8])
        
        # Build training data and train classifier
        self.build_training_pipeline()
        
        print(f"✅ SigLIP Classifier with TrOCR initialized")
        print(f"   🐴 Horse samples: {len(self.horse_embeddings)}")
        print(f"   🔢 TrOCR numbers detected: {len([l for l in self.horse_labels if l >= 0])}")
    
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
    
    def setup_trocr(self):
        """Initialize TrOCR model"""
        if not TROCR_AVAILABLE:
            print("❌ TrOCR not available - install transformers")
            return
        
        try:
            model_name = "microsoft/dit-base-finetuned-rvlcdip"
            self.trocr_processor =  AutoProcessor.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
            self.trocr_model = AutoModelForImageClassification.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
            self.trocr_model.to(self.device)
            self.trocr_model.eval()
            print("✅ TrOCR loaded")
        except Exception as e:
            print(f"❌ TrOCR setup failed: {e}")
    
    def detect_horse_number(self, crop):
        """TrOCR-based horse number detection - CLEAN IMPLEMENTATION"""
        if not self.trocr_model or not self.trocr_processor:
            return -1
        
        try:
            # Convert BGR to RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            from PIL import Image
            pil_image = Image.fromarray(crop_rgb.astype(np.uint8))
            
            # TrOCR processing
            pixel_values = self.trocr_processor(pil_image, return_tensors="pt").pixel_values.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.trocr_model.generate(pixel_values)
            
            generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Extract first digit
            clean_text = ''.join(filter(str.isdigit, generated_text.strip()))
            
            if len(clean_text) >= 1:
                number = int(clean_text[0])
                if 0 <= number <= 9:
                    return number
            
            return -1
            
        except Exception as e:
            return -1
    
    
    def build_training_pipeline(self):
        """Complete training pipeline: video sampling → detection → OCR labeling → classifier training"""
        video_path = Path(self.config.video_path)
        if not video_path.exists():
            print(f"❌ Video file not found: {video_path}")
            return
        
        print(f"🎬 Processing video: {video_path}")
        
        # Step 1: Sample video and extract horse crops
        all_horse_crops = self.process_video_frames(video_path)
        
        if not all_horse_crops:
            print("❌ No horse crops found")
            return
        
        # Step 2: Extract embeddings and OCR labels
        print(f"   🔄 Extracting embeddings for {len(all_horse_crops)} horse crops...")
        horse_embeddings = self.extract_batch_embeddings(all_horse_crops)
        
        print(f"   🔢 Running OCR on {len(all_horse_crops)} horse crops...")
        horse_labels = self.extract_ocr_labels(all_horse_crops)
        
        # Step 3: Filter valid samples (where OCR detected a number)
        valid_indices = [i for i, label in enumerate(horse_labels) if label >= 0]
        
        if len(valid_indices) < 10:
            print(f"❌ Insufficient OCR detections: {len(valid_indices)} valid samples")
            return
        
        valid_embeddings = horse_embeddings[valid_indices]
        valid_labels = [horse_labels[i] for i in valid_indices]
        
        print(f"   ✅ Valid samples: {len(valid_indices)}/{len(all_horse_crops)}")
        print(f"   📊 Label distribution: {dict(zip(*np.unique(valid_labels, return_counts=True)))}")
        
        # Step 4: Save crops by OCR labels
        self.save_labeled_crops(all_horse_crops, horse_labels)
        
        # Step 5: Train/test validation
        self.train_and_validate_classifier(valid_embeddings, valid_labels)
        
        # Step 6: Store for inference
        self.horse_embeddings = valid_embeddings
        self.horse_labels = valid_labels
    
    def process_video_frames(self, video_path):
        """Sample video at 10 FPS and extract horse crops"""
        from detectors import DetectionManager
        from models import SuperAnimalQuadruped
        
        # Setup detection
        superanimal = SuperAnimalQuadruped(device=self.device, config=self.config)
        detection_manager = DetectionManager(self.config, superanimal)
        
        all_horse_crops = []
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample every 15 seconds (1 frame per 15 seconds)
        frame_interval = int(fps * 15)  # frames per 15 seconds
        sampled_frames = 0
        
        print(f"   📹 Video: {fps:.1f} FPS, {total_frames} frames")
        print(f"   🎯 Sampling every {frame_interval} frames (1 frame per 15 seconds)")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample at 10 FPS interval
            if frame_count % frame_interval == 0:
                # Detect horses only
                horse_detections = detection_manager.detect_horses(frame)
                
                # Extract crops with multi-scale
                if sv and len(horse_detections) > 0:
                    for bbox in horse_detections.xyxy:
                        crops = self.extract_multiscale_crops(frame, bbox)
                        all_horse_crops.extend(crops)
                
                sampled_frames += 1
                
                # Limit to avoid too many samples
                if sampled_frames >= 200:  # Max 200 sampled frames
                    break
            
            frame_count += 1
        
        cap.release()
        
        print(f"   🎬 Processed {sampled_frames} frames, extracted {len(all_horse_crops)} horse crops")
        return all_horse_crops
    
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
    
    def extract_ocr_labels(self, crops):
        """Extract TrOCR labels and save crops"""
        labels = []
        
        # Create inspection directory
        inspect_dir = Path("crop_inspection")
        inspect_dir.mkdir(exist_ok=True)
        
        detected_dir = inspect_dir / "detected"
        failed_dir = inspect_dir / "failed"
        detected_dir.mkdir(exist_ok=True)
        failed_dir.mkdir(exist_ok=True)
        
        print(f"   💾 Saving crops to {inspect_dir} for inspection")
        
        for i, crop in enumerate(crops):
            number = self.detect_horse_number(crop)
            labels.append(number)
            
            # Save crop
            if number >= 0:
                crop_path = detected_dir / f"crop_{i:04d}_num_{number}.jpg"
                print(f"   🔢 Crop {i}: Detected number {number}")
            else:
                crop_path = failed_dir / f"crop_{i:04d}_failed.jpg"
            
            cv2.imwrite(str(crop_path), crop)
        
        detected_count = sum(1 for l in labels if l >= 0)
        print(f"   📊 TrOCR Results: {detected_count}/{len(crops)} detected")
        
        return labels
    
    def save_labeled_crops(self, crops, labels):
        """Save crops organized by OCR labels"""
        output_dir = Path("labeled_horse_crops")
        output_dir.mkdir(exist_ok=True)
        
        # Create number subdirectories (0-9 + unknown)
        for number in range(10):
            number_dir = output_dir / f"horse_{number}"
            number_dir.mkdir(exist_ok=True)
        
        unknown_dir = output_dir / "unknown"
        unknown_dir.mkdir(exist_ok=True)
        
        # Save crops to respective directories
        for i, (crop, label) in enumerate(zip(crops, labels)):
            if label >= 0:
                target_dir = output_dir / f"horse_{label}"
            else:
                target_dir = unknown_dir
            
            crop_path = target_dir / f"crop_{i:04d}.jpg"
            cv2.imwrite(str(crop_path), crop)
        
        valid_count = sum(1 for l in labels if l >= 0)
        print(f"   💾 Saved {len(crops)} crops: {valid_count} labeled, {len(crops)-valid_count} unknown")
    
    def train_and_validate_classifier(self, embeddings, labels):
        """Train classifier with dynamic class handling"""
        if len(embeddings) < 4:
            print("❌ Insufficient data for training (need at least 4 samples)")
            return
        
        # Filter classes with insufficient samples (need at least 2 for train/test split)
        from collections import Counter
        label_counts = Counter(labels)
        valid_classes = [cls for cls, count in label_counts.items() if count >= 2]
        
        if len(valid_classes) < 2:
            print("❌ Need at least 2 classes with 2+ samples each")
            return
        
        # Filter data to only include valid classes
        valid_indices = [i for i, label in enumerate(labels) if label in valid_classes]
        filtered_embeddings = embeddings[valid_indices]
        filtered_labels = [labels[i] for i in valid_indices]
        
        print(f"   🎯 Training on {len(valid_classes)} classes: {sorted(valid_classes)}")
        print(f"   📊 Filtered data: {len(filtered_embeddings)} samples")
        
        # Create class mapping for dynamic classes
        self.class_mapping = {cls: i for i, cls in enumerate(sorted(valid_classes))}
        self.reverse_mapping = {i: cls for cls, i in self.class_mapping.items()}
        
        # Map labels to sequential indices for classifier
        mapped_labels = [self.class_mapping[label] for label in filtered_labels]
        
        # Train/test split with stratification
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                filtered_embeddings, mapped_labels, test_size=0.3, random_state=42, stratify=mapped_labels
            )
        except ValueError:
            # Fallback without stratification if still issues
            X_train, X_test, y_train, y_test = train_test_split(
                filtered_embeddings, mapped_labels, test_size=0.3, random_state=42
            )
            print("   ⚠️ Using non-stratified split due to class imbalance")
        
        # Train classifier
        self.horse_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.horse_classifier.fit(X_train, y_train)
        
        # Validate
        y_pred = self.horse_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"   🎯 Classifier accuracy: {accuracy:.3f}")
        
        if accuracy >= 0.9:
            print("   ✅ Target accuracy (90%) achieved!")
        else:
            print(f"   ⚠️ Below target accuracy (90%). Current: {accuracy:.1%}")
        
        # Save classifier and mappings
        classifier_data = {
            'classifier': self.horse_classifier,
            'class_mapping': self.class_mapping,
            'reverse_mapping': self.reverse_mapping,
            'accuracy': accuracy
        }
        joblib.dump(classifier_data, "horse_number_classifier.pkl")
        print("   💾 Classifier and mappings saved")
        
        # Store valid classes for inference
        self.valid_classes = set(valid_classes)
        print(f"   🐴 Trackable horses: {sorted(valid_classes)}")
    
    def classify_detections(self, frame, detections, detection_type='horse'):
        """Classify horse detections using SigLIP + OCR with dynamic classes"""
        if not sv or len(detections) == 0 or detection_type != 'horse':
            return np.full(len(detections), -1) if len(detections) > 0 else np.array([])
        
        if not self.horse_classifier or len(self.horse_embeddings) == 0:
            return np.full(len(detections), -1)
        
        # Check if we have valid classes from training
        if not hasattr(self, 'reverse_mapping') or not hasattr(self, 'valid_classes'):
            print("❌ No valid classes available for classification")
            return np.full(len(detections), -1)
        
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
            
            # Classify using trained classifier
            try:
                # Get prediction (this will be mapped class index)
                mapped_prediction = self.horse_classifier.predict([avg_embedding])[0]
                confidence = self.horse_classifier.predict_proba([avg_embedding]).max()
                
                # Map back to original horse number
                if mapped_prediction in self.reverse_mapping and confidence > self.horse_threshold:
                    original_horse_number = self.reverse_mapping[mapped_prediction]
                    class_ids.append(original_horse_number)
                else:
                    class_ids.append(-1)
                    
            except Exception as e:
                class_ids.append(-1)
        
        return np.array(class_ids)
    
    def update_tracker_ids(self, detections, class_ids, detection_type='horse'):
        """Update tracker IDs based on classification for horses only"""
        if not sv or not hasattr(detections, 'tracker_id') or detections.tracker_id is None:
            return detections
        
        if detection_type != 'horse':
            return detections
        
        updated_detections = sv.Detections(
            xyxy=detections.xyxy.copy(),
            confidence=detections.confidence.copy() if hasattr(detections, 'confidence') else None,
            class_id=detections.class_id.copy() if hasattr(detections, 'class_id') else None,
            tracker_id=detections.tracker_id.copy()
        )
        
        # Update tracker IDs for horses (use OCR number directly as ID)
        for i, class_id in enumerate(class_ids):
            if class_id >= 0:
                updated_detections.tracker_id[i] = class_id  # Use OCR number as tracking ID
        
        return updated_detections
    
    def get_classification_stats(self):
        """Get classification statistics for dynamic classes"""
        valid_labels = [l for l in self.horse_labels if l >= 0]
        trackable_horses = getattr(self, 'valid_classes', set())
        accuracy = getattr(self, 'accuracy', None)
        
        return {
            'horse_samples': len(self.horse_embeddings),
            'valid_ocr_detections': len(valid_labels),
            'trackable_horses': sorted(list(trackable_horses)),
            'num_trackable': len(trackable_horses),
            'accuracy': accuracy,
            'horse_threshold': self.horse_threshold,
            'scales': self.scales,
            'trocr_available': self.trocr_model is not None,
            'siglip_available': self.siglip_model is not None
        }