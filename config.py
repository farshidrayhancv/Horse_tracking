import yaml
import json
from pathlib import Path

class Config:
    def __init__(self, config_file: str = None):
        # Available detectors
        self.HUMAN_DETECTORS = {
            'rtdetr': 'RT-DETR (HuggingFace)',
            'superanimal': 'SuperAnimal Faster R-CNN (fallback only)'
        }
        
        self.HORSE_DETECTORS = {
            'rtdetr': 'RT-DETR (HuggingFace)', 
            'superanimal': 'SuperAnimal Faster R-CNN',
            'both': 'RT-DETR primary + SuperAnimal fallback'
        }
        
        self.HUMAN_POSE_ESTIMATORS = {
            'vitpose': 'ViTPose 17 keypoints (HuggingFace)',
            'none': 'No human pose estimation'
        }
        
        self.HORSE_POSE_ESTIMATORS = {
            'superanimal': 'SuperAnimal 39 keypoints only',
            'vitpose': 'ViTPose 17 keypoints only (treats horses as humans)',
            'dual': 'Dual competition: SuperAnimal vs ViTPose (best confidence wins)',
            'none': 'No horse pose estimation'
        }
        
        # SAM model selection
        self.SAM_MODELS = {
            'mobilesam': 'MobileSAM (lightweight, faster)',
            'sam2': 'SAM2 (Meta\'s latest, more accurate)',
            'none': 'No segmentation (simple crops only)'
        }
        
        # ===== BASIC CONFIGURATION =====
        self.video_path = None
        self.output_path = None
        self.display = False
        self.device = "cuda"
        self.max_frames = None
        
        # Model selection defaults
        self.human_detector = 'rtdetr'
        self.horse_detector = 'rtdetr'
        self.human_pose_estimator = 'vitpose'
        self.horse_pose_estimator = 'superanimal'
        self.sam_model = 'sam2'
        
        # ===== CONFIDENCE THRESHOLDS =====
        self.confidence_human_detection = 0.5
        self.confidence_horse_detection = 0.5
        self.confidence_human_pose = 0.5
        self.confidence_horse_pose_superanimal = 0.5
        self.confidence_horse_pose_vitpose = 0.5
        
        # ===== BASIC SETTINGS =====
        self.jockey_overlap_threshold = 0.4
        
        # ===== SAMURAI TRACKING SETTINGS =====
        # Core ReID Pipeline
        self.enable_reid_pipeline = True
        self.reid_similarity_threshold = 0.3
        self.reid_embedding_history_size = 15
        
        # Component toggles
        self.enable_mobile_sam = True  # Backward compatibility
        self.enable_depth_anything = True
        self.enable_megadescriptor = True
        
        # ===== SAMURAI MOTION-AWARE TRACKING =====
        self.samurai_motion_weight = 0.5
        self.samurai_memory_size = 50
        self.samurai_similarity_threshold = 0.15
        self.samurai_max_lost_frames = 25
        
        # ===== DISTANCE AND MOTION SETTINGS =====
        self.motion_distance_threshold = 200
        self.acceleration_prediction = True
        self.visual_memory_enabled = True
        
        # ===== TRACK MANAGEMENT SETTINGS =====
        self.max_tracks_per_frame = 15
        self.track_consolidation_enabled = True
        self.consolidation_similarity_threshold = 0.7
        
        # ===== PERFORMANCE TUNING =====
        self.rgb_weight = 0.7
        self.depth_weight = 0.3
        self.depth_shape_consistency = True
        self.consistency_bonus_threshold = 0.4
        
        # ===== ADVANCED SAMURAI SETTINGS =====
        self.samurai_template_update_threshold = 0.6
        self.visual_recovery_threshold = 0.3
        self.motion_weight = 0.3  # Alternative name for samurai_motion_weight
        self.feature_weight_by_confidence = True
        self.motion_prediction_enabled = True
        
        # ===== VISUALIZATION SETTINGS =====
        self.show_motion_predictions = True
        self.show_track_history = True
        self.show_confidence_scores = True
        self.visualization_history_length = 10
        
        # ===== MEMORY AND CLEANUP SETTINGS =====
        self.track_memory_cleanup_interval = 50
        self.max_inactive_tracks = 100
        self.confidence_decay_rate = 0.95
        
        # ===== DETECTION AND FILTERING =====
        self.detection_overlap_threshold = 0.3
        self.min_detection_confidence = 0.3
        
        # ===== QUALITY AND VALIDATION =====
        self.mask_quality_threshold = 0.1
        self.temporal_consistency_weight = 0.2
        self.spatial_consistency_weight = 0.3
        
        # Load from file if provided
        if config_file:
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: str):
        """Enhanced config loading that handles ANY setting from YAML"""
        config_path = Path(config_file)
        if not config_path.exists():
            print(f"⚠️ Config file '{config_file}' not found, using defaults")
            return
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    print(f"⚠️ Unsupported config file format: {config_path.suffix}")
                    return
            
            if not config_data:
                print(f"⚠️ Config file is empty or invalid")
                return
            
            # Load ALL values from config file
            loaded_settings = []
            new_settings = []
            
            for key, value in config_data.items():
                if hasattr(self, key):
                    # Setting exists in class - update it
                    setattr(self, key, value)
                    loaded_settings.append(key)
                else:
                    # NEW setting from YAML - add it dynamically
                    setattr(self, key, value)
                    new_settings.append(key)
            
            print(f"✅ Configuration loaded from {config_file}")
            print(f"   📋 Loaded {len(loaded_settings)} existing settings")
            
            if new_settings:
                print(f"   🆕 Added {len(new_settings)} new settings from YAML:")
                for setting in new_settings:
                    print(f"      - {setting}: {getattr(self, setting)}")
            
            # Show SAMURAI status
            if getattr(self, 'enable_reid_pipeline', False):
                sam_model = getattr(self, 'sam_model', 'none')
                print(f"   🎯 SAMURAI tracking: ENABLED with {sam_model.upper()}")
                
                # Validate key SAMURAI settings
                key_settings = [
                    'motion_distance_threshold', 
                    'acceleration_prediction',
                    'visual_memory_enabled',
                    'max_tracks_per_frame',
                    'track_consolidation_enabled'
                ]
                
                print(f"   🔧 Key SAMURAI settings loaded:")
                for setting in key_settings:
                    if hasattr(self, setting):
                        value = getattr(self, setting)
                        print(f"      - {setting}: {value}")
                    else:
                        print(f"      - {setting}: NOT SET (will use default)")
            else:
                print(f"   🎯 SAMURAI tracking: DISABLED")
                
        except Exception as e:
            print(f"❌ Error loading config file: {e}")
            import traceback
            traceback.print_exc()
    
    def get_all_settings(self):
        """Get all current settings as a dictionary"""
        settings = {}
        for key in dir(self):
            if not key.startswith('_') and not key.isupper() and not callable(getattr(self, key)):
                settings[key] = getattr(self, key)
        return settings
    
    def validate_samurai_settings(self):
        """Validate that all SAMURAI settings are properly loaded"""
        required_settings = {
            'motion_distance_threshold': 200,
            'acceleration_prediction': True,
            'visual_memory_enabled': True,
            'max_tracks_per_frame': 15,
            'track_consolidation_enabled': True,
            'consolidation_similarity_threshold': 0.7,
            'samurai_motion_weight': 0.5,
            'samurai_memory_size': 50,
            'samurai_similarity_threshold': 0.15,
            'samurai_max_lost_frames': 25
        }
        
        missing_settings = []
        for setting, default_value in required_settings.items():
            if not hasattr(self, setting):
                setattr(self, setting, default_value)
                missing_settings.append(setting)
        
        if missing_settings:
            print(f"⚠️ Added missing SAMURAI settings with defaults:")
            for setting in missing_settings:
                print(f"   - {setting}: {getattr(self, setting)}")
        
        return len(missing_settings) == 0
    
    def enable_samurai_features(self):
        """Enable all SAMURAI features with optimal settings"""
        self.enable_reid_pipeline = True
        self.sam_model = 'sam2'
        self.samurai_motion_weight = 0.5
        self.samurai_memory_size = 50
        self.reid_similarity_threshold = 0.25
        self.motion_distance_threshold = 200
        self.acceleration_prediction = True
        self.visual_memory_enabled = True
        self.track_consolidation_enabled = True
        self.enable_depth_anything = True
        self.enable_megadescriptor = True
        print("✅ SAMURAI features enabled with optimal settings")
    
    def disable_samurai_features(self):
        """Disable SAMURAI features for faster processing"""
        self.enable_reid_pipeline = False
        self.sam_model = 'none'
        self.enable_depth_anything = False
        self.enable_megadescriptor = False
        self.track_consolidation_enabled = False
        print("✅ SAMURAI features disabled for faster processing")
    
    def set_performance_mode(self, mode: str):
        """Set performance mode: 'speed', 'balanced', or 'accuracy'"""
        if mode == 'speed':
            self.sam_model = 'mobilesam'
            self.samurai_memory_size = 20
            self.reid_embedding_history_size = 8
            self.enable_depth_anything = False
            self.motion_distance_threshold = 150
            self.max_tracks_per_frame = 12
            print("🚀 Performance mode: SPEED (faster processing)")
            
        elif mode == 'balanced':
            self.sam_model = 'sam2'
            self.samurai_memory_size = 30
            self.reid_embedding_history_size = 12
            self.enable_depth_anything = True
            self.motion_distance_threshold = 200
            self.max_tracks_per_frame = 15
            print("⚖️ Performance mode: BALANCED (good speed + accuracy)")
            
        elif mode == 'accuracy':
            self.sam_model = 'sam2'
            self.samurai_memory_size = 50
            self.reid_embedding_history_size = 20
            self.enable_depth_anything = True
            self.samurai_max_lost_frames = 30
            self.motion_distance_threshold = 250
            self.max_tracks_per_frame = 20
            print("🎯 Performance mode: ACCURACY (best tracking quality)")
            
        else:
            print(f"❌ Invalid mode. Available: 'speed', 'balanced', 'accuracy'")
    
    def set_human_detector(self, detector: str):
        if detector in self.HUMAN_DETECTORS:
            self.human_detector = detector
            print(f"✅ Human detector: {self.HUMAN_DETECTORS[detector]}")
        else:
            print(f"❌ Invalid human detector. Available: {list(self.HUMAN_DETECTORS.keys())}")
    
    def set_horse_detector(self, detector: str):
        if detector in self.HORSE_DETECTORS:
            self.horse_detector = detector
            print(f"✅ Horse detector: {self.HORSE_DETECTORS[detector]}")
        else:
            print(f"❌ Invalid horse detector. Available: {list(self.HORSE_DETECTORS.keys())}")
    
    def set_human_pose_estimator(self, estimator: str):
        if estimator in self.HUMAN_POSE_ESTIMATORS:
            self.human_pose_estimator = estimator
            print(f"✅ Human pose estimator: {self.HUMAN_POSE_ESTIMATORS[estimator]}")
        else:
            print(f"❌ Invalid human pose estimator. Available: {list(self.HUMAN_POSE_ESTIMATORS.keys())}")
    
    def set_horse_pose_estimator(self, estimator: str):
        if estimator in self.HORSE_POSE_ESTIMATORS:
            self.horse_pose_estimator = estimator
            print(f"✅ Horse pose estimator: {self.HORSE_POSE_ESTIMATORS[estimator]}")
        else:
            print(f"❌ Invalid horse pose estimator. Available: {list(self.HORSE_POSE_ESTIMATORS.keys())}")
    
    def set_sam_model(self, model: str):
        if model in self.SAM_MODELS:
            self.sam_model = model
            print(f"✅ SAM model: {self.SAM_MODELS[model]}")
        else:
            print(f"❌ Invalid SAM model. Available: {list(self.SAM_MODELS.keys())}")
    
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for all models"""
        self.confidence_human_detection = threshold
        self.confidence_horse_detection = threshold
        self.confidence_human_pose = threshold
        self.confidence_horse_pose_superanimal = threshold
        self.confidence_horse_pose_vitpose = threshold
        print(f"✅ All confidence thresholds set to {threshold}")
    
    def print_config(self):
        print("\n🔧 Current Configuration:")
        print(f"   Human detector: {self.HUMAN_DETECTORS[self.human_detector]}")
        print(f"   Horse detector: {self.HORSE_DETECTORS[self.horse_detector]}")
        print(f"   Human pose: {self.HUMAN_POSE_ESTIMATORS[self.human_pose_estimator]}")
        print(f"   Horse pose: {self.HORSE_POSE_ESTIMATORS[self.horse_pose_estimator]}")
        print(f"   SAM model: {self.SAM_MODELS[self.sam_model]}")
        print(f"   Confidence - Human detection: {self.confidence_human_detection}")
        print(f"   Confidence - Horse detection: {self.confidence_horse_detection}")
        print(f"   Confidence - Human pose: {self.confidence_human_pose}")
        print(f"   Confidence - Horse pose (SuperAnimal): {self.confidence_horse_pose_superanimal}")
        print(f"   Confidence - Horse pose (ViTPose): {self.confidence_horse_pose_vitpose}")
        print(f"   Device: {self.device}")
        print(f"   Display: {self.display}")
        
        # SAMURAI status with ALL settings
        if getattr(self, 'enable_reid_pipeline', False):
            print(f"\n🎯 SAMURAI Enhanced Tracking: ENABLED")
            
            # Core settings
            print(f"   Core Settings:")
            print(f"   - SAM model: {getattr(self, 'sam_model', 'not set')}")
            print(f"   - Motion weight: {getattr(self, 'samurai_motion_weight', 'not set')}")
            print(f"   - Memory size: {getattr(self, 'samurai_memory_size', 'not set')} frames")
            print(f"   - Similarity threshold: {getattr(self, 'samurai_similarity_threshold', 'not set')}")
            print(f"   - Max lost frames: {getattr(self, 'samurai_max_lost_frames', 'not set')}")
            
            # Motion settings
            print(f"   Motion & Distance:")
            print(f"   - Distance threshold: {getattr(self, 'motion_distance_threshold', 'not set')} pixels")
            print(f"   - Acceleration prediction: {getattr(self, 'acceleration_prediction', 'not set')}")
            print(f"   - Visual memory: {getattr(self, 'visual_memory_enabled', 'not set')}")
            
            # Track management
            print(f"   Track Management:")
            print(f"   - Max tracks/frame: {getattr(self, 'max_tracks_per_frame', 'not set')}")
            print(f"   - Track consolidation: {getattr(self, 'track_consolidation_enabled', 'not set')}")
            print(f"   - Consolidation threshold: {getattr(self, 'consolidation_similarity_threshold', 'not set')}")
            
            # RGB-D fusion
            print(f"   RGB-D Fusion:")
            print(f"   - RGB weight: {getattr(self, 'rgb_weight', 'not set')}")
            print(f"   - Depth weight: {getattr(self, 'depth_weight', 'not set')}")
            print(f"   - Depth consistency: {getattr(self, 'depth_shape_consistency', 'not set')}")
            
            # Components
            print(f"   Components:")
            print(f"   - Depth-Anything: {getattr(self, 'enable_depth_anything', 'not set')}")
            print(f"   - MegaDescriptor: {getattr(self, 'enable_megadescriptor', 'not set')}")
            
        else:
            print(f"\n🎯 SAMURAI Enhanced Tracking: DISABLED")
    
    def print_all_settings(self):
        """Print ALL loaded settings for debugging"""
        print("\n📋 ALL LOADED SETTINGS:")
        settings = self.get_all_settings()
        
        # Group settings by category
        categories = {
            'Basic': ['video_path', 'output_path', 'display', 'device', 'max_frames'],
            'Models': ['human_detector', 'horse_detector', 'human_pose_estimator', 'horse_pose_estimator', 'sam_model'],
            'Confidence': [k for k in settings.keys() if 'confidence' in k],
            'SAMURAI Core': [k for k in settings.keys() if 'samurai' in k or 'reid' in k],
            'Motion': [k for k in settings.keys() if 'motion' in k or 'acceleration' in k or 'visual_memory' in k],
            'Tracking': [k for k in settings.keys() if 'track' in k or 'consolidation' in k],
            'Performance': [k for k in settings.keys() if any(word in k for word in ['rgb', 'depth', 'weight', 'threshold'])],
            'Other': []
        }
        
        # Categorize all settings
        categorized = set()
        for category, keys in categories.items():
            if category != 'Other':
                categorized.update(keys)
        
        # Add uncategorized settings to 'Other'
        for key in settings.keys():
            if key not in categorized:
                categories['Other'].append(key)
        
        # Print by category
        for category, keys in categories.items():
            if not keys:
                continue
            print(f"\n  {category}:")
            for key in sorted(keys):
                if key in settings:
                    value = settings[key]
                    print(f"    {key}: {value}")
    
    def create_template_config(self, filename: str = "template_config.yaml"):
        """Create a template config with all available settings"""
        template = {
            '# Basic Settings': None,
            'video_path': 'inputs/your_video.mp4',
            'output_path': None,
            'display': False,
            'device': 'cuda',
            'max_frames': None,
            
            '# Model Selection': None,
            'human_detector': 'rtdetr',
            'horse_detector': 'rtdetr', 
            'human_pose_estimator': 'vitpose',
            'horse_pose_estimator': 'superanimal',
            'sam_model': 'sam2',
            
            '# Confidence Thresholds': None,
            'confidence_human_detection': 0.5,
            'confidence_horse_detection': 0.5,
            'confidence_human_pose': 0.5,
            'confidence_horse_pose_superanimal': 0.5,
            'confidence_horse_pose_vitpose': 0.5,
            
            '# Basic Settings': None,
            'jockey_overlap_threshold': 0.4,
            
            '# SAMURAI Tracking': None,
            'enable_reid_pipeline': True,
            'samurai_motion_weight': 0.5,
            'samurai_memory_size': 50,
            'samurai_similarity_threshold': 0.15,
            'samurai_max_lost_frames': 25,
            
            '# Motion and Distance': None,
            'motion_distance_threshold': 200,
            'acceleration_prediction': True,
            'visual_memory_enabled': True,
            
            '# Track Management': None,
            'max_tracks_per_frame': 15,
            'track_consolidation_enabled': True,
            'consolidation_similarity_threshold': 0.7,
            
            '# RGB-D Fusion': None,
            'reid_similarity_threshold': 0.3,
            'reid_embedding_history_size': 15,
            'enable_depth_anything': True,
            'enable_megadescriptor': True,
            'rgb_weight': 0.7,
            'depth_weight': 0.3,
            'depth_shape_consistency': True,
            'consistency_bonus_threshold': 0.4
        }
        
        try:
            with open(filename, 'w') as f:
                f.write("# Complete SAMURAI Configuration Template\n")
                f.write("# All available settings with descriptions\n\n")
                
                for key, value in template.items():
                    if key.startswith('#'):
                        f.write(f"\n{key}\n")
                    elif value is not None:
                        f.write(f"{key}: {value}\n")
                        
            print(f"✅ Template config created: {filename}")
        except Exception as e:
            print(f"❌ Error creating template: {e}")