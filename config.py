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
        
        # ===== ENHANCED REID PIPELINE SETTINGS =====
        # Core ReID Pipeline
        self.enable_reid_pipeline = True
        self.reid_similarity_threshold = 0.3
        
        # Enhanced Track Quality Monitoring
        self.track_stability_threshold = 0.4  # Threshold for considering track unstable
        self.track_newness_threshold = 5      # Frames to consider track "new"
        self.quality_stability_window = 10    # Window for calculating stability metrics
        
        # Memory and Motion Settings
        self.samurai_memory_size = 15
        self.motion_distance_threshold = 150  # Lower for broadcast footage
        
        # Component toggles
        self.enable_depth_anything = True
        
        # ===== PERFORMANCE TUNING =====
        self.max_tracks_per_frame = 11  # Limit to expected number of horses
        
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
            
            # Show Enhanced ReID status
            if getattr(self, 'enable_reid_pipeline', False):
                sam_model = getattr(self, 'sam_model', 'none')
                print(f"   🎯 Enhanced ReID Pipeline: ENABLED with {sam_model.upper()}")
                print(f"   🧠 Intelligent Track Assignment: Quality-based reassignment")
                print(f"   📊 Track Stability Monitoring: {getattr(self, 'track_stability_threshold', 'not set')}")
            else:
                print(f"   🎯 Enhanced ReID Pipeline: DISABLED")
                
        except Exception as e:
            print(f"❌ Error loading config file: {e}")
            import traceback
            traceback.print_exc()
    
    def enable_enhanced_reid(self):
        """Enable enhanced ReID features with optimal settings"""
        self.enable_reid_pipeline = True
        self.sam_model = 'sam2'
        self.reid_similarity_threshold = 0.3
        self.track_stability_threshold = 0.4
        self.track_newness_threshold = 5
        self.motion_distance_threshold = 150
        self.enable_depth_anything = True
        print("✅ Enhanced ReID features enabled")
    
    def disable_enhanced_reid(self):
        """Disable enhanced ReID for faster processing"""
        self.enable_reid_pipeline = False
        self.sam_model = 'none'
        self.enable_depth_anything = False
        print("✅ Enhanced ReID features disabled")
    
    def set_performance_mode(self, mode: str):
        """Set performance mode: 'speed', 'balanced', or 'accuracy'"""
        if mode == 'speed':
            self.sam_model = 'mobilesam'
            self.samurai_memory_size = 10
            self.enable_depth_anything = False
            self.motion_distance_threshold = 100
            self.quality_stability_window = 5
            print("🚀 Performance mode: SPEED")
            
        elif mode == 'balanced':
            self.sam_model = 'sam2'
            self.samurai_memory_size = 15
            self.enable_depth_anything = True
            self.motion_distance_threshold = 150
            self.quality_stability_window = 10
            print("⚖️ Performance mode: BALANCED")
            
        elif mode == 'accuracy':
            self.sam_model = 'sam2'
            self.samurai_memory_size = 20
            self.enable_depth_anything = True
            self.motion_distance_threshold = 200
            self.quality_stability_window = 15
            print("🎯 Performance mode: ACCURACY")
            
        else:
            print(f"❌ Invalid mode. Available: 'speed', 'balanced', 'accuracy'")
    
    def print_config(self):
        """Print current configuration"""
        print("\n🔧 Current Configuration:")
        print(f"   Human detector: {self.HUMAN_DETECTORS[self.human_detector]}")
        print(f"   Horse detector: {self.HORSE_DETECTORS[self.horse_detector]}")
        print(f"   Human pose: {self.HUMAN_POSE_ESTIMATORS[self.human_pose_estimator]}")
        print(f"   Horse pose: {self.HORSE_POSE_ESTIMATORS[self.horse_pose_estimator]}")
        print(f"   SAM model: {self.SAM_MODELS[self.sam_model]}")
        print(f"   Device: {self.device}")
        print(f"   Display: {self.display}")
        
        # Enhanced ReID status
        if getattr(self, 'enable_reid_pipeline', False):
            print(f"\n🎯 Enhanced ReID Pipeline: ENABLED")
            print(f"   Core Settings:")
            print(f"   - SAM model: {getattr(self, 'sam_model', 'not set')}")
            print(f"   - Similarity threshold: {getattr(self, 'reid_similarity_threshold', 'not set')}")
            print(f"   - Memory size: {getattr(self, 'samurai_memory_size', 'not set')} frames")
            
            print(f"   Quality Monitoring:")
            print(f"   - Stability threshold: {getattr(self, 'track_stability_threshold', 'not set')}")
            print(f"   - Newness threshold: {getattr(self, 'track_newness_threshold', 'not set')} frames")
            print(f"   - Stability window: {getattr(self, 'quality_stability_window', 'not set')} frames")
            
            print(f"   Motion Settings:")
            print(f"   - Distance threshold: {getattr(self, 'motion_distance_threshold', 'not set')} pixels")
            
            print(f"   Components:")
            print(f"   - Depth-Anything: {getattr(self, 'enable_depth_anything', 'not set')}")
            
        else:
            print(f"\n🎯 Enhanced ReID Pipeline: DISABLED")
    
    def create_template_config(self, filename: str = "enhanced_config_template.yaml"):
        """Create template config with enhanced ReID settings"""
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
            
            '# Enhanced ReID Pipeline': None,
            'enable_reid_pipeline': True,
            'reid_similarity_threshold': 0.3,
            
            '# Track Quality Monitoring': None,
            'track_stability_threshold': 0.4,
            'track_newness_threshold': 5,
            'quality_stability_window': 10,
            
            '# Memory and Motion': None,
            'samurai_memory_size': 15,
            'motion_distance_threshold': 150,
            
            '# Components': None,
            'enable_depth_anything': True,
            
            '# Performance': None,
            'max_tracks_per_frame': 11
        }
        
        try:
            with open(filename, 'w') as f:
                f.write("# Enhanced ReID Configuration Template\n")
                f.write("# Intelligent track assignment with quality monitoring\n\n")
                
                for key, value in template.items():
                    if key.startswith('#'):
                        f.write(f"\n{key}\n")
                    elif value is not None:
                        f.write(f"{key}: {value}\n")
                        
            print(f"✅ Enhanced template config created: {filename}")
        except Exception as e:
            print(f"❌ Error creating template: {e}")