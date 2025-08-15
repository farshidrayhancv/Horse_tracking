import yaml
import json
from pathlib import Path

class Config:
    def __init__(self, config_file: str = None):
        # ===== BASIC CONFIGURATION =====
        self.video_path = None
        self.output_path = None
        self.display = False
        self.device = "cuda"
        self.max_frames = None
        
        # ===== ROBOFLOW API CONFIGURATION =====
        self.roboflow_api_key = None
        self.roboflow_model_id = None
        self.roboflow_confidence = 0.5
        
        # ===== POSE ESTIMATION CONFIGURATION =====
        self.horse_pose_estimator = 'superanimal'  # superanimal, vitpose, both
        self.confidence_horse_pose_superanimal = 0.3
        self.confidence_horse_pose_vitpose = 0.3
        
        # ===== TRACKING CONFIGURATION =====
        self.tracker_type = 'deepocsort'
        
        # Deep OC-SORT parameters
        self.deepocsort_config = {
            'max_age': 100,
            'min_hits': 5,
            'det_thresh': 0.65,
            'iou_threshold': 0.3,
            'per_class': False,
            'delta_t': 3,
            'inertia': 0.3,
            'Q_xy_scaling': 0.01,
            'Q_s_scaling': 0.0001,
            'asso_func': 'iou',
            'w_association_emb': 0.6,
            'alpha_fixed_emb': 0.9,
            'aw_param': 0.5,
            'embedding_off': True,
            'cmc_off': False,
            'aw_off': False,
        }
        
        # ===== REID CONFIGURATION =====
        self.enable_reid = True
        self.reid_similarity_threshold = 0.3
        self.reid_memory_size = 15
        
        # ===== PERFORMANCE TUNING =====
        self.max_racers = 10
        
        # Load from file if provided
        if config_file:
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: str):
        """Load configuration from YAML or JSON file"""
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
            
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    loaded_settings.append(key)
                else:
                    # Add new settings dynamically
                    setattr(self, key, value)
                    loaded_settings.append(f"{key} (new)")
            
            print(f"✅ Configuration loaded from {config_file}")
            print(f"   📋 Loaded {len(loaded_settings)} settings")
            
            # Show key configurations
            print(f"   🐴 Roboflow Model: {getattr(self, 'roboflow_model_id', 'not set')}")
            print(f"   🎯 Tracking Method: {self.tracker_type.upper()}")
            print(f"   🦴 Pose Estimation: {self.horse_pose_estimator}")
            print(f"   🔄 ReID: {'ENABLED' if self.enable_reid else 'DISABLED'}")
                
        except Exception as e:
            print(f"❌ Error loading config file: {e}")
            import traceback
            traceback.print_exc()
    
    def print_config(self):
        """Print current configuration"""
        print("\n🔧 Current Configuration:")
        print(f"   Roboflow Model: {getattr(self, 'roboflow_model_id', 'not set')}")
        print(f"   Roboflow Confidence: {self.roboflow_confidence}")
        print(f"   Pose Estimator: {self.horse_pose_estimator}")
        print(f"   Pose Confidence (SuperAnimal): {self.confidence_horse_pose_superanimal}")
        print(f"   Pose Confidence (ViTPose): {self.confidence_horse_pose_vitpose}")
        print(f"   Tracker: {self.tracker_type.upper()}")
        print(f"   ReID: {'ENABLED' if self.enable_reid else 'DISABLED'}")
        print(f"   Device: {self.device}")
        print(f"   Display: {self.display}")
        if self.output_path:
            print(f"   Output: {self.output_path}")
    
    def create_template_config(self, filename: str = "racer_config_template.yaml"):
        """Create template config for new racer detection system"""
        template = {
            '# Basic Settings': None,
            'video_path': 'inputs/race_video.mp4',
            'output_path': None,
            'display': False,
            'device': 'cuda',
            'max_frames': None,
            
            '# Roboflow Configuration': None,
            'roboflow_api_key': 'your_roboflow_api_key_here',
            'roboflow_model_id': 'your_model_id/version',
            'roboflow_confidence': 0.5,
            
            '# Pose Estimation': None,
            'horse_pose_estimator': 'both',  # superanimal, vitpose, both
            'confidence_horse_pose_superanimal': 0.3,
            'confidence_horse_pose_vitpose': 0.3,
            
            '# Tracking Configuration': None,
            'tracker_type': 'deepocsort',
            'deepocsort_config': {
                'max_age': 100,
                'min_hits': 5,
                'det_thresh': 0.65,
                'iou_threshold': 0.3,
                'per_class': False,
                'delta_t': 3,
                'inertia': 0.3,
                'Q_xy_scaling': 0.01,
                'Q_s_scaling': 0.0001,
                'asso_func': 'iou',
                'w_association_emb': 0.6,
                'alpha_fixed_emb': 0.9,
                'aw_param': 0.5,
                'embedding_off': True,
                'cmc_off': False,
                'aw_off': False,
            },
            
            '# ReID Configuration': None,
            'enable_reid': True,
            'reid_similarity_threshold': 0.3,
            'reid_memory_size': 15,
            
            '# Performance': None,
            'max_racers': 10
        }
        
        try:
            with open(filename, 'w') as f:
                f.write("# Simplified Racer Detection Configuration\n")
                f.write("# Single Roboflow model detects horse+jockey as compound entity\n\n")
                
                for key, value in template.items():
                    if key.startswith('#'):
                        f.write(f"\n{key}\n")
                    elif value is not None:
                        if isinstance(value, dict):
                            f.write(f"{key}:\n")
                            for sub_key, sub_value in value.items():
                                f.write(f"  {sub_key}: {sub_value}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                        
            print(f"✅ Template config created: {filename}")
        except Exception as e:
            print(f"❌ Error creating template: {e}")