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
        
        # ===== CONFIDENCE THRESHOLDS =====
        self.confidence_human_detection = 0.5
        self.confidence_horse_detection = 0.5
        self.confidence_human_pose = 0.5
        self.confidence_horse_pose_superanimal = 0.5
        self.confidence_horse_pose_vitpose = 0.5
        
        # ===== BASIC SETTINGS =====
        self.jockey_overlap_threshold = 0.4
        
        # ===== BOOSTTRACK CONFIGURATION =====
        self.tracker_type = 'boosttrack'  # 'boosttrack' or 'bytetrack'
        
        # BoostTrack parameters
        self.boosttrack_config = {
            'max_age': 60,
            'min_hits': 3,
            'det_thresh': 0.6,
            'iou_threshold': 0.3,
            'use_ecc': True,
            'min_box_area': 10,
            'aspect_ratio_thresh': 1.6,
            'cmc_method': 'ecc',
            'lambda_iou': 0.5,
            'lambda_mhd': 0.25,
            'lambda_shape': 0.25,
            'use_dlo_boost': True,
            'use_duo_boost': True,
            'dlo_boost_coef': 0.65,
            's_sim_corr': False,
            'use_rich_s': False,
            'use_sb': False,
            'use_vt': False,
            'with_reid': False
        }
        
        # ===== SIGLIP CLASSIFICATION =====
        self.enable_siglip_classification = True
        self.reference_image_path = "horse_9.png"
        self.max_horses = 9
        self.max_jockeys = 9
        self.siglip_confidence_threshold = 0.8
        
        # ===== PERFORMANCE TUNING =====
        self.max_tracks_per_frame = 9
        
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
                    setattr(self, key, value)
                    loaded_settings.append(key)
                else:
                    setattr(self, key, value)
                    new_settings.append(key)
            
            print(f"✅ Configuration loaded from {config_file}")
            print(f"   📋 Loaded {len(loaded_settings)} existing settings")
            
            if new_settings:
                print(f"   🆕 Added {len(new_settings)} new settings from YAML:")
                for setting in new_settings:
                    print(f"      - {setting}: {getattr(self, setting)}")
            
            # Show tracking method
            tracker_type = getattr(self, 'tracker_type', 'bytetrack')
            print(f"   🎯 Tracking Method: {tracker_type.upper()}")
            
            if tracker_type == 'boosttrack':
                boosttrack_config = getattr(self, 'boosttrack_config', {})
                print(f"   ⚡ BoostTrack Config: {len(boosttrack_config)} parameters")
            
            # Show SigLIP classification status
            if getattr(self, 'enable_siglip_classification', False):
                ref_image = getattr(self, 'reference_image_path', 'not set')
                max_horses = getattr(self, 'max_horses', 9)
                max_jockeys = getattr(self, 'max_jockeys', 9)
                print(f"   🔍 SigLIP Classification: ENABLED")
                print(f"   📷 Reference Image: {ref_image}")
                print(f"   🐴 Max Horses: {max_horses}, Max Jockeys: {max_jockeys}")
            else:
                print(f"   🔍 SigLIP Classification: DISABLED")
                
        except Exception as e:
            print(f"❌ Error loading config file: {e}")
            import traceback
            traceback.print_exc()
    
    def set_performance_mode(self, mode: str):
        """Set performance mode: 'speed', 'balanced', or 'accuracy'"""
        if mode == 'speed':
            self.boosttrack_config.update({
                'track_high_thresh': 0.7,
                'track_low_thresh': 0.2,
                'track_buffer': 20,
                'match_thresh': 0.7
            })
            print("🚀 Performance mode: SPEED")
            
        elif mode == 'balanced':
            self.boosttrack_config.update({
                'track_high_thresh': 0.6,
                'track_low_thresh': 0.1,
                'track_buffer': 30,
                'match_thresh': 0.8
            })
            print("⚖️ Performance mode: BALANCED")
            
        elif mode == 'accuracy':
            self.boosttrack_config.update({
                'track_high_thresh': 0.5,
                'track_low_thresh': 0.05,
                'track_buffer': 50,
                'match_thresh': 0.9
            })
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
        print(f"   Device: {self.device}")
        print(f"   Display: {self.display}")
        
        # Tracking status
        tracker_type = getattr(self, 'tracker_type', 'bytetrack')
        print(f"\n🎯 Tracking Method: {tracker_type.upper()}")
        
        if tracker_type == 'boosttrack':
            print(f"   BoostTrack Settings:")
            for key, value in self.boosttrack_config.items():
                print(f"   - {key}: {value}")
        
        # SigLIP classification status
        if getattr(self, 'enable_siglip_classification', False):
            print(f"\n🔍 SigLIP Classification: ENABLED")
            print(f"   Reference Image: {getattr(self, 'reference_image_path', 'not set')}")
            print(f"   Max Horses: {getattr(self, 'max_horses', 9)}")
            print(f"   Max Jockeys: {getattr(self, 'max_jockeys', 9)}")
            print(f"   Confidence Threshold: {getattr(self, 'siglip_confidence_threshold', 0.3)}")
        else:
            print(f"\n🔍 SigLIP Classification: DISABLED")
    
    def create_template_config(self, filename: str = "boosttrack_config_template.yaml"):
        """Create template config with BoostTrack settings"""
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
            
            '# Confidence Thresholds': None,
            'confidence_human_detection': 0.5,
            'confidence_horse_detection': 0.5,
            'confidence_human_pose': 0.5,
            'confidence_horse_pose_superanimal': 0.5,
            'confidence_horse_pose_vitpose': 0.5,
            
            '# Tracking Method': None,
            'tracker_type': 'boosttrack',
            
            '# BoostTrack Configuration': None,
            'boosttrack_config': {
                'track_high_thresh': 0.6,
                'track_low_thresh': 0.1,
                'new_track_thresh': 0.7,
                'track_buffer': 30,
                'match_thresh': 0.8,
                'proximity_thresh': 0.5,
                'appearance_thresh': 0.25,
                'cmc_method': 'sparseOptFlow',
                'frame_rate': 25,
                'lambda_': 0.98
            },
            
            '# SigLIP Classification': None,
            'enable_siglip_classification': True,
            'reference_image_path': 'horse_9.png',
            'max_horses': 9,
            'max_jockeys': 9,
            'siglip_confidence_threshold': 0.3,
            
            '# Performance': None,
            'max_tracks_per_frame': 9
        }
        
        try:
            with open(filename, 'w') as f:
                f.write("# BoostTrack + SigLIP Configuration Template\n")
                f.write("# Advanced tracking with individual horse/jockey identification\n\n")
                
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
                        
            print(f"✅ BoostTrack template config created: {filename}")
        except Exception as e:
            print(f"❌ Error creating template: {e}")