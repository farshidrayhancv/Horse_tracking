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
        
        # Default configuration
        self.human_detector = 'rtdetr'
        self.horse_detector = 'both'
        self.human_pose_estimator = 'vitpose'
        self.horse_pose_estimator = 'dual'

        
        # Separate confidence thresholds for each model/task
        self.confidence_human_detection = 0.3
        self.confidence_horse_detection = 0.3
        self.confidence_human_pose = 0.3
        self.confidence_horse_pose_superanimal = 0.3
        self.confidence_horse_pose_vitpose = 0.3
        
        self.jockey_overlap_threshold = 0.4
        self.display = False
        self.max_frames = None
        self.output_path = None
        self.device = "cpu"
        self.video_path = None


        self.enable_reid_pipeline = False
        self.reid_similarity_threshold = 0.7
        self.reid_embedding_history_size = 5
        self.enable_mobile_sam = True
        self.enable_depth_anything = True
        self.enable_megadescriptor = True
        
        if config_file:
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: str):
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
            
            # Load all values
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            print(f"✅ Configuration loaded from {config_file}")
        except Exception as e:
            print(f"❌ Error loading config file: {e}")
    
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
    
    def set_confidence_threshold(self, threshold: float):
        self.confidence_human_detection = threshold
        self.confidence_horse_detection = threshold
        self.confidence_human_pose = threshold
        self.confidence_horse_pose_superanimal = threshold
        self.confidence_horse_pose_vitpose = threshold
    
    def print_config(self):
        print("\n🔧 Current Configuration:")
        print(f"   Human detector: {self.HUMAN_DETECTORS[self.human_detector]}")
        print(f"   Horse detector: {self.HORSE_DETECTORS[self.horse_detector]}")
        print(f"   Human pose: {self.HUMAN_POSE_ESTIMATORS[self.human_pose_estimator]}")
        print(f"   Horse pose: {self.HORSE_POSE_ESTIMATORS[self.horse_pose_estimator]}")
        print(f"   Confidence - Human detection: {self.confidence_human_detection}")
        print(f"   Confidence - Horse detection: {self.confidence_horse_detection}")
        print(f"   Confidence - Human pose: {self.confidence_human_pose}")
        print(f"   Confidence - Horse pose (SuperAnimal): {self.confidence_horse_pose_superanimal}")
        print(f"   Confidence - Horse pose (ViTPose): {self.confidence_horse_pose_vitpose}")
        print(f"   Device: {self.device}")
        print(f"   Display: {self.display}")