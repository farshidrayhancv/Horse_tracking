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
        
        # ===== MODEL SELECTION =====
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
        
        # ===== DETECTION LIMITS =====
        self.jockey_overlap_threshold = 0.4
        self.max_tracks_per_frame = 11
        
        # ===== BYTETRACK PARAMETERS =====
        self.bytetrack_track_activation_threshold = 0.6
        self.bytetrack_lost_track_buffer = 50
        self.bytetrack_minimum_matching_threshold = 0.6
        self.bytetrack_minimum_consecutive_frames = 5
        
        # ===== ENHANCED REID PIPELINE CORE =====
        self.enable_reid_pipeline = True
        self.reid_similarity_threshold = 0.3
        
        # ===== TRACK STABILITY & ASSIGNMENT THRESHOLDS =====
        self.track_stability_threshold = 0.4        # Threshold for considering track unstable
        self.track_newness_threshold = 5            # Frames to consider track "new"
        self.initial_assignment_threshold = 0.5     # Easy initial assignment to memory tracks
        self.reassignment_threshold = 0.7           # Hard to steal existing tracks (hysteresis)
        
        # ===== STABILITY CONTROLS (Anti-Oscillation) =====
        self.cooling_period = 10                    # Frames to lock after reassignment
        self.oscillation_threshold = 3             # Max oscillations before penalty
        self.stability_bonus_factor = 0.2          # Bonus for stable tracks
        self.stability_penalty_factor = 0.5        # Penalty for oscillating tracks
        
        # ===== QUALITY MONITORING =====
        self.quality_stability_window = 10         # Window for calculating stability metrics
        self.confidence_variance_threshold = 0.2   # Max allowed confidence variance
        self.position_variance_threshold = 100     # Max allowed position variance
        self.min_valid_keypoints = 5               # Minimum keypoints for valid pose
        
        # ===== MOTION PREDICTION =====
        self.motion_distance_threshold = 150       # Motion prediction threshold (pixels)
        self.max_motion_speed = 30.0               # Maximum speed limit (pixels/frame)
        self.motion_weight = 0.25                  # Weight of motion in similarity calculation
        self.motion_prediction_frames = 3          # Frames to use for motion prediction
        
        # ===== MEMORY MANAGEMENT =====
        self.samurai_memory_size = 15              # Memory size for track history
        self.memory_cleanup_interval = 50          # Frames between memory cleanup
        self.track_expiry_frames = 100             # Frames before track expires
        
        # ===== FEATURE EXTRACTION & SIMILARITY =====
        self.visual_feature_weight = 0.6           # Weight of visual features
        self.motion_feature_weight = 0.25          # Weight of motion features  
        self.stability_feature_weight = 0.15       # Weight of stability features
        self.cosine_similarity_weight = 0.7        # Weight of cosine similarity
        self.l2_similarity_weight = 0.3            # Weight of L2 distance similarity
        
        # ===== SAM SEGMENTATION PARAMETERS =====
        self.sam_multimask_output = True           # Generate multiple mask candidates
        self.sam_center_point_weight = 1.0         # Weight for center point prompt
        self.sam_bbox_prompt_enabled = True        # Use bbox as prompt
        self.sam_confidence_threshold = 0.5        # Minimum SAM mask confidence
        
        # ===== DEPTH PROCESSING =====
        self.enable_depth_anything = True          # Enable depth estimation
        self.depth_weight = 0.3                    # Weight of depth features in RGB-D fusion
        self.depth_variance_threshold = 1000       # Max depth variance for valid mask
        self.depth_normalization_range = 255       # Depth map normalization range
        
        # ===== CROP & FEATURE EXTRACTION =====
        self.crop_padding_factor = 0.05            # Padding around crops (5%)
        self.min_crop_size = 50                    # Minimum crop size (pixels)
        self.feature_vector_size = 64              # Size of extracted feature vectors
        self.color_histogram_bins = 8              # Bins for color histograms
        
        # ===== PERFORMANCE & DEBUG =====
        self.debug_logging = True                  # Enable comprehensive debug logging
        self.save_intermediate_crops = False       # Save RGB/depth crops for debugging
        self.visualization_enabled = True          # Enable enhanced visualizations
        self.progress_bar_enabled = True           # Show progress bars
        
        # ===== SCENARIO-SPECIFIC TUNING =====
        # These will be overridden by scenario configs
        self.scenario_name = "default"
        self.expected_duration_seconds = 120
        self.average_speed_mph = 35
        self.camera_setup = "broadcast"
        self.track_type = "oval"
        
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
                for setting in new_settings[:5]:  # Show first 5
                    print(f"      - {setting}: {getattr(self, setting)}")
                if len(new_settings) > 5:
                    print(f"      ... and {len(new_settings) - 5} more")
            
            # Show scenario-specific info if available
            if hasattr(self, 'scenario_name'):
                print(f"   🎯 Scenario: {self.scenario_name}")
                if hasattr(self, 'expected_duration_seconds'):
                    print(f"   ⏱️ Duration: {self.expected_duration_seconds}s @ {getattr(self, 'average_speed_mph', 'unknown')}mph")
                
        except Exception as e:
            print(f"❌ Error loading config file: {e}")
            import traceback
            traceback.print_exc()
    
    def apply_horse_racing_preset(self):
        """Apply optimized parameters for horse racing scenario"""
        print("🏇 Applying HORSE RACING preset...")
        
        # Higher confidence thresholds for stable tracking
        self.confidence_horse_detection = 0.8
        self.confidence_human_detection = 0.7
        self.confidence_horse_pose_superanimal = 0.6
        
        # Stricter track stability (less aggressive reassignment)
        self.track_stability_threshold = 0.6        # Higher = more stable required
        self.track_newness_threshold = 8            # Longer grace period for new tracks
        self.initial_assignment_threshold = 0.6     # Harder initial assignment
        self.reassignment_threshold = 0.8           # Much harder to steal tracks
        
        # Enhanced stability controls
        self.cooling_period = 20                    # Longer cooling after reassignment
        self.oscillation_threshold = 2             # Lower tolerance for oscillation
        
        # Motion parameters for 35mph horses
        self.motion_distance_threshold = 400       # Much larger for fast movement
        self.max_motion_speed = 50.0               # Higher speed limit
        self.motion_weight = 0.35                  # Higher weight for motion consistency
        
        # Longer memory for occlusions
        self.samurai_memory_size = 25              # Remember tracks longer
        self.track_expiry_frames = 150             # Keep tracks alive longer
        
        # Quality monitoring for broadcast footage
        self.quality_stability_window = 15         # Longer window for stability
        self.bytetrack_lost_track_buffer = 75      # Keep lost tracks longer
        
        # Feature extraction optimized for horses
        self.visual_feature_weight = 0.5           # Reduce visual weight (occlusions)
        self.motion_feature_weight = 0.35          # Increase motion weight
        self.stability_feature_weight = 0.15       # Keep stability weight
        
        print("✅ Horse racing preset applied - optimized for fast movement and broadcast footage")
    
    def apply_speed_preset(self):
        """Apply settings optimized for processing speed"""
        print("🚀 Applying SPEED preset...")
        
        self.sam_model = 'mobilesam'
        self.samurai_memory_size = 8
        self.enable_depth_anything = False
        self.motion_distance_threshold = 200
        self.quality_stability_window = 5
        self.debug_logging = False
        self.save_intermediate_crops = False
        
        print("✅ Speed preset applied")
    
    def apply_accuracy_preset(self):
        """Apply settings optimized for maximum accuracy"""
        print("🎯 Applying ACCURACY preset...")
        
        self.sam_model = 'sam2'
        self.samurai_memory_size = 30
        self.enable_depth_anything = True
        self.motion_distance_threshold = 300
        self.quality_stability_window = 20
        self.cooling_period = 25
        self.track_stability_threshold = 0.7
        
        print("✅ Accuracy preset applied")
    
    def create_horse_racing_config(self, filename: str = "horse_racing_optimized.yaml"):
        """Create optimized config for horse racing"""
        config_data = {
            # Header
            '# Horse Racing Optimized Configuration': None,
            '# Tuned for 35mph horses on oval track with broadcast cameras': None,
            '# Expected race duration: ~120 seconds': None,
            '': None,
            
            # Basic settings
            'scenario_name': 'horse_racing',
            'expected_duration_seconds': 120,
            'average_speed_mph': 35,
            'camera_setup': 'broadcast',
            'track_type': 'oval',
            '': None,
            
            # Video settings
            'video_path': 'inputs/horse_race.mp4',
            'output_path': None,
            'display': False,
            'device': 'cuda',
            'max_frames': None,
            '': None,
            
            # Model selection
            'human_detector': 'rtdetr',
            'horse_detector': 'rtdetr',
            'human_pose_estimator': 'vitpose', 
            'horse_pose_estimator': 'superanimal',
            'sam_model': 'sam2',
            '': None,
            
            # HIGHER confidence thresholds for stable tracking
            'confidence_horse_detection': 0.8,
            'confidence_human_detection': 0.7,
            'confidence_horse_pose_superanimal': 0.6,
            'confidence_horse_pose_vitpose': 0.5,
            'confidence_human_pose': 0.5,
            '': None,
            
            # STRICTER track stability (prevent fragmentation)
            'track_stability_threshold': 0.6,
            'track_newness_threshold': 8,
            'initial_assignment_threshold': 0.6,
            'reassignment_threshold': 0.8,
            '': None,
            
            # ENHANCED stability controls (prevent oscillation)
            'cooling_period': 20,
            'oscillation_threshold': 2,
            'stability_bonus_factor': 0.3,
            'stability_penalty_factor': 0.6,
            '': None,
            
            # MOTION parameters for 35mph horses
            'motion_distance_threshold': 400,
            'max_motion_speed': 50.0,
            'motion_weight': 0.35,
            'motion_prediction_frames': 3,
            '': None,
            
            # LONGER memory for occlusions
            'samurai_memory_size': 25,
            'memory_cleanup_interval': 75,
            'track_expiry_frames': 150,
            '': None,
            
            # QUALITY monitoring for broadcast
            'quality_stability_window': 15,
            'confidence_variance_threshold': 0.15,
            'position_variance_threshold': 150,
            '': None,
            
            # BYTETRACK tuning for fast objects
            'bytetrack_track_activation_threshold': 0.7,
            'bytetrack_lost_track_buffer': 75,
            'bytetrack_minimum_matching_threshold': 0.7,
            'bytetrack_minimum_consecutive_frames': 3,
            '': None,
            
            # FEATURE weights optimized for horses
            'visual_feature_weight': 0.5,
            'motion_feature_weight': 0.35,
            'stability_feature_weight': 0.15,
            '': None,
            
            # Enhanced ReID pipeline
            'enable_reid_pipeline': True,
            'reid_similarity_threshold': 0.4,
            'enable_depth_anything': True,
            '': None,
            
            # Performance
            'max_tracks_per_frame': 11,
            'debug_logging': True,
            'visualization_enabled': True,
            'progress_bar_enabled': True
        }
        
        try:
            with open(filename, 'w') as f:
                f.write("# Horse Racing Optimized Configuration\n")
                f.write("# Tuned for 35mph horses, 120s races, broadcast cameras\n")
                f.write("# Key optimizations:\n")
                f.write("#   - Higher confidence thresholds (reduce false detections)\n")
                f.write("#   - Stricter stability controls (prevent track fragmentation)\n") 
                f.write("#   - Larger motion thresholds (handle 35mph movement)\n")
                f.write("#   - Longer memory (handle occlusions)\n")
                f.write("#   - Anti-oscillation controls (prevent rapid reassignments)\n\n")
                
                for key, value in config_data.items():
                    if key.startswith('#') or key == '':
                        if key:
                            f.write(f"{key}\n")
                        else:
                            f.write("\n")
                    elif value is not None:
                        f.write(f"{key}: {value}\n")
                        
            print(f"✅ Horse racing config created: {filename}")
            print(f"🎯 Key optimizations:")
            print(f"   - Motion threshold: 400px (vs 150px default) for 35mph horses")
            print(f"   - Cooling period: 20 frames (vs 10) to prevent oscillation")
            print(f"   - Stability threshold: 0.6 (vs 0.4) for stable tracking")
            print(f"   - Memory size: 25 frames (vs 15) for occlusion handling")
            
        except Exception as e:
            print(f"❌ Error creating config: {e}")
    
    def print_config(self):
        """Print current configuration with scenario analysis"""
        print("\n🔧 Current Configuration:")
        print(f"   Scenario: {getattr(self, 'scenario_name', 'default')}")
        if hasattr(self, 'expected_duration_seconds'):
            duration = self.expected_duration_seconds
            speed = getattr(self, 'average_speed_mph', 0)
            print(f"   Expected: {duration}s race @ {speed}mph")
        
        print(f"\n🎛️ Model Pipeline:")
        print(f"   Human detector: {self.HUMAN_DETECTORS[self.human_detector]}")
        print(f"   Horse detector: {self.HORSE_DETECTORS[self.horse_detector]}")
        print(f"   Human pose: {self.HUMAN_POSE_ESTIMATORS[self.human_pose_estimator]}")
        print(f"   Horse pose: {self.HORSE_POSE_ESTIMATORS[self.horse_pose_estimator]}")
        print(f"   SAM model: {self.SAM_MODELS[self.sam_model]}")
        print(f"   Device: {self.device}")
        
        print(f"\n🎯 Tracking Stability:")
        print(f"   Track stability threshold: {self.track_stability_threshold}")
        print(f"   Cooling period: {self.cooling_period} frames")
        print(f"   Reassignment threshold: {self.reassignment_threshold}")
        print(f"   Motion distance threshold: {self.motion_distance_threshold}px")
        
        # Analyze configuration for scenario
        if hasattr(self, 'average_speed_mph') and self.average_speed_mph > 0:
            self._analyze_motion_parameters()
    
    def _analyze_motion_parameters(self):
        """Analyze if motion parameters are appropriate for scenario"""
        speed_mph = getattr(self, 'average_speed_mph', 0)
        
        if speed_mph > 0:
            # Rough estimation: 35mph ≈ 15.6 m/s
            # At 30fps, that's ~0.52m per frame
            # If 1 pixel ≈ 2cm, that's ~26 pixels per frame movement
            estimated_movement_per_frame = speed_mph * 0.74  # rough conversion
            
            print(f"\n📊 Motion Analysis:")
            print(f"   Estimated movement: ~{estimated_movement_per_frame:.1f} pixels/frame")
            print(f"   Current threshold: {self.motion_distance_threshold}px")
            
            if self.motion_distance_threshold < estimated_movement_per_frame * 3:
                print(f"   ⚠️ WARNING: Motion threshold may be too low for {speed_mph}mph")
                recommended = int(estimated_movement_per_frame * 8)
                print(f"   💡 Recommend: motion_distance_threshold: {recommended}")
            else:
                print(f"   ✅ Motion threshold appropriate for scenario")