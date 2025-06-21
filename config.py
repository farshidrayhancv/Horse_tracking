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
        
        # ===== 3D POSE INTEGRATION SETTINGS =====
        # Core 3D Pose Processing
        self.enable_3d_poses = True
        self.enable_3d_reid_features = True
        self.enable_gpu_acceleration_3d = True
        self.enable_batch_processing_3d = True
        
        # Depth Processing and Smoothing
        self.depth_smoothing_algorithm = 'adaptive_gaussian'  # 'gaussian', 'bilateral', 'adaptive_gaussian', 'none'
        self.depth_smoothing_factor = 0.3
        self.depth_temporal_smoothing = True
        self.depth_temporal_window = 5
        self.depth_outlier_threshold = 2.5  # Standard deviations for outlier removal
        self.depth_noise_reduction_strength = 0.7
        
        # Advanced Depth Quality Enhancement
        self.enable_depth_hole_filling = True
        self.depth_hole_filling_method = 'inpaint'  # 'inpaint', 'interpolation', 'nearest'
        self.depth_edge_preservation = True
        self.depth_edge_threshold = 10.0
        
        # 3D Pose Quality Metrics
        self.min_valid_keypoints_3d = 5
        self.pose_3d_quality_threshold = 0.3
        self.enable_geometric_consistency_check = True
        self.geometric_plausibility_threshold = 0.6
        
        # Enhanced 3D Geometric Features for ReID
        self.enable_volumetric_features = True
        self.enable_spatial_distribution_features = True
        self.enable_depth_gradient_features = True
        self.enable_pose_compactness_features = True
        self.enable_temporal_3d_features = True
        
        # 3D Feature Weighting for ReID
        self.visual_feature_weight = 0.35
        self.pose_3d_feature_weight = 0.30
        self.geometric_feature_weight = 0.20
        self.motion_weight = 0.10
        self.stability_weight = 0.05
        
        # GPU Acceleration Settings
        self.gpu_batch_size_3d = 32
        self.enable_gpu_depth_processing = True
        self.enable_gpu_geometric_calculations = True
        self.gpu_memory_optimization = True
        
        # Performance Optimization
        self.parallel_pose_processing = True
        self.max_concurrent_poses = 4
        self.enable_pose_caching = True
        self.pose_cache_size = 50
        
        # ===== 3D VISUALIZATION SETTINGS =====
        # Depth Visualization
        self.depth_visualization_scale = 2.0
        self.pose_3d_transparency = 0.7
        self.enable_depth_color_coding = True
        self.show_3d_pose_quality = True
        
        # 3D Pose Display Options
        self.show_depth_values = True
        self.show_geometric_features = False
        self.depth_color_scheme = 'jet'  # 'jet', 'viridis', 'plasma', 'hot'
        self.pose_3d_point_size_factor = 1.5
        
        # ===== ADVANCED 3D PROCESSING =====
        # Kalman Filtering for 3D Poses
        self.enable_3d_kalman_filtering = True
        self.kalman_process_noise = 0.1
        self.kalman_measurement_noise = 0.5
        
        # 3D Pose Temporal Consistency
        self.enable_temporal_pose_smoothing = True
        self.temporal_smoothing_window = 7
        self.temporal_consistency_weight = 0.3
        
        # Advanced Geometric Analysis
        self.enable_pose_symmetry_analysis = True
        self.enable_limb_length_consistency = True
        self.limb_length_variance_threshold = 0.2
        
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
            
            # Show 3D Pose Integration status
            if getattr(self, 'enable_3d_poses', False):
                depth_algorithm = getattr(self, 'depth_smoothing_algorithm', 'none')
                gpu_accel = getattr(self, 'enable_gpu_acceleration_3d', False)
                batch_proc = getattr(self, 'enable_batch_processing_3d', False)
                print(f"   🎯 3D Pose Integration: ENABLED")
                print(f"      - Depth Smoothing: {depth_algorithm.upper()}")
                print(f"      - GPU Acceleration: {'ON' if gpu_accel else 'OFF'}")
                print(f"      - Batch Processing: {'ON' if batch_proc else 'OFF'}")
                print(f"      - Geometric Features: {'ON' if getattr(self, 'enable_volumetric_features', False) else 'OFF'}")
            else:
                print(f"   🎯 3D Pose Integration: DISABLED")
                
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
    
    def enable_3d_pose_integration(self):
        """Enable 3D pose integration with optimal settings"""
        self.enable_3d_poses = True
        self.enable_3d_reid_features = True
        self.enable_gpu_acceleration_3d = True
        self.enable_batch_processing_3d = True
        self.depth_smoothing_algorithm = 'adaptive_gaussian'
        self.enable_volumetric_features = True
        self.enable_geometric_consistency_check = True
        print("✅ 3D Pose Integration enabled with GPU acceleration")
    
    def disable_enhanced_reid(self):
        """Disable enhanced ReID for faster processing"""
        self.enable_reid_pipeline = False
        self.sam_model = 'none'
        self.enable_depth_anything = False
        self.enable_3d_poses = False
        print("✅ Enhanced ReID features disabled")
    
    def set_performance_mode(self, mode: str):
        """Set performance mode: 'speed', 'balanced', or 'accuracy'"""
        if mode == 'speed':
            self.sam_model = 'mobilesam'
            self.samurai_memory_size = 10
            self.enable_depth_anything = False
            self.motion_distance_threshold = 100
            self.quality_stability_window = 5
            # 3D pose speed optimizations
            self.enable_3d_poses = False
            self.enable_gpu_acceleration_3d = True
            self.gpu_batch_size_3d = 16
            self.depth_smoothing_algorithm = 'none'
            print("🚀 Performance mode: SPEED (3D poses disabled)")
            
        elif mode == 'balanced':
            self.sam_model = 'sam2'
            self.samurai_memory_size = 15
            self.enable_depth_anything = True
            self.motion_distance_threshold = 150
            self.quality_stability_window = 10
            # 3D pose balanced settings
            self.enable_3d_poses = True
            self.enable_gpu_acceleration_3d = True
            self.gpu_batch_size_3d = 8
            self.depth_smoothing_algorithm = 'gaussian'
            self.enable_volumetric_features = True
            print("⚖️ Performance mode: BALANCED (3D poses with GPU acceleration)")
            
        elif mode == 'accuracy':
            self.sam_model = 'sam2'
            self.samurai_memory_size = 20
            self.enable_depth_anything = True
            self.motion_distance_threshold = 200
            self.quality_stability_window = 15
            # 3D pose accuracy settings
            self.enable_3d_poses = True
            self.enable_gpu_acceleration_3d = True
            self.gpu_batch_size_3d = 4
            self.depth_smoothing_algorithm = 'adaptive_gaussian'
            self.enable_volumetric_features = True
            self.enable_geometric_consistency_check = True
            self.enable_temporal_3d_features = True
            print("🎯 Performance mode: ACCURACY (Full 3D pose analysis)")
            
        else:
            print(f"❌ Invalid mode. Available: 'speed', 'balanced', 'accuracy'")
    
    def set_3d_depth_quality(self, quality: str):
        """Set 3D depth processing quality: 'fast', 'standard', 'high'"""
        if quality == 'fast':
            self.depth_smoothing_algorithm = 'gaussian'
            self.depth_temporal_smoothing = False
            self.enable_depth_hole_filling = False
            self.depth_noise_reduction_strength = 0.3
            print("🚀 3D Depth Quality: FAST")
            
        elif quality == 'standard':
            self.depth_smoothing_algorithm = 'adaptive_gaussian'
            self.depth_temporal_smoothing = True
            self.depth_temporal_window = 3
            self.enable_depth_hole_filling = True
            self.depth_hole_filling_method = 'interpolation'
            self.depth_noise_reduction_strength = 0.5
            print("⚖️ 3D Depth Quality: STANDARD")
            
        elif quality == 'high':
            self.depth_smoothing_algorithm = 'adaptive_gaussian'
            self.depth_temporal_smoothing = True
            self.depth_temporal_window = 7
            self.enable_depth_hole_filling = True
            self.depth_hole_filling_method = 'inpaint'
            self.depth_edge_preservation = True
            self.depth_noise_reduction_strength = 0.8
            print("🎯 3D Depth Quality: HIGH")
            
        else:
            print(f"❌ Invalid quality. Available: 'fast', 'standard', 'high'")
    
    def print_config(self):
        """Print current configuration including 3D settings"""
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
        
        # 3D Pose Integration status
        if getattr(self, 'enable_3d_poses', False):
            print(f"\n🎯 3D Pose Integration: ENABLED")
            print(f"   Core 3D Settings:")
            print(f"   - Depth smoothing: {getattr(self, 'depth_smoothing_algorithm', 'not set')}")
            print(f"   - GPU acceleration: {getattr(self, 'enable_gpu_acceleration_3d', 'not set')}")
            print(f"   - Batch processing: {getattr(self, 'enable_batch_processing_3d', 'not set')}")
            print(f"   - GPU batch size: {getattr(self, 'gpu_batch_size_3d', 'not set')}")
            
            print(f"   Quality Enhancement:")
            print(f"   - Temporal smoothing: {getattr(self, 'depth_temporal_smoothing', 'not set')}")
            print(f"   - Hole filling: {getattr(self, 'enable_depth_hole_filling', 'not set')}")
            print(f"   - Edge preservation: {getattr(self, 'depth_edge_preservation', 'not set')}")
            print(f"   - Noise reduction: {getattr(self, 'depth_noise_reduction_strength', 'not set')}")
            
            print(f"   Geometric Features:")
            print(f"   - Volumetric features: {getattr(self, 'enable_volumetric_features', 'not set')}")
            print(f"   - Spatial distribution: {getattr(self, 'enable_spatial_distribution_features', 'not set')}")
            print(f"   - Depth gradients: {getattr(self, 'enable_depth_gradient_features', 'not set')}")
            print(f"   - Temporal 3D: {getattr(self, 'enable_temporal_3d_features', 'not set')}")
            
            print(f"   ReID Feature Weights:")
            print(f"   - Visual: {getattr(self, 'visual_feature_weight', 0):.2f}")
            print(f"   - 3D Pose: {getattr(self, 'pose_3d_feature_weight', 0):.2f}")
            print(f"   - Geometric: {getattr(self, 'geometric_feature_weight', 0):.2f}")
            print(f"   - Motion: {getattr(self, 'motion_weight', 0):.2f}")
            
        else:
            print(f"\n🎯 3D Pose Integration: DISABLED")
    
    def create_template_config(self, filename: str = "enhanced_3d_config_template.yaml"):
        """Create template config with enhanced 3D pose settings"""
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
            
            '# 3D Pose Integration': None,
            'enable_3d_poses': True,
            'enable_3d_reid_features': True,
            'enable_gpu_acceleration_3d': True,
            'enable_batch_processing_3d': True,
            
            '# Depth Processing': None,
            'depth_smoothing_algorithm': 'adaptive_gaussian',
            'depth_smoothing_factor': 0.3,
            'depth_temporal_smoothing': True,
            'depth_temporal_window': 5,
            'depth_outlier_threshold': 2.5,
            'depth_noise_reduction_strength': 0.7,
            
            '# Advanced Depth Quality': None,
            'enable_depth_hole_filling': True,
            'depth_hole_filling_method': 'inpaint',
            'depth_edge_preservation': True,
            'depth_edge_threshold': 10.0,
            
            '# 3D Pose Quality': None,
            'min_valid_keypoints_3d': 5,
            'pose_3d_quality_threshold': 0.3,
            'enable_geometric_consistency_check': True,
            'geometric_plausibility_threshold': 0.6,
            
            '# 3D Geometric Features': None,
            'enable_volumetric_features': True,
            'enable_spatial_distribution_features': True,
            'enable_depth_gradient_features': True,
            'enable_pose_compactness_features': True,
            'enable_temporal_3d_features': True,
            
            '# ReID Feature Weighting': None,
            'visual_feature_weight': 0.35,
            'pose_3d_feature_weight': 0.30,
            'geometric_feature_weight': 0.20,
            'motion_weight': 0.10,
            'stability_weight': 0.05,
            
            '# GPU Acceleration': None,
            'gpu_batch_size_3d': 8,
            'enable_gpu_depth_processing': True,
            'enable_gpu_geometric_calculations': True,
            'gpu_memory_optimization': True,
            
            '# Performance Optimization': None,
            'parallel_pose_processing': True,
            'max_concurrent_poses': 4,
            'enable_pose_caching': True,
            'pose_cache_size': 50,
            
            '# 3D Visualization': None,
            'depth_visualization_scale': 2.0,
            'pose_3d_transparency': 0.7,
            'enable_depth_color_coding': True,
            'show_3d_pose_quality': True,
            'depth_color_scheme': 'jet',
            
            '# Advanced 3D Processing': None,
            'enable_3d_kalman_filtering': True,
            'kalman_process_noise': 0.1,
            'kalman_measurement_noise': 0.5,
            'enable_temporal_pose_smoothing': True,
            'temporal_smoothing_window': 7,
            
            '# Performance': None,
            'max_tracks_per_frame': 11
        }
        
        try:
            with open(filename, 'w') as f:
                f.write("# Enhanced 3D Pose Integration Configuration Template\n")
                f.write("# Advanced ReID with 3D pose features, GPU acceleration, and depth processing\n\n")
                
                for key, value in template.items():
                    if key.startswith('#'):
                        f.write(f"\n{key}\n")
                    elif value is not None:
                        f.write(f"{key}: {value}\n")
                        
            print(f"✅ Enhanced 3D template config created: {filename}")
        except Exception as e:
            print(f"❌ Error creating template: {e}")