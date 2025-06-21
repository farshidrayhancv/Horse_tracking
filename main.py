"""
Complete 3D Enhanced main.py with GPU Acceleration, Batch Processing, and Performance Monitoring
Final integration of all 3D pose features with comprehensive tracking and optimization
"""

import cv2
import numpy as np
import torch
import warnings
import re
import time
from pathlib import Path
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor

# Local imports
from config import Config
from models import SuperAnimalQuadruped
from detectors import DetectionManager
from pose_estimators import PoseEstimationManager
from visualizers import Visualizer
from reid_pipeline import EnhancedReIDPipeline
from debug_logger import TrackingDebugLogger

# Suppress warnings but keep errors and status prints
warnings.filterwarnings("ignore")

# Check dependencies
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    from transformers import AutoProcessor, VitPoseForPoseEstimation, RTDetrForObjectDetection
    VITPOSE_AVAILABLE = True
except ImportError:
    VITPOSE_AVAILABLE = False

try:
    from dlclibrary import download_huggingface_model
    SUPERANIMAL_AVAILABLE = True
except ImportError:
    SUPERANIMAL_AVAILABLE = False

try:
    import supervision as sv
    from supervision import ByteTrack
except ImportError:
    sv = None


class PerformanceMonitor:
    """Real-time performance monitoring for 3D pose processing"""
    
    def __init__(self, config):
        self.config = config
        self.frame_times = []
        self.component_times = {}
        self.gpu_memory_usage = []
        self.cpu_usage = []
        self.memory_usage = []
        
        # Component timing
        self.timing_contexts = {}
        
    def start_timing(self, component: str):
        """Start timing a component"""
        self.timing_contexts[component] = time.time()
    
    def end_timing(self, component: str) -> float:
        """End timing and return duration"""
        if component in self.timing_contexts:
            duration = time.time() - self.timing_contexts[component]
            if component not in self.component_times:
                self.component_times[component] = []
            self.component_times[component].append(duration)
            return duration
        return 0.0
    
    def update_system_metrics(self):
        """Update system resource usage metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        self.cpu_usage.append(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.memory_usage.append(memory.used / 1024 / 1024)  # MB
        
        # GPU memory (if available)
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            self.gpu_memory_usage.append(gpu_memory)
        else:
            self.gpu_memory_usage.append(0)
    
    def get_performance_summary(self) -> dict:
        """Get performance summary"""
        summary = {
            'avg_frame_time': np.mean(self.frame_times) if self.frame_times else 0,
            'avg_fps': 1.0 / np.mean(self.frame_times) if self.frame_times else 0,
            'component_times': {},
            'resource_usage': {
                'avg_cpu_percent': np.mean(self.cpu_usage) if self.cpu_usage else 0,
                'avg_memory_mb': np.mean(self.memory_usage) if self.memory_usage else 0,
                'avg_gpu_memory_mb': np.mean(self.gpu_memory_usage) if self.gpu_memory_usage else 0,
                'peak_gpu_memory_mb': np.max(self.gpu_memory_usage) if self.gpu_memory_usage else 0
            }
        }
        
        # Component timing summary
        for component, times in self.component_times.items():
            summary['component_times'][component] = {
                'avg_ms': np.mean(times) * 1000,
                'total_ms': np.sum(times) * 1000,
                'percentage': (np.sum(times) / np.sum(self.frame_times)) * 100 if self.frame_times else 0
            }
        
        return summary
    
    def log_frame_time(self, frame_time: float):
        """Log frame processing time"""
        self.frame_times.append(frame_time)


class HybridPoseSystem:
    def __init__(self, video_path: str, config: Config):
        self.video_path = Path(video_path)
        self.config = config
        
        # Initialize performance monitor
        self.perf_monitor = PerformanceMonitor(config)
        
        # Initialize debug logger
        self.debug_logger = TrackingDebugLogger(self.config)
        self.debug_logger.set_video_name(str(self.video_path))
        
        # Parse expected counts from filename
        self.expected_horses, self.expected_jockeys = self.parse_filename_counts()
        
        # Setup video
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        
        if self.total_frames <= 0:
            self.total_frames = float('inf')
        
        # Setup tracking
        self.setup_trackers()
        
        # Setup models and components
        self.setup_models()
        
        # Setup Enhanced ReID Pipeline with 3D pose integration
        self.reid_pipeline = None
        if getattr(self.config, 'enable_reid_pipeline', True):
            try:
                # Add stability configuration to config object
                self._ensure_config_completeness()
                self.reid_pipeline = EnhancedReIDPipeline(self.config)
            except Exception as e:
                self.config.enable_reid_pipeline = False
        
        # GPU optimization
        if getattr(self.config, 'gpu_memory_optimization', False) and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Print configuration
        self.config.print_config()
        print(f"🎯 Expected: {self.expected_horses} horses, {self.expected_jockeys} jockeys")
        print(f"🐴🏇 3D Enhanced Horse Tracking System ready")
        print(f"📊 Performance monitoring: ENABLED")
    
    def _ensure_config_completeness(self):
        """Ensure all required config parameters exist"""
        defaults = {
            'cooling_period': 10,
            'oscillation_threshold': 3,
            'initial_assignment_threshold': 0.5,
            'reassignment_threshold': 0.7,
            'enable_3d_poses': True,
            'enable_gpu_acceleration_3d': True,
            'enable_batch_processing_3d': True,
            'depth_smoothing_algorithm': 'adaptive_gaussian',
            'gpu_batch_size_3d': 8,
            'parallel_pose_processing': True,
            'max_concurrent_poses': 4
        }
        
        for key, value in defaults.items():
            if not hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def parse_filename_counts(self):
        """Parse filename to extract expected horse/jockey counts"""
        filename = self.video_path.stem
        
        # Look for pattern like horse_11, horse_22, etc.
        match = re.search(r'horse_(\d+)', filename, re.IGNORECASE)
        if match:
            count = int(match.group(1))
            return count, count
        
        return 10, 10  # Default to 10 horses, 10 jockeys
    
    def setup_trackers(self):
        """Initialize ByteTracks for horses and humans"""
        if not sv:
            self.horse_tracker = None
            self.human_tracker = None
            return
        
        # Initialize trackers with video FPS
        self.horse_tracker = ByteTrack(
            frame_rate=int(self.fps),
            track_activation_threshold=0.6,
            lost_track_buffer=50,
            minimum_matching_threshold=0.6,
            minimum_consecutive_frames=5,
        )

        self.human_tracker = ByteTrack(
            frame_rate=int(self.fps),
            track_activation_threshold=0.6,
            lost_track_buffer=50,
            minimum_matching_threshold=0.6,
            minimum_consecutive_frames=5,
        )
    
    def setup_models(self):
        # Setup SuperAnimal model if needed
        self.superanimal = None
        if self.config.horse_detector in ['superanimal', 'both'] or self.config.horse_pose_estimator in ['superanimal', 'dual']:
            self.superanimal = SuperAnimalQuadruped(device=self.config.device, config=self.config)
        
        # Setup detection manager
        self.detection_manager = DetectionManager(self.config, self.superanimal)
        
        # Setup pose estimation manager
        self.pose_manager = PoseEstimationManager(self.config, self.superanimal)
        
        # Setup visualizer
        self.visualizer = Visualizer(self.config, self.superanimal)
    
    def limit_detections(self, detections, max_count, detection_type="object"):
        """Limit detections to expected count, keeping highest confidence ones"""
        if not sv or len(detections) == 0:
            return detections
        
        if len(detections) <= max_count:
            return detections
        
        # Sort by confidence (descending) and take top N
        sorted_indices = np.argsort(detections.confidence)[::-1]
        top_indices = sorted_indices[:max_count]
        
        limited_detections = sv.Detections(
            xyxy=detections.xyxy[top_indices],
            confidence=detections.confidence[top_indices],
            class_id=detections.class_id[top_indices]
        )
        
        return limited_detections
    
    def visualize_motion_predictions(self, frame, tracking_info):
        """Visualize motion predictions"""
        if not tracking_info:
            return frame
        
        motion_predictions = tracking_info.get('motion_predictions', {})
        for track_id, predicted_pos in motion_predictions.items():
            if predicted_pos is not None:
                center = (int(predicted_pos[0]), int(predicted_pos[1]))
                # Draw predicted position as yellow cross
                cv2.drawMarker(frame, center, (0, 255, 255), cv2.MARKER_CROSS, 15, 2)
                cv2.putText(frame, f"P{track_id}", (center[0]+10, center[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        return frame
    
    def visualize_3d_pose_info(self, frame, poses_3d, tracking_info, perf_summary):
        """Enhanced 3D pose information overlay with performance metrics"""
        if not poses_3d:
            return frame
        
        # Count 3D pose statistics
        total_3d_poses = len(poses_3d)
        valid_3d_poses = len([p for p in poses_3d if p and p.get('confidence_3d', 0) > 0.3])
        avg_3d_quality = np.mean([p.get('confidence_3d', 0) for p in poses_3d if p]) if poses_3d else 0
        
        # 3D pose improvements from ReID
        pose_3d_improvements = tracking_info.get('pose_3d_improvements', 0) if tracking_info else 0
        
        # Performance metrics
        avg_fps = perf_summary.get('avg_fps', 0)
        gpu_memory = perf_summary.get('resource_usage', {}).get('avg_gpu_memory_mb', 0)
        
        # Draw enhanced 3D pose info overlay (top-left corner)
        info_lines = [
            f"3D Pose Integration: {total_3d_poses} poses",
            f"Valid 3D: {valid_3d_poses} (>{0.3:.1f} quality)",
            f"Avg 3D Quality: {avg_3d_quality:.3f}",
            f"ReID 3D Improvements: {pose_3d_improvements}",
            f"Performance: {avg_fps:.1f} FPS",
            f"GPU Memory: {gpu_memory:.0f} MB"
        ]
        
        # Semi-transparent background for 3D info
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 10 + len(info_lines) * 25), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 25
            cv2.putText(frame, line, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame
    
    def visualize_segmentation_masks(self, segmentation_masks, tracked_horses, frame_shape):
        """Create visualization showing only segmentation masks on blank background"""
        # Create blank canvas (black background)
        mask_canvas = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)
        
        if segmentation_masks and len(segmentation_masks) > 0:
            # Draw each horse mask with unique colors
            for i, mask in enumerate(segmentation_masks):
                if mask is not None:
                    # Get track-based color
                    if sv and len(tracked_horses) > 0 and i < len(tracked_horses.xyxy):
                        track_id = tracked_horses.tracker_id[i] if hasattr(tracked_horses, 'tracker_id') and i < len(tracked_horses.tracker_id) else i
                        color = self.visualizer.get_track_color(track_id)
                    else:
                        # Fallback colors
                        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                        color = colors[i % len(colors)]
                    
                    # Fill mask area with color
                    mask_canvas[mask] = color
                    
                    # Add white contour outline for better visibility
                    mask_area = mask.astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(mask_canvas, contours, -1, (255, 255, 255), 2)
        
        return mask_canvas

    def visualize_depth_with_masks(self, depth_map, segmentation_masks, tracked_horses):
        """Create depth visualization with highlighted horse segmentation masks"""
        # Convert depth to color map
        depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
        
        if segmentation_masks and sv and len(tracked_horses) > 0:
            # Overlay each horse mask with unique colors
            for i, (mask, box) in enumerate(zip(segmentation_masks, tracked_horses.xyxy)):
                if mask is not None:
                    # Get track-based color
                    track_id = tracked_horses.tracker_id[i] if hasattr(tracked_horses, 'tracker_id') and i < len(tracked_horses.tracker_id) else i
                    color = self.visualizer.get_track_color(track_id)
                    
                    # Create colored mask overlay
                    colored_mask = np.zeros_like(depth_colored)
                    colored_mask[mask] = color
                    
                    # Blend with depth map
                    mask_area = mask.astype(np.uint8) * 255
                    depth_colored = cv2.addWeighted(depth_colored, 0.7, colored_mask, 0.3, 0)
                    
                    # Add contour outline
                    contours, _ = cv2.findContours(mask_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(depth_colored, contours, -1, color, 2)
        
        return depth_colored

    def associate_poses_with_tracks(self, poses, tracked_detections):
        """Associate pose results with tracked detection IDs"""
        if not sv or not poses or len(tracked_detections) == 0:
            return poses
        
        # Add track IDs to pose results
        for i, pose in enumerate(poses):
            if i < len(tracked_detections.tracker_id):
                pose['track_id'] = tracked_detections.tracker_id[i]
            else:
                pose['track_id'] = -1  # Untracked
        
        return poses
    
    def process_video_with_parallel_execution(self):
        """Process video with parallel execution for optimal performance"""
        # Determine output path
        if self.config.output_path:
            output_path = self.config.output_path
        else:
            input_stem = self.video_path.stem if self.video_path.suffix else self.video_path.name
            output_path = str(self.video_path.parent / f"{input_stem}_3d_gpu_enhanced_output.mp4")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        frame_count = 0
        max_frames = self.config.max_frames or (self.total_frames if self.total_frames != float('inf') else 10000)
        paused = False
        
        stats = {
            'humans_detected': 0, 'horses_detected': 0,
            'human_poses': 0, 'horse_poses': 0,
            'human_poses_3d': 0, 'horse_poses_3d': 0,
            'superanimal_wins': 0, 'vitpose_wins': 0,
            'tracked_horses': 0, 'tracked_humans': 0,
            'active_horse_tracks': set(), 'active_human_tracks': set(),
            'reid_reassignments': 0, 'reid_3d_improvements': 0,
            'avg_3d_quality': 0.0, 'gpu_batches_processed': 0,
            'parallel_operations': 0
        }
        
        # Initialize progress bar with enhanced information
        if TQDM_AVAILABLE and not self.config.display and self.total_frames != float('inf'):
            pbar = tqdm(total=max_frames, desc="GPU-Accelerated 3D Processing", 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
        else:
            pbar = None
        
        # Setup display window
        if self.config.display:
            window_name = "GPU-Accelerated 3D Enhanced Horse Racing System"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            display_width = min(1200, self.width)
            display_height = int(self.height * (display_width / self.width))
            cv2.resizeWindow(window_name, display_width, display_height)

        # Thread pool for parallel processing
        max_workers = getattr(self.config, 'max_concurrent_poses', 4)
        executor = ThreadPoolExecutor(max_workers=max_workers) if getattr(self.config, 'parallel_pose_processing', False) else None

        while frame_count < max_frames:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # LOG: Frame start
            frame_start_time = time.time()
            self.debug_logger.log_frame_start(frame_count, frame.shape)
            self.perf_monitor.start_timing('total_frame')
            
            # STEP 1: Detect objects with timing
            self.perf_monitor.start_timing('detection')
            human_detections = self.detection_manager.detect_humans(frame)
            horse_detections = self.detection_manager.detect_horses(frame)
            detection_time = self.perf_monitor.end_timing('detection')
            
            # LOG: Detections
            detection_method = f"H:{self.config.human_detector}/Ho:{self.config.horse_detector}"
            self.debug_logger.log_detections(human_detections, horse_detections, detection_method, detection_time)
            
            # STEP 2: Limit detections to expected counts
            human_detections = self.limit_detections(human_detections, self.expected_jockeys, "jockeys")
            horse_detections = self.limit_detections(horse_detections, self.expected_horses, "horses")
            
            # STEP 3: Track detections with timing
            self.perf_monitor.start_timing('tracking')
            if sv and self.horse_tracker and self.human_tracker:
                tracked_horses = self.horse_tracker.update_with_detections(horse_detections)
                tracked_humans = self.human_tracker.update_with_detections(human_detections)
                
                # Update tracking stats
                if len(tracked_horses) > 0:
                    stats['active_horse_tracks'].update(tracked_horses.tracker_id)
                if len(tracked_humans) > 0:
                    stats['active_human_tracks'].update(tracked_humans.tracker_id)
            else:
                tracked_horses = horse_detections
                tracked_humans = human_detections
            tracking_time = self.perf_monitor.end_timing('tracking')
            
            # LOG: Tracking updates
            self.debug_logger.log_tracking_update(tracked_humans, tracked_horses, "ByteTrack", tracking_time)
            
            # Filter humans to jockeys only
            if sv:
                jockey_detections = self.detection_manager.filter_jockeys(tracked_humans, tracked_horses)
            else:
                jockey_detections = tracked_humans
            
            # STEP 4: Apply Enhanced ReID Pipeline with timing
            self.perf_monitor.start_timing('reid')
            tracking_info = {}
            depth_map = None
            
            if self.reid_pipeline and getattr(self.config, 'enable_reid_pipeline', False):
                # Process horses with Enhanced ReID
                horse_rgb_crops, horse_depth_crops, depth_map, horse_reid_features, horse_depth_stats = self.reid_pipeline.process_frame(frame, tracked_horses)
                
                # Process jockeys with standard pipeline
                jockey_rgb_crops, jockey_depth_crops, _, jockey_reid_features, jockey_depth_stats = self.reid_pipeline.process_frame(frame, jockey_detections)
                
                # LOG: Depth processing
                self.debug_logger.log_depth_processing(depth_map, horse_depth_stats, self.perf_monitor.end_timing('reid'))
            
            # STEP 5: GPU-Accelerated 3D Pose Estimation with timing
            self.perf_monitor.start_timing('pose_estimation_3d')
            
            if getattr(self.config, 'enable_batch_processing_3d', False) and executor:
                # Parallel pose estimation
                stats['parallel_operations'] += 1
                
                # Submit parallel tasks
                future_human_poses = executor.submit(
                    self.pose_manager.estimate_human_poses, frame, jockey_detections, depth_map
                )
                future_horse_poses = executor.submit(
                    self.pose_manager.estimate_horse_poses, frame, tracked_horses, depth_map
                )
                
                # Get results
                human_poses_3d = future_human_poses.result()
                horse_poses_3d = future_horse_poses.result()
            else:
                # Sequential pose estimation
                human_poses_3d = self.pose_manager.estimate_human_poses(frame, jockey_detections, depth_map)
                horse_poses_3d = self.pose_manager.estimate_horse_poses(frame, tracked_horses, depth_map)
            
            pose_3d_time = self.perf_monitor.end_timing('pose_estimation_3d')
            
            # Count GPU batches
            if horse_poses_3d and any('pose_3d_features' in pose for pose in horse_poses_3d if pose):
                stats['gpu_batches_processed'] += 1
            
            # Apply intelligent track assignment with 3D pose features
            if self.reid_pipeline and getattr(self.config, 'enable_reid_pipeline', False):
                self.perf_monitor.start_timing('reid_3d')
                
                tracked_horses_enhanced = self.reid_pipeline.enhance_tracking(
                    tracked_horses, horse_reid_features, horse_depth_stats, horse_poses_3d)
                jockey_detections_enhanced = self.reid_pipeline.enhance_tracking(
                    jockey_detections, jockey_reid_features, jockey_depth_stats, human_poses_3d)
                
                # Get tracking information
                tracking_info = self.reid_pipeline.get_tracking_info()
                stats['reid_reassignments'] = self.reid_pipeline.get_reassignment_count()
                stats['reid_3d_improvements'] = self.reid_pipeline.get_3d_improvements_count()
                
                reid_3d_time = self.perf_monitor.end_timing('reid_3d')
                
                # LOG: ReID process
                similarity_scores = {}
                assignments = {}
                untracked_count = 0
                self.debug_logger.log_reid_process(horse_reid_features, similarity_scores, assignments, untracked_count, reid_3d_time, stats['reid_3d_improvements'])
                
                # LOG: Enhanced tracking
                self.debug_logger.log_samurai_tracking(tracking_info, [])
                
                # Update detections with enhanced tracking
                tracked_horses = tracked_horses_enhanced
                jockey_detections = jockey_detections_enhanced
            
            # Update statistics
            stats['humans_detected'] += len(jockey_detections) if sv else len(jockey_detections)
            stats['horses_detected'] += len(tracked_horses) if sv else len(tracked_horses)
            stats['tracked_horses'] = len(stats['active_horse_tracks'])
            stats['tracked_humans'] = len(stats['active_human_tracks'])
            
            # LOG: Pose estimation
            self.debug_logger.log_pose_estimation(human_poses_3d, horse_poses_3d, pose_3d_time)
            
            # STEP 6: Associate poses with track IDs
            human_poses_3d = self.associate_poses_with_tracks(human_poses_3d, jockey_detections)
            horse_poses_3d = self.associate_poses_with_tracks(horse_poses_3d, tracked_horses)
            
            # Update pose statistics
            stats['human_poses'] += len(human_poses_3d)
            stats['horse_poses'] += len(horse_poses_3d)
            stats['human_poses_3d'] += len([p for p in human_poses_3d if p and 'keypoints_3d' in p])
            stats['horse_poses_3d'] += len([p for p in horse_poses_3d if p and 'keypoints_3d' in p])
            
            # Calculate average 3D quality
            all_3d_poses = [p for p in human_poses_3d + horse_poses_3d if p and 'confidence_3d' in p]
            if all_3d_poses:
                stats['avg_3d_quality'] = np.mean([p['confidence_3d'] for p in all_3d_poses])
            
            # Count method wins for dual mode
            if self.config.horse_pose_estimator == 'dual':
                superanimal_count = sum(1 for pose in horse_poses_3d if pose and pose.get('method') == 'SuperAnimal')
                vitpose_count = sum(1 for pose in horse_poses_3d if pose and pose.get('method') == 'ViTPose')
                stats['superanimal_wins'] += superanimal_count
                stats['vitpose_wins'] += vitpose_count
            
            # STEP 7: Enhanced Visualization with timing
            self.perf_monitor.start_timing('visualization')
            
            frame = self.visualizer.annotate_detections_with_tracking(frame, jockey_detections, tracked_horses)
            
            # Draw 3D poses with track IDs
            for pose_result in human_poses_3d:
                if pose_result:
                    frame = self.visualizer.draw_human_pose_with_tracking(frame, pose_result)
            
            for pose_result in horse_poses_3d:
                if pose_result:
                    frame = self.visualizer.draw_horse_pose_with_tracking(frame, pose_result)
            
            # Add pose method labels
            frame = self.visualizer.draw_pose_labels(frame, [p for p in horse_poses_3d if p])
            
            # STEP 8: Add motion prediction visualizations
            frame = self.visualize_motion_predictions(frame, tracking_info)
            
            # STEP 9: Add enhanced 3D pose and performance information
            perf_summary = self.perf_monitor.get_performance_summary()
            frame = self.visualize_3d_pose_info(frame, horse_poses_3d + human_poses_3d, tracking_info, perf_summary)
            
            # Add enhanced depth and segmentation visualization
            if self.reid_pipeline and getattr(self.config, 'enable_reid_pipeline', False):
                # Get segmentation masks for visualization
                segmentation_masks = self.reid_pipeline.get_current_masks()
                
                if depth_map is not None:
                    # Create depth visualization with highlighted horse masks (top right)
                    depth_with_masks = self.visualize_depth_with_masks(depth_map, segmentation_masks, tracked_horses)
                    depth_display = cv2.resize(depth_with_masks, (300, 200))
                    frame[10:210, self.width-310:self.width-10] = depth_display
                    cv2.putText(frame, "GPU 3D Enhanced Depth", (self.width-300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    cv2.putText(frame, f"Batches: {stats['gpu_batches_processed']}", (self.width-300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                    
                    # Create 3D depth visualization (bottom left)
                    if all_3d_poses:
                        depth_3d_viz = self.visualizer.create_3d_pose_depth_visualization(all_3d_poses, frame.shape)
                        depth_3d_display = cv2.resize(depth_3d_viz, (300, 200))
                        frame[self.height-210:self.height-10, 10:310] = depth_3d_display
                        cv2.putText(frame, "3D Pose Depth Distribution", (20, self.height-220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            visualization_time = self.perf_monitor.end_timing('visualization')
            
            # Add comprehensive info overlay
            human_count = len(jockey_detections) if sv else len(jockey_detections)
            horse_count = len(tracked_horses) if sv else len(tracked_horses)
            
            frame = self.draw_comprehensive_info_overlay(
                frame, frame_count, max_frames, human_count, horse_count,
                len(human_poses_3d), len(horse_poses_3d), stats, tracking_info, perf_summary,
                self.expected_horses, self.expected_jockeys
            )
            
            # Write frame to output video
            out.write(frame)
            
            # Display frame if requested
            if self.config.display:
                cv2.imshow(window_name, frame)
                
                # Handle key presses
                key = cv2.waitKey(1 if not paused else 0) & 0xFF
                
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord(' '):  # SPACE
                    paused = not paused
                elif paused and key != 255:  # Any other key when paused
                    pass  # Continue to next frame
            
            frame_count += 1
            
            # LOG: Frame end and update performance monitoring
            frame_end_time = time.time()
            total_frame_time = self.perf_monitor.end_timing('total_frame')
            self.debug_logger.log_frame_end(total_frame_time)
            self.perf_monitor.log_frame_time(total_frame_time)
            self.perf_monitor.update_system_metrics()
            
            # Log performance metrics to debug logger
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            else:
                gpu_memory_mb = 0
            frame_rate = 1.0 / total_frame_time if total_frame_time > 0 else 0
            self.debug_logger.log_performance_metrics(gpu_memory_mb, frame_rate)
            
            # Update progress bar with comprehensive information
            if pbar:
                pose_3d_status = f"3D:{stats['horse_poses_3d']+stats['human_poses_3d']}"
                quality_status = f"Q:{stats['avg_3d_quality']:.2f}"
                perf_status = f"FPS:{frame_rate:.1f}"
                gpu_status = f"GPU:{gpu_memory_mb:.0f}MB"
                batch_status = f"Batch:{stats['gpu_batches_processed']}"
                parallel_status = f"||:{stats['parallel_operations']}" if executor else ""
                reid_3d_status = f"3D-ReID:{stats['reid_3d_improvements']}"
                
                pbar.set_postfix_str(f"H:{human_count}/{self.expected_jockeys} Ho:{horse_count}/{self.expected_horses} {pose_3d_status} {quality_status} {perf_status} {gpu_status} {batch_status} {parallel_status} {reid_3d_status}")
                pbar.update(1)
            
            # Periodic GPU memory cleanup
            if frame_count % 100 == 0 and getattr(self.config, 'gpu_memory_optimization', False):
                torch.cuda.empty_cache()
                gc.collect()
        
        # Cleanup
        if executor:
            executor.shutdown(wait=True)
        
        self.cap.release()
        out.release()
        
        if self.config.display:
            cv2.destroyAllWindows()
        
        if pbar:
            pbar.close()
        
        # Final performance summary
        final_perf_summary = self.perf_monitor.get_performance_summary()
        
        # SAVE ALL DEBUG LOGS with performance data
        log_files = self.debug_logger.save_logs(output_path)
        
        return output_path, final_perf_summary
    
    def draw_comprehensive_info_overlay(self, frame: np.ndarray, frame_count: int, max_frames: int, 
                                       human_count: int, horse_count: int, human_poses: int, 
                                       horse_poses: int, stats: dict, tracking_info: dict, perf_summary: dict,
                                       expected_horses: int = 10, expected_jockeys: int = 10):
        """Draw comprehensive info overlay with 3D pose, performance, and GPU metrics"""
        total_display = str(max_frames) if max_frames != float('inf') else "∞"
        
        info_lines = [
            f"Frame: {frame_count+1}/{total_display}",
            f"Expected: {expected_horses} horses, {expected_jockeys} jockeys",
            f"Config: H-Det:{self.config.human_detector} H-Pose:{self.config.human_pose_estimator}",
            f"        Horse-Det:{self.config.horse_detector} Horse-Pose:{self.config.horse_pose_estimator}",
            f"Detected - Jockeys:{human_count}/{expected_jockeys} Horses:{horse_count}/{expected_horses}",
            f"2D Poses - Humans:{human_poses} Horses:{horse_poses}",
        ]
        
        # Add 3D pose and performance statistics
        if stats:
            tracked_humans = stats.get('tracked_humans', 0)
            tracked_horses = stats.get('tracked_horses', 0)
            poses_3d_total = stats.get('human_poses_3d', 0) + stats.get('horse_poses_3d', 0)
            avg_3d_quality = stats.get('avg_3d_quality', 0)
            gpu_batches = stats.get('gpu_batches_processed', 0)
            parallel_ops = stats.get('parallel_operations', 0)
            
            info_lines.append(f"🔄 Unique Tracks - Humans:{tracked_humans} Horses:{tracked_horses}")
            info_lines.append(f"🎯 3D Poses: {poses_3d_total} (Avg Quality: {avg_3d_quality:.3f})")
            info_lines.append(f"🚀 GPU Processing: {gpu_batches} batches, {parallel_ops} parallel ops")
        
        # Add performance metrics
        if perf_summary:
            avg_fps = perf_summary.get('avg_fps', 0)
            resource_usage = perf_summary.get('resource_usage', {})
            gpu_memory = resource_usage.get('avg_gpu_memory_mb', 0)
            cpu_usage = resource_usage.get('avg_cpu_percent', 0)
            
            info_lines.append(f"⚡ Performance: {avg_fps:.1f} FPS, GPU:{gpu_memory:.0f}MB, CPU:{cpu_usage:.0f}%")
            
            # Component timing breakdown
            component_times = perf_summary.get('component_times', {})
            if component_times:
                bottleneck = max(component_times.items(), key=lambda x: x[1].get('percentage', 0))
                info_lines.append(f"   Bottleneck: {bottleneck[0]} ({bottleneck[1].get('percentage', 0):.1f}%)")
        
        # Add Enhanced ReID info with 3D integration
        if getattr(self.config, 'enable_reid_pipeline', False) and tracking_info:
            sam_model_display = getattr(self.config, 'sam_model', 'none').upper()
            info_lines.append(f"🎯 GPU-Accelerated 3D ReID: {sam_model_display} + Batch Processing")
            
            active_tracks = tracking_info.get('active_tracks', 0)
            reid_reassignments = stats.get('reid_reassignments', 0) if stats else 0
            reid_3d_improvements = stats.get('reid_3d_improvements', 0) if stats else 0
            
            info_lines.append(f"   Active:{active_tracks} Reassign:{reid_reassignments} 3D-Improve:{reid_3d_improvements}")
            
            # Feature weights
            visual_weight = getattr(self.config, 'visual_feature_weight', 0.35)
            pose_3d_weight = getattr(self.config, 'pose_3d_feature_weight', 0.30)
            geometric_weight = getattr(self.config, 'geometric_feature_weight', 0.20)
            
            info_lines.append(f"   Weights: Visual({visual_weight:.2f}) 3D-Pose({pose_3d_weight:.2f}) Geometric({geometric_weight:.2f})")
        else:
            info_lines.append(f"🎯 GPU-Accelerated 3D ReID: DISABLED")
        
        info_lines.append(f"🎨 3D Integration: GPU Depth + Batch 3D Conversion + Parallel Processing")
        info_lines.append(f"📊 Advanced Logging: Performance + 3D Metrics + GPU Monitoring")
        
        if self.config.horse_pose_estimator == 'dual' and stats:
            info_lines.append(f"Competition - SuperAnimal:{stats.get('superanimal_wins', 0)} ViTPose:{stats.get('vitpose_wins', 0)}")
        
        if self.config.display:
            info_lines.append(f"Controls: SPACE=Pause Q=Quit | GPU-accelerated 3D visualizations")
        
        # Semi-transparent background
        overlay = frame.copy()
        overlay_height = 25 + len(info_lines) * 18
        cv2.rectangle(overlay, (5, 5), (1100, overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        for i, line in enumerate(info_lines):
            y_pos = 25 + i * 18
            cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def process_video(self):
        """Main video processing method"""
        try:
            output_path, perf_summary = self.process_video_with_parallel_execution()
            
            # Print comprehensive final report
            print(f"\n✅ GPU-Accelerated 3D Processing Complete!")
            print(f"🎯 Output: {output_path}")
            
            print(f"\n📊 Final Performance Report:")
            print(f"   Average FPS: {perf_summary['avg_fps']:.2f}")
            print(f"   Total Frame Time: {perf_summary['avg_frame_time']*1000:.1f}ms avg")
            
            # Component timing breakdown
            component_times = perf_summary.get('component_times', {})
            if component_times:
                print(f"\n⏱️ Component Timing Breakdown:")
                for component, timing in sorted(component_times.items(), key=lambda x: x[1]['percentage'], reverse=True):
                    print(f"   {component}: {timing['avg_ms']:.1f}ms avg ({timing['percentage']:.1f}%)")
            
            # Resource usage
            resource_usage = perf_summary.get('resource_usage', {})
            print(f"\n💾 Resource Usage:")
            print(f"   Average GPU Memory: {resource_usage.get('avg_gpu_memory_mb', 0):.0f} MB")
            print(f"   Peak GPU Memory: {resource_usage.get('peak_gpu_memory_mb', 0):.0f} MB")
            print(f"   Average CPU Usage: {resource_usage.get('avg_cpu_percent', 0):.1f}%")
            print(f"   Average RAM Usage: {resource_usage.get('avg_memory_mb', 0):.0f} MB")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error during processing: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python main.py config.yaml")
        sys.exit(1)
    
    # Load config from file
    config_file = sys.argv[1]
    config = Config(config_file)
    
    # Get video path from config
    video_path = getattr(config, 'video_path', None)
    if not video_path:
        print("❌ Error: Config file must specify 'video_path'")
        sys.exit(1)
    
    # Auto-detect device
    if config.device == "cpu" and torch.cuda.is_available():
        config.device = "cuda"
        print(f"🚀 GPU detected, switching to CUDA acceleration")
    
    # Check if video file exists
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"❌ Error: Video file '{video_path}' does not exist")
        sys.exit(1)
    
    try:
        print(f"🎬 Initializing GPU-Accelerated 3D Horse Tracking System...")
        system = HybridPoseSystem(video_path, config)
        output_path = system.process_video()
        
        if output_path:
            print(f"\n🎊 Processing completed successfully!")
            print(f"📁 Output saved to: {output_path}")
        else:
            print(f"\n❌ Processing failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ System Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
    