import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

@dataclass
class Detection:
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    pose_keypoints: Optional[np.ndarray] = None
    depth_map: Optional[np.ndarray] = None
    sam_mask: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None

@dataclass
class JockeyHorsePair:
    jockey_detection: Detection
    horse_detection: Detection
    pair_confidence: float
    spatial_score: float
    depth_score: float
    pose_score: float
    motion_score: float
    combined_bbox: List[float]  # Combined bounding box
    pair_id: int

class JockeyHorsePairing:
    def __init__(self):
        # Spatial relationship parameters optimized for racing
        self.max_vertical_distance = 100  # pixels
        self.max_horizontal_distance = 50  # pixels
        self.depth_tolerance = 0.15  # 15% depth difference
        self.size_ratio_min = 0.2  # jockey should be 20-60% of horse size
        self.size_ratio_max = 0.6
        
        # Motion correlation parameters
        self.motion_history_frames = 5
        self.motion_correlation_threshold = 0.7
        
        # Pose alignment parameters
        self.horse_spine_keypoint_indices = [0, 1, 2, 3, 4]  # withers, back, loin, croup, tail
        self.jockey_hip_keypoint_indices = [11, 12]  # left/right hip in COCO format
        
        # Previous frame data for motion analysis
        self.previous_detections = {}
        self.motion_history = {}
        
    def calculate_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Calculate center point of bounding box"""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def calculate_bbox_area(self, bbox: List[float]) -> float:
        """Calculate area of bounding box"""
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    def get_spatial_relationship_score(self, jockey: Detection, horse: Detection) -> float:
        """Calculate spatial relationship score between jockey and horse"""
        jockey_center = self.calculate_bbox_center(jockey.bbox)
        horse_center = self.calculate_bbox_center(horse.bbox)
        
        # Vertical alignment: jockey should be above horse center
        vertical_distance = abs(jockey_center[1] - horse.bbox[1])  # Distance from jockey to horse top
        vertical_score = max(0, 1 - (vertical_distance / self.max_vertical_distance))
        
        # Horizontal alignment: jockey should be centered on horse
        horizontal_distance = abs(jockey_center[0] - horse_center[0])
        horizontal_score = max(0, 1 - (horizontal_distance / self.max_horizontal_distance))
        
        # Containment: jockey should be within horse's horizontal bounds
        jockey_in_horse_bounds = (jockey.bbox[0] >= horse.bbox[0] - 20 and 
                                 jockey.bbox[2] <= horse.bbox[2] + 20)
        containment_score = 1.0 if jockey_in_horse_bounds else 0.3
        
        # Size relationship: jockey should be smaller than horse
        jockey_area = self.calculate_bbox_area(jockey.bbox)
        horse_area = self.calculate_bbox_area(horse.bbox)
        size_ratio = jockey_area / horse_area if horse_area > 0 else 0
        size_score = 1.0 if self.size_ratio_min <= size_ratio <= self.size_ratio_max else 0.5
        
        # Combined spatial score
        spatial_score = (vertical_score * 0.4 + horizontal_score * 0.3 + 
                        containment_score * 0.2 + size_score * 0.1)
        
        return spatial_score
    
    def get_depth_consistency_score(self, jockey: Detection, horse: Detection) -> float:
        """Calculate depth consistency score between jockey and horse"""
        if jockey.depth_map is None or horse.depth_map is None:
            return 0.5  # neutral score if depth not available
        
        # Extract depth values from detection regions
        jockey_bbox_int = [int(x) for x in jockey.bbox]
        horse_bbox_int = [int(x) for x in horse.bbox]
        
        jockey_depth_region = jockey.depth_map[jockey_bbox_int[1]:jockey_bbox_int[3], 
                                             jockey_bbox_int[0]:jockey_bbox_int[2]]
        horse_depth_region = horse.depth_map[horse_bbox_int[1]:horse_bbox_int[3], 
                                           horse_bbox_int[0]:horse_bbox_int[2]]
        
        if jockey_depth_region.size == 0 or horse_depth_region.size == 0:
            return 0.5
        
        # Calculate median depth for each detection
        jockey_median_depth = np.median(jockey_depth_region)
        horse_median_depth = np.median(horse_depth_region)
        
        # Depth difference should be minimal (they're together)
        depth_difference = abs(jockey_median_depth - horse_median_depth) / max(jockey_median_depth, horse_median_depth)
        depth_score = max(0, 1 - (depth_difference / self.depth_tolerance))
        
        return depth_score
    
    def get_pose_alignment_score(self, jockey: Detection, horse: Detection) -> float:
        """Calculate pose alignment score using keypoints"""
        if jockey.pose_keypoints is None or horse.pose_keypoints is None:
            return 0.5  # neutral score if pose not available
        
        # Extract relevant keypoints
        horse_spine_points = []
        for idx in self.horse_spine_keypoint_indices:
            if idx < len(horse.pose_keypoints):
                keypoint = horse.pose_keypoints[idx]
                if keypoint[2] > 0.3:  # confidence threshold
                    horse_spine_points.append(keypoint[:2])
        
        jockey_hip_points = []
        for idx in self.jockey_hip_keypoint_indices:
            if idx < len(jockey.pose_keypoints):
                keypoint = jockey.pose_keypoints[idx]
                if keypoint[2] > 0.3:  # confidence threshold
                    jockey_hip_points.append(keypoint[:2])
        
        if len(horse_spine_points) < 2 or len(jockey_hip_points) < 1:
            return 0.5
        
        # Calculate alignment between jockey hips and horse spine
        horse_spine_center = np.mean(horse_spine_points, axis=0)
        jockey_hip_center = np.mean(jockey_hip_points, axis=0)
        
        # Distance between jockey hips and horse spine center
        alignment_distance = np.linalg.norm(jockey_hip_center - horse_spine_center)
        pose_score = max(0, 1 - (alignment_distance / 100))  # 100 pixel max distance
        
        return pose_score
    
    def get_motion_correlation_score(self, jockey: Detection, horse: Detection, frame_id: int) -> float:
        """Calculate motion correlation score based on movement history"""
        jockey_key = f"jockey_{id(jockey)}"
        horse_key = f"horse_{id(horse)}"
        
        # Initialize motion history if first frame
        if frame_id == 0:
            return 0.5  # neutral score for first frame
        
        # Get current centers
        jockey_center = self.calculate_bbox_center(jockey.bbox)
        horse_center = self.calculate_bbox_center(horse.bbox)
        
        # Store current positions
        if jockey_key not in self.motion_history:
            self.motion_history[jockey_key] = []
        if horse_key not in self.motion_history:
            self.motion_history[horse_key] = []
        
        self.motion_history[jockey_key].append(jockey_center)
        self.motion_history[horse_key].append(horse_center)
        
        # Keep only recent history
        if len(self.motion_history[jockey_key]) > self.motion_history_frames:
            self.motion_history[jockey_key] = self.motion_history[jockey_key][-self.motion_history_frames:]
        if len(self.motion_history[horse_key]) > self.motion_history_frames:
            self.motion_history[horse_key] = self.motion_history[horse_key][-self.motion_history_frames:]
        
        # Calculate motion vectors if we have enough history
        if len(self.motion_history[jockey_key]) < 3 or len(self.motion_history[horse_key]) < 3:
            return 0.5
        
        # Calculate motion vectors for last few frames
        jockey_motion = []
        horse_motion = []
        
        for i in range(1, min(len(self.motion_history[jockey_key]), len(self.motion_history[horse_key]))):
            jockey_motion.append([
                self.motion_history[jockey_key][i][0] - self.motion_history[jockey_key][i-1][0],
                self.motion_history[jockey_key][i][1] - self.motion_history[jockey_key][i-1][1]
            ])
            horse_motion.append([
                self.motion_history[horse_key][i][0] - self.motion_history[horse_key][i-1][0],
                self.motion_history[horse_key][i][1] - self.motion_history[horse_key][i-1][1]
            ])
        
        if len(jockey_motion) == 0 or len(horse_motion) == 0:
            return 0.5
        
        # Calculate correlation between motion vectors
        jockey_motion = np.array(jockey_motion)
        horse_motion = np.array(horse_motion)
        
        # Flatten motion vectors for correlation calculation
        jockey_flat = jockey_motion.flatten()
        horse_flat = horse_motion.flatten()
        
        if len(jockey_flat) != len(horse_flat) or np.std(jockey_flat) == 0 or np.std(horse_flat) == 0:
            return 0.5
        
        correlation = np.corrcoef(jockey_flat, horse_flat)[0, 1]
        motion_score = max(0, correlation) if not np.isnan(correlation) else 0.5
        
        return motion_score
    
    def get_sam_overlap_score(self, jockey: Detection, horse: Detection) -> float:
        """Calculate SAM mask overlap score"""
        if jockey.sam_mask is None or horse.sam_mask is None:
            return 0.5
        
        # Ensure masks are same size
        if jockey.sam_mask.shape != horse.sam_mask.shape:
            return 0.5
        
        # Calculate intersection and union
        intersection = np.logical_and(jockey.sam_mask, horse.sam_mask)
        jockey_area = np.sum(jockey.sam_mask)
        horse_area = np.sum(horse.sam_mask)
        intersection_area = np.sum(intersection)
        
        if jockey_area == 0:
            return 0.0
        
        # Jockey should overlap with horse (they're together)
        overlap_ratio = intersection_area / jockey_area
        overlap_score = min(1.0, overlap_ratio * 2)  # Scale so 50% overlap = 1.0 score
        
        return overlap_score
    
    def create_combined_bbox(self, jockey: Detection, horse: Detection) -> List[float]:
        """Create combined bounding box for jockey-horse pair"""
        combined_x1 = min(jockey.bbox[0], horse.bbox[0])
        combined_y1 = min(jockey.bbox[1], horse.bbox[1])
        combined_x2 = max(jockey.bbox[2], horse.bbox[2])
        combined_y2 = max(jockey.bbox[3], horse.bbox[3])
        
        return [combined_x1, combined_y1, combined_x2, combined_y2]
    
    def calculate_pair_confidence(self, spatial_score: float, depth_score: float, 
                                pose_score: float, motion_score: float, sam_score: float) -> float:
        """Calculate overall pair confidence with weighted scores"""
        weights = {
            'spatial': 0.35,    # Most important - spatial relationship
            'depth': 0.20,      # Important - depth consistency
            'pose': 0.20,       # Important - pose alignment
            'motion': 0.15,     # Motion correlation
            'sam': 0.10         # SAM overlap
        }
        
        confidence = (spatial_score * weights['spatial'] +
                     depth_score * weights['depth'] +
                     pose_score * weights['pose'] +
                     motion_score * weights['motion'] +
                     sam_score * weights['sam'])
        
        return confidence
    
    def pair_jockeys_with_horses(self, jockey_detections: List[Detection], 
                               horse_detections: List[Detection], 
                               frame_id: int) -> List[JockeyHorsePair]:
        """Main function to pair jockeys with horses using Hungarian algorithm"""
        if not jockey_detections or not horse_detections:
            return []
        
        num_jockeys = len(jockey_detections)
        num_horses = len(horse_detections)
        
        # Create cost matrix for Hungarian algorithm
        cost_matrix = np.zeros((num_jockeys, num_horses))
        score_details = {}
        
        for i, jockey in enumerate(jockey_detections):
            for j, horse in enumerate(horse_detections):
                # Calculate all component scores
                spatial_score = self.get_spatial_relationship_score(jockey, horse)
                depth_score = self.get_depth_consistency_score(jockey, horse)
                pose_score = self.get_pose_alignment_score(jockey, horse)
                motion_score = self.get_motion_correlation_score(jockey, horse, frame_id)
                sam_score = self.get_sam_overlap_score(jockey, horse)
                
                # Calculate overall confidence
                pair_confidence = self.calculate_pair_confidence(
                    spatial_score, depth_score, pose_score, motion_score, sam_score
                )
                
                # Store scores for later use
                score_details[(i, j)] = {
                    'spatial': spatial_score,
                    'depth': depth_score,
                    'pose': pose_score,
                    'motion': motion_score,
                    'sam': sam_score,
                    'confidence': pair_confidence
                }
                
                # Hungarian algorithm minimizes cost, so use negative confidence
                cost_matrix[i, j] = -pair_confidence
        
        # Solve assignment problem using Hungarian algorithm
        jockey_indices, horse_indices = linear_sum_assignment(cost_matrix)
        
        # Create jockey-horse pairs
        pairs = []
        pair_id = 0
        
        for jockey_idx, horse_idx in zip(jockey_indices, horse_indices):
            scores = score_details[(jockey_idx, horse_idx)]
            
            # Only create pair if confidence is above threshold
            if scores['confidence'] > 0.4:  # Minimum confidence threshold
                combined_bbox = self.create_combined_bbox(
                    jockey_detections[jockey_idx], 
                    horse_detections[horse_idx]
                )
                
                pair = JockeyHorsePair(
                    jockey_detection=jockey_detections[jockey_idx],
                    horse_detection=horse_detections[horse_idx],
                    pair_confidence=scores['confidence'],
                    spatial_score=scores['spatial'],
                    depth_score=scores['depth'],
                    pose_score=scores['pose'],
                    motion_score=scores['motion'],
                    combined_bbox=combined_bbox,
                    pair_id=pair_id
                )
                
                pairs.append(pair)
                pair_id += 1
        
        return pairs
    
    def extract_pair_features(self, pair: JockeyHorsePair, frame: np.ndarray) -> np.ndarray:
        """Extract combined features for the jockey-horse pair for ReID"""
        # Extract region from combined bbox
        bbox = [int(x) for x in pair.combined_bbox]
        pair_region = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        
        if pair_region.size == 0:
            return np.zeros(512)  # Return zero vector if extraction fails
        
        # Resize to standard size for feature extraction
        pair_region_resized = cv2.resize(pair_region, (128, 256))
        
        # Extract color features (HSV histogram)
        hsv = cv2.cvtColor(pair_region_resized, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        color_features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
        
        # Extract texture features (LBP-like)
        gray = cv2.cvtColor(pair_region_resized, cv2.COLOR_BGR2GRAY)
        
        # Simple texture features using gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Spatial pyramid features
        h, w = magnitude.shape
        spatial_features = []
        
        # Divide into 2x2 grid and extract features from each region
        for i in range(2):
            for j in range(2):
                region = magnitude[i*h//2:(i+1)*h//2, j*w//2:(j+1)*w//2]
                if region.size > 0:
                    spatial_features.extend([
                        np.mean(region),
                        np.std(region),
                        np.max(region),
                        np.min(region)
                    ])
                else:
                    spatial_features.extend([0, 0, 0, 0])
        
        # Combine jockey-specific features (top part of combined region)
        jockey_region = pair_region_resized[:128, :]  # Top half
        jockey_hsv = cv2.cvtColor(jockey_region, cv2.COLOR_BGR2HSV)
        jockey_color = cv2.calcHist([jockey_hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        
        # Combine horse-specific features (bottom part)
        horse_region = pair_region_resized[128:, :]  # Bottom half
        horse_hsv = cv2.cvtColor(horse_region, cv2.COLOR_BGR2HSV)
        horse_color = cv2.calcHist([horse_hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        
        # Combine all features
        all_features = np.concatenate([
            color_features / (np.sum(color_features) + 1e-8),  # Normalized color
            np.array(spatial_features),                         # Spatial texture
            jockey_color.flatten() / (np.sum(jockey_color) + 1e-8),  # Jockey color
            horse_color.flatten() / (np.sum(horse_color) + 1e-8)     # Horse color
        ])
        
        # Ensure fixed size output
        if len(all_features) > 512:
            all_features = all_features[:512]
        elif len(all_features) < 512:
            all_features = np.pad(all_features, (0, 512 - len(all_features)), 'constant')
        
        return all_features

# Integration function for main tracking pipeline
def integrate_jockey_horse_pairing(detections_dict: Dict, frame: np.ndarray, frame_id: int, 
                                 depth_map: np.ndarray = None) -> List[JockeyHorsePair]:
    """
    Integration function to be called from main tracking pipeline
    
    Args:
        detections_dict: Dictionary with 'human' and 'horse' detection lists
        frame: Current frame image
        frame_id: Current frame number
        depth_map: Optional depth map from depth estimation
    
    Returns:
        List of jockey-horse pairs for tracking
    """
    pairing_system = JockeyHorsePairing()
    
    # Extract detections
    jockey_detections = detections_dict.get('human', [])
    horse_detections = detections_dict.get('horse', [])
    
    # Add depth map to detections if available
    if depth_map is not None:
        for detection in jockey_detections + horse_detections:
            detection.depth_map = depth_map
    
    # Create pairs
    pairs = pairing_system.pair_jockeys_with_horses(
        jockey_detections, horse_detections, frame_id
    )
    
    # Extract features for each pair
    for pair in pairs:
        pair_features = pairing_system.extract_pair_features(pair, frame)
        # Store features in both detections for compatibility
        pair.jockey_detection.features = pair_features
        pair.horse_detection.features = pair_features
    
    return pairs