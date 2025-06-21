import numpy as np
import torch
import cv2
import math
from typing import List, Dict, Tuple, Optional
from collections import deque
import concurrent.futures
from functools import lru_cache

try:
    import supervision as sv
except ImportError:
    sv = None

try:
    from transformers import AutoProcessor, VitPoseForPoseEstimation
    VITPOSE_AVAILABLE = True
except ImportError:
    VITPOSE_AVAILABLE = False

try:
    import scipy.ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class AdvancedDepthProcessor:
    """GPU-accelerated depth processing with advanced filtering and smoothing"""
    
    def __init__(self, config, device="cuda"):
        self.config = config
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Temporal smoothing buffers
        self.depth_history = deque(maxlen=getattr(config, 'depth_temporal_window', 5))
        self.depth_weights = self._create_temporal_weights()
        
        # Edge detection kernel for preservation
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                   dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                   dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        
        # Gaussian kernels for different smoothing levels
        self.gaussian_kernels = self._create_gaussian_kernels()
        
        # Kalman filter parameters for temporal consistency
        if getattr(config, 'enable_3d_kalman_filtering', False):
            self.kalman_filters = {}
            self.process_noise = getattr(config, 'kalman_process_noise', 0.1)
            self.measurement_noise = getattr(config, 'kalman_measurement_noise', 0.5)
    
    def _create_temporal_weights(self) -> torch.Tensor:
        """Create weights for temporal smoothing (more recent frames have higher weight)"""
        window_size = getattr(self.config, 'depth_temporal_window', 5)
        weights = torch.exp(torch.linspace(-2, 0, window_size))
        weights = weights / weights.sum()
        return weights.to(self.device)
    
    def _create_gaussian_kernels(self) -> Dict[str, torch.Tensor]:
        """Create Gaussian kernels for different smoothing levels"""
        kernels = {}
        for sigma in [0.5, 1.0, 1.5, 2.0]:
            size = int(6 * sigma + 1)
            if size % 2 == 0:
                size += 1
            x = torch.arange(size, dtype=torch.float32, device=self.device) - size // 2
            y = x.unsqueeze(0)
            x = x.unsqueeze(1)
            kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
            kernel = kernel / kernel.sum()
            kernels[f'sigma_{sigma}'] = kernel.unsqueeze(0).unsqueeze(0)
        return kernels
    
    @torch.no_grad()
    def process_depth_map_gpu(self, depth_map: np.ndarray) -> Tuple[torch.Tensor, Dict]:
        """GPU-accelerated depth map processing with advanced filtering"""
        # Convert to GPU tensor
        if isinstance(depth_map, np.ndarray):
            depth_tensor = torch.from_numpy(depth_map).float().to(self.device)
        else:
            depth_tensor = depth_map.float().to(self.device)
        
        # Add batch and channel dimensions
        if depth_tensor.dim() == 2:
            depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
        elif depth_tensor.dim() == 3:
            depth_tensor = depth_tensor.unsqueeze(0)
        
        original_depth = depth_tensor.clone()
        
        # Step 1: Outlier removal
        if getattr(self.config, 'depth_outlier_threshold', 0) > 0:
            depth_tensor = self._remove_outliers_gpu(depth_tensor)
        
        # Step 2: Hole filling
        if getattr(self.config, 'enable_depth_hole_filling', False):
            depth_tensor = self._fill_holes_gpu(depth_tensor)
        
        # Step 3: Edge-preserving smoothing
        smoothing_algorithm = getattr(self.config, 'depth_smoothing_algorithm', 'gaussian')
        if smoothing_algorithm != 'none':
            depth_tensor = self._apply_smoothing_gpu(depth_tensor, smoothing_algorithm)
        
        # Step 4: Temporal smoothing
        if getattr(self.config, 'depth_temporal_smoothing', False):
            depth_tensor = self._apply_temporal_smoothing_gpu(depth_tensor)
        
        # Step 5: Noise reduction
        noise_strength = getattr(self.config, 'depth_noise_reduction_strength', 0)
        if noise_strength > 0:
            depth_tensor = self._reduce_noise_gpu(depth_tensor, noise_strength)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_depth_quality_gpu(original_depth, depth_tensor)
        
        return depth_tensor.squeeze(), quality_metrics
    
    @torch.no_grad()
    def _remove_outliers_gpu(self, depth_tensor: torch.Tensor) -> torch.Tensor:
        """Remove depth outliers using GPU-accelerated statistics"""
        threshold = getattr(self.config, 'depth_outlier_threshold', 2.5)
        
        # Calculate statistics on GPU
        valid_mask = depth_tensor > 0
        valid_depths = depth_tensor[valid_mask]
        
        if len(valid_depths) == 0:
            return depth_tensor
        
        mean_depth = valid_depths.mean()
        std_depth = valid_depths.std()
        
        # Mark outliers
        outlier_mask = torch.abs(depth_tensor - mean_depth) > (threshold * std_depth)
        outlier_mask = outlier_mask & valid_mask
        
        # Replace outliers with local median
        if outlier_mask.any():
            # Use morphological operations for local median approximation
            kernel_size = 5
            padding = kernel_size // 2
            padded = torch.nn.functional.pad(depth_tensor, (padding, padding, padding, padding), mode='reflect')
            
            # Create unfold operation for local neighborhoods
            unfolded = torch.nn.functional.unfold(padded, kernel_size, stride=1)
            unfolded = unfolded.view(1, kernel_size*kernel_size, depth_tensor.shape[2], depth_tensor.shape[3])
            
            # Approximate median with sorting
            sorted_vals, _ = torch.sort(unfolded, dim=1)
            local_median = sorted_vals[:, kernel_size*kernel_size//2]
            
            depth_tensor = torch.where(outlier_mask, local_median, depth_tensor)
        
        return depth_tensor
    
    @torch.no_grad()
    def _fill_holes_gpu(self, depth_tensor: torch.Tensor) -> torch.Tensor:
        """Fill holes in depth map using GPU operations"""
        method = getattr(self.config, 'depth_hole_filling_method', 'inpaint')
        
        # Identify holes (zero or very small values)
        hole_mask = depth_tensor < 0.01
        
        if not hole_mask.any():
            return depth_tensor
        
        if method == 'nearest':
            # Distance transform approach
            filled = self._nearest_neighbor_fill_gpu(depth_tensor, hole_mask)
        elif method == 'interpolation':
            # Bilinear interpolation
            filled = self._interpolation_fill_gpu(depth_tensor, hole_mask)
        else:  # inpaint
            # More sophisticated inpainting
            filled = self._inpaint_fill_gpu(depth_tensor, hole_mask)
        
        return filled
    
    @torch.no_grad()
    def _nearest_neighbor_fill_gpu(self, depth_tensor: torch.Tensor, hole_mask: torch.Tensor) -> torch.Tensor:
        """Simple nearest neighbor hole filling on GPU"""
        # Use max pooling to propagate nearest valid values
        kernel_sizes = [3, 5, 7, 9]
        filled = depth_tensor.clone()
        
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            
            # Dilate valid regions
            valid_mask = (filled > 0).float()
            dilated_valid = torch.nn.functional.max_pool2d(valid_mask, kernel_size, stride=1, padding=padding)
            
            # Propagate values
            dilated_depth = torch.nn.functional.max_pool2d(filled, kernel_size, stride=1, padding=padding)
            
            # Update only previously invalid regions
            update_mask = (filled == 0) & (dilated_valid > 0)
            filled = torch.where(update_mask, dilated_depth, filled)
            
            if not (filled == 0).any():
                break
        
        return filled
    
    @torch.no_grad()
    def _interpolation_fill_gpu(self, depth_tensor: torch.Tensor, hole_mask: torch.Tensor) -> torch.Tensor:
        """Bilinear interpolation hole filling"""
        # Create coordinate grids
        h, w = depth_tensor.shape[-2:]
        y_coords = torch.arange(h, dtype=torch.float32, device=self.device).view(-1, 1).repeat(1, w)
        x_coords = torch.arange(w, dtype=torch.float32, device=self.device).view(1, -1).repeat(h, 1)
        
        # Get valid coordinates and values
        valid_mask = ~hole_mask.squeeze()
        valid_depths = depth_tensor.squeeze()[valid_mask]
        valid_y = y_coords[valid_mask]
        valid_x = x_coords[valid_mask]
        
        # For each hole pixel, find nearby valid pixels and interpolate
        hole_coords = torch.nonzero(hole_mask.squeeze())
        filled = depth_tensor.clone()
        
        if len(valid_depths) > 0 and len(hole_coords) > 0:
            # Batch process holes for efficiency
            for hole_y, hole_x in hole_coords:
                # Find nearest valid pixels (simplified approach)
                distances = torch.sqrt((valid_y - hole_y)**2 + (valid_x - hole_x)**2)
                
                # Use 4 nearest neighbors
                _, nearest_indices = torch.topk(distances, min(4, len(distances)), largest=False)
                nearest_depths = valid_depths[nearest_indices]
                nearest_distances = distances[nearest_indices]
                
                # Inverse distance weighting
                weights = 1.0 / (nearest_distances + 1e-6)
                weights = weights / weights.sum()
                
                interpolated_depth = torch.sum(nearest_depths * weights)
                filled[0, 0, hole_y, hole_x] = interpolated_depth
        
        return filled
    
    @torch.no_grad()
    def _inpaint_fill_gpu(self, depth_tensor: torch.Tensor, hole_mask: torch.Tensor) -> torch.Tensor:
        """Advanced inpainting-based hole filling"""
        # Use iterative inpainting with edge-aware filtering
        filled = depth_tensor.clone()
        hole_mask_current = hole_mask.clone()
        
        max_iterations = 10
        for iteration in range(max_iterations):
            if not hole_mask_current.any():
                break
            
            # Apply edge-preserving filter to propagate values
            smoothed = self._apply_edge_preserving_filter_gpu(filled)
            
            # Update holes with smoothed values
            filled = torch.where(hole_mask_current, smoothed, filled)
            
            # Reduce hole mask (fill from edges inward)
            hole_mask_current = self._erode_mask_gpu(hole_mask_current)
        
        return filled
    
    @torch.no_grad()
    def _apply_smoothing_gpu(self, depth_tensor: torch.Tensor, algorithm: str) -> torch.Tensor:
        """Apply various smoothing algorithms on GPU"""
        if algorithm == 'gaussian':
            return self._gaussian_smooth_gpu(depth_tensor)
        elif algorithm == 'bilateral':
            return self._bilateral_filter_gpu(depth_tensor)
        elif algorithm == 'adaptive_gaussian':
            return self._adaptive_gaussian_smooth_gpu(depth_tensor)
        else:
            return depth_tensor
    
    @torch.no_grad()
    def _gaussian_smooth_gpu(self, depth_tensor: torch.Tensor) -> torch.Tensor:
        """Standard Gaussian smoothing"""
        smoothing_factor = getattr(self.config, 'depth_smoothing_factor', 0.3)
        sigma_key = f'sigma_{1.0 + smoothing_factor}'
        
        # Use closest available kernel
        available_sigmas = list(self.gaussian_kernels.keys())
        if sigma_key not in available_sigmas:
            sigma_key = available_sigmas[min(range(len(available_sigmas)), 
                                          key=lambda i: abs(float(available_sigmas[i].split('_')[1]) - (1.0 + smoothing_factor)))]
        
        kernel = self.gaussian_kernels[sigma_key]
        return torch.nn.functional.conv2d(depth_tensor, kernel, padding='same')
    
    @torch.no_grad()
    def _bilateral_filter_gpu(self, depth_tensor: torch.Tensor) -> torch.Tensor:
        """Simplified bilateral filtering on GPU"""
        # Implement a simplified version using separable filters
        smoothing_factor = getattr(self.config, 'depth_smoothing_factor', 0.3)
        
        # Apply Gaussian spatial filter
        spatial_filtered = self._gaussian_smooth_gpu(depth_tensor)
        
        # Range filtering based on local variance
        local_var = self._calculate_local_variance_gpu(depth_tensor)
        edge_strength = torch.exp(-local_var / (2 * (smoothing_factor + 0.1)**2))
        
        # Blend based on edge strength
        return edge_strength * depth_tensor + (1 - edge_strength) * spatial_filtered
    
    @torch.no_grad()
    def _adaptive_gaussian_smooth_gpu(self, depth_tensor: torch.Tensor) -> torch.Tensor:
        """Adaptive Gaussian smoothing based on local edge strength"""
        # Calculate edge magnitude
        edge_mag = self._calculate_edge_magnitude_gpu(depth_tensor)
        
        # Normalize edge magnitude
        edge_mag_norm = edge_mag / (edge_mag.max() + 1e-6)
        
        # Apply different levels of smoothing based on edge strength
        smoothing_factor = getattr(self.config, 'depth_smoothing_factor', 0.3)
        
        # Strong smoothing in flat regions
        strong_smooth = torch.nn.functional.conv2d(depth_tensor, self.gaussian_kernels['sigma_1.5'], padding='same')
        
        # Light smoothing in edge regions
        light_smooth = torch.nn.functional.conv2d(depth_tensor, self.gaussian_kernels['sigma_0.5'], padding='same')
        
        # Adaptive blending
        alpha = torch.sigmoid(5 * (edge_mag_norm - 0.3))  # Sigmoid transition
        return alpha * light_smooth + (1 - alpha) * strong_smooth
    
    @torch.no_grad()
    def _calculate_edge_magnitude_gpu(self, depth_tensor: torch.Tensor) -> torch.Tensor:
        """Calculate edge magnitude using Sobel operators"""
        grad_x = torch.nn.functional.conv2d(depth_tensor, self.sobel_x, padding='same')
        grad_y = torch.nn.functional.conv2d(depth_tensor, self.sobel_y, padding='same')
        return torch.sqrt(grad_x**2 + grad_y**2)
    
    @torch.no_grad()
    def _calculate_local_variance_gpu(self, depth_tensor: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """Calculate local variance using GPU operations"""
        # Mean filter
        ones_kernel = torch.ones(1, 1, kernel_size, kernel_size, device=self.device) / (kernel_size**2)
        local_mean = torch.nn.functional.conv2d(depth_tensor, ones_kernel, padding='same')
        
        # Mean of squares
        local_mean_sq = torch.nn.functional.conv2d(depth_tensor**2, ones_kernel, padding='same')
        
        # Variance = E[X²] - E[X]²
        return local_mean_sq - local_mean**2
    
    @torch.no_grad()
    def _apply_temporal_smoothing_gpu(self, depth_tensor: torch.Tensor) -> torch.Tensor:
        """Apply temporal smoothing across frames"""
        self.depth_history.append(depth_tensor.clone())
        
        if len(self.depth_history) < 2:
            return depth_tensor
        
        # Weighted average of recent frames
        stacked = torch.stack(list(self.depth_history), dim=0)
        weights = self.depth_weights[-len(self.depth_history):].view(-1, 1, 1, 1, 1)
        
        return torch.sum(stacked * weights, dim=0)
    
    @torch.no_grad()
    def _reduce_noise_gpu(self, depth_tensor: torch.Tensor, strength: float) -> torch.Tensor:
        """Reduce noise while preserving edges"""
        # Non-local means approximation
        kernel_size = 3
        search_window = 7
        
        # Create patches
        patches = torch.nn.functional.unfold(depth_tensor, kernel_size, padding=kernel_size//2)
        patches = patches.view(1, kernel_size**2, depth_tensor.shape[2], depth_tensor.shape[3])
        
        # Calculate patch similarities (simplified)
        center_patch = patches[:, kernel_size**2//2:kernel_size**2//2+1]
        similarities = torch.exp(-torch.sum((patches - center_patch)**2, dim=1, keepdim=True) / (strength + 1e-6))
        
        # Weighted average
        weighted_patches = patches * similarities
        denoised = torch.sum(weighted_patches, dim=1, keepdim=True) / torch.sum(similarities, dim=1, keepdim=True)
        
        return denoised
    
    @torch.no_grad()
    def _calculate_depth_quality_gpu(self, original: torch.Tensor, processed: torch.Tensor) -> Dict:
        """Calculate depth quality metrics on GPU"""
        # Edge preservation
        orig_edges = self._calculate_edge_magnitude_gpu(original)
        proc_edges = self._calculate_edge_magnitude_gpu(processed)
        edge_preservation = torch.corrcoef(torch.stack([orig_edges.flatten(), proc_edges.flatten()]))[0, 1]
        
        # Smoothness
        smoothness = 1.0 / (1.0 + torch.mean(proc_edges))
        
        # Noise level (approximated by high-frequency content)
        noise_level = torch.std(processed - torch.nn.functional.avg_pool2d(processed, 3, stride=1, padding=1))
        
        # Coverage (non-zero pixels)
        coverage = torch.mean((processed > 0).float())
        
        return {
            'edge_preservation': float(edge_preservation.cpu()) if not torch.isnan(edge_preservation) else 0.0,
            'smoothness': float(smoothness.cpu()),
            'noise_level': float(noise_level.cpu()),
            'coverage': float(coverage.cpu())
        }
    
    @torch.no_grad()
    def _apply_edge_preserving_filter_gpu(self, depth_tensor: torch.Tensor) -> torch.Tensor:
        """Apply edge-preserving filter for inpainting"""
        if not getattr(self.config, 'depth_edge_preservation', True):
            return self._gaussian_smooth_gpu(depth_tensor)
        
        # Calculate edges
        edge_mag = self._calculate_edge_magnitude_gpu(depth_tensor)
        edge_threshold = getattr(self.config, 'depth_edge_threshold', 10.0)
        
        # Create edge mask
        edge_mask = edge_mag > edge_threshold
        
        # Apply different filtering based on edges
        smooth_result = self._gaussian_smooth_gpu(depth_tensor)
        
        # Preserve edges
        return torch.where(edge_mask, depth_tensor, smooth_result)
    
    @torch.no_grad()
    def _erode_mask_gpu(self, mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """Erode a binary mask using GPU operations"""
        # Use morphological erosion via min pooling
        padding = kernel_size // 2
        eroded = torch.nn.functional.max_pool2d(-mask.float(), kernel_size, stride=1, padding=padding)
        return (-eroded) > 0.5


class GPU3DPoseProcessor:
    """GPU-accelerated 3D pose processing with batch operations"""
    
    def __init__(self, config, device="cuda"):
        self.config = config
        self.device = device if torch.cuda.is_available() else "cpu"
        self.depth_processor = AdvancedDepthProcessor(config, device)
        
        # Pose caching for performance
        if getattr(config, 'enable_pose_caching', False):
            self.pose_cache = {}
            self.cache_size = getattr(config, 'pose_cache_size', 50)
        
        # Kalman filters for temporal consistency
        self.kalman_filters = {}
        
        # Batch processing settings
        self.batch_size = getattr(config, 'gpu_batch_size_3d', 8)
    
    @torch.no_grad()
    def batch_convert_2d_to_3d(self, poses_2d: List[Dict], depth_map: np.ndarray, 
                              bboxes: List[np.ndarray]) -> List[Dict]:
        """Convert multiple 2D poses to 3D in batch for GPU efficiency"""
        if not poses_2d or depth_map is None:
            return poses_2d
        
        # Process depth map once
        depth_tensor, depth_quality = self.depth_processor.process_depth_map_gpu(depth_map)
        
        # Group poses into batches
        pose_batches = [poses_2d[i:i + self.batch_size] for i in range(0, len(poses_2d), self.batch_size)]
        bbox_batches = [bboxes[i:i + self.batch_size] for i in range(0, len(bboxes), self.batch_size)]
        
        all_poses_3d = []
        
        for pose_batch, bbox_batch in zip(pose_batches, bbox_batches):
            batch_poses_3d = self._process_pose_batch_gpu(pose_batch, depth_tensor, bbox_batch, depth_quality)
            all_poses_3d.extend(batch_poses_3d)
        
        return all_poses_3d
    
    @torch.no_grad()
    def _process_pose_batch_gpu(self, pose_batch: List[Dict], depth_tensor: torch.Tensor, 
                               bbox_batch: List[np.ndarray], depth_quality: Dict) -> List[Dict]:
        """Process a batch of poses on GPU"""
        batch_poses_3d = []
        
        # Stack keypoints for batch processing
        batch_keypoints = []
        batch_info = []
        
        for i, (pose_2d, bbox) in enumerate(zip(pose_batch, bbox_batch)):
            keypoints_2d = pose_2d.get('keypoints', [])
            if len(keypoints_2d) == 0:
                batch_poses_3d.append(None)
                continue
            
            # Convert to tensor
            if hasattr(keypoints_2d, 'cpu'):
                keypoints_np = keypoints_2d.cpu().numpy()
            else:
                keypoints_np = np.array(keypoints_2d)
            
            batch_keypoints.append(keypoints_np)
            batch_info.append({
                'pose_2d': pose_2d,
                'bbox': bbox,
                'index': i
            })
        
        if not batch_keypoints:
            return [None] * len(pose_batch)
        
        # Convert to GPU tensors
        max_keypoints = max(len(kp) for kp in batch_keypoints)
        
        # Pad keypoints to same length
        padded_keypoints = []
        for kp in batch_keypoints:
            if len(kp) < max_keypoints:
                padding = np.zeros((max_keypoints - len(kp), kp.shape[1]))
                kp_padded = np.vstack([kp, padding])
            else:
                kp_padded = kp
            padded_keypoints.append(kp_padded)
        
        # Stack into batch tensor
        keypoints_tensor = torch.from_numpy(np.stack(padded_keypoints)).float().to(self.device)
        
        # Extract depths for all keypoints in batch
        batch_depths = self._extract_batch_depths_gpu(keypoints_tensor, depth_tensor)
        
        # Process each pose in the batch
        for i, (depths, info) in enumerate(zip(batch_depths, batch_info)):
            pose_3d = self._create_3d_pose_from_depths(
                keypoints_tensor[i], depths, info['pose_2d'], info['bbox'], depth_quality
            )
            batch_poses_3d.append(pose_3d)
        
        return batch_poses_3d
    
    @torch.no_grad()
    def _extract_batch_depths_gpu(self, keypoints_tensor: torch.Tensor, depth_tensor: torch.Tensor) -> torch.Tensor:
        """Extract depth values for batch of keypoints on GPU"""
        batch_size, num_keypoints, coord_dims = keypoints_tensor.shape
        h, w = depth_tensor.shape[-2:]
        
        # Extract x, y coordinates
        if coord_dims >= 2:
            x_coords = keypoints_tensor[:, :, 0]  # [batch_size, num_keypoints]
            y_coords = keypoints_tensor[:, :, 1]  # [batch_size, num_keypoints]
        else:
            return torch.zeros(batch_size, num_keypoints, device=self.device)
        
        # Clip coordinates to valid range
        x_coords = torch.clamp(x_coords, 0, w - 1)
        y_coords = torch.clamp(y_coords, 0, h - 1)
        
        # Convert to integer indices
        x_indices = x_coords.long()
        y_indices = y_coords.long()
        
        # Extract depths using advanced indexing
        depths = torch.zeros_like(x_coords)
        
        for b in range(batch_size):
            for k in range(num_keypoints):
                if coord_dims >= 3:  # Check confidence if available
                    confidence = keypoints_tensor[b, k, 2]
                    if confidence <= 0:
                        continue
                
                y_idx = y_indices[b, k]
                x_idx = x_indices[b, k]
                
                # Sample with neighborhood for robustness
                neighborhood_size = 2
                y_start = max(0, y_idx - neighborhood_size)
                y_end = min(h, y_idx + neighborhood_size + 1)
                x_start = max(0, x_idx - neighborhood_size)
                x_end = min(w, x_idx + neighborhood_size + 1)
                
                neighborhood = depth_tensor[y_start:y_end, x_start:x_end]
                valid_depths = neighborhood[neighborhood > 0]
                
                if len(valid_depths) > 0:
                    depths[b, k] = torch.median(valid_depths)
        
        return depths
    
    def _create_3d_pose_from_depths(self, keypoints_2d: torch.Tensor, depths: torch.Tensor, 
                                   pose_2d: Dict, bbox: np.ndarray, depth_quality: Dict) -> Dict:
        """Create 3D pose from 2D keypoints and depths"""
        keypoints_2d_np = keypoints_2d.cpu().numpy()
        depths_np = depths.cpu().numpy()
        
        # Create 3D keypoints
        keypoints_3d = []
        valid_count = 0
        
        for i, (kp_2d, depth) in enumerate(zip(keypoints_2d_np, depths_np)):
            if len(kp_2d) >= 2:
                x, y = kp_2d[0], kp_2d[1]
                confidence = kp_2d[2] if len(kp_2d) >= 3 else 1.0
                
                # Check validity
                if x > 0 and y > 0 and confidence > 0 and depth > 0:
                    keypoints_3d.append([x, y, depth, confidence])
                    valid_count += 1
                else:
                    keypoints_3d.append([0, 0, 0, 0])
            else:
                keypoints_3d.append([0, 0, 0, 0])
        
        keypoints_3d = np.array(keypoints_3d)
        
        # Apply advanced smoothing and filtering
        if getattr(self.config, 'enable_temporal_pose_smoothing', False):
            keypoints_3d = self._apply_temporal_smoothing_3d(keypoints_3d, pose_2d.get('track_id', -1))
        
        # Calculate enhanced 3D pose quality
        pose_3d_quality = self._calculate_enhanced_3d_quality(keypoints_3d, depth_quality)
        
        # Extract geometric features
        geometric_features = self._extract_enhanced_geometric_features(keypoints_3d)
        
        # Create 3D pose result
        pose_3d = {
            'keypoints_3d': keypoints_3d,
            'keypoints_2d': pose_2d.get('keypoints', []),
            'bbox': bbox,
            'method': pose_2d.get('method', 'unknown'),
            'confidence': pose_2d.get('confidence', 0.0),
            'confidence_3d': pose_3d_quality,
            'track_id': pose_2d.get('track_id', -1),
            'box_index': pose_2d.get('box_index', -1),
            'valid_keypoints_3d': valid_count,
            'pose_3d_features': geometric_features,
            'depth_quality_metrics': depth_quality
        }
        
        return pose_3d
    
    def _apply_temporal_smoothing_3d(self, keypoints_3d: np.ndarray, track_id: int) -> np.ndarray:
        """Apply Kalman filtering for temporal consistency"""
        if not getattr(self.config, 'enable_3d_kalman_filtering', False):
            return keypoints_3d
        
        if track_id not in self.kalman_filters:
            self._initialize_kalman_filter(track_id, keypoints_3d)
        
        # Apply Kalman filtering to each valid keypoint
        smoothed_keypoints = keypoints_3d.copy()
        
        for i, kp in enumerate(keypoints_3d):
            if kp[3] > 0:  # Valid keypoint
                filter_key = f"{track_id}_{i}"
                if filter_key in self.kalman_filters:
                    # Predict and update
                    kalman = self.kalman_filters[filter_key]
                    measurement = kp[:3]  # x, y, z
                    
                    # Simple Kalman update (simplified implementation)
                    if hasattr(kalman, 'state'):
                        kalman.state = kalman.state * 0.8 + measurement * 0.2
                        smoothed_keypoints[i, :3] = kalman.state
                    else:
                        kalman.state = measurement
        
        return smoothed_keypoints
    
    def _initialize_kalman_filter(self, track_id: int, keypoints_3d: np.ndarray):
        """Initialize Kalman filters for a track"""
        for i, kp in enumerate(keypoints_3d):
            if kp[3] > 0:  # Valid keypoint
                filter_key = f"{track_id}_{i}"
                # Simple filter object (would be replaced with proper Kalman filter)
                self.kalman_filters[filter_key] = type('KalmanFilter', (), {
                    'state': kp[:3].copy(),
                    'process_noise': getattr(self.config, 'kalman_process_noise', 0.1),
                    'measurement_noise': getattr(self.config, 'kalman_measurement_noise', 0.5)
                })()
    
    def _calculate_enhanced_3d_quality(self, keypoints_3d: np.ndarray, depth_quality: Dict) -> float:
        """Calculate enhanced 3D pose quality score"""
        valid_keypoints = keypoints_3d[keypoints_3d[:, 3] > 0]
        
        if len(valid_keypoints) < getattr(self.config, 'min_valid_keypoints_3d', 5):
            return 0.0
        
        # Basic quality components
        valid_ratio = len(valid_keypoints) / len(keypoints_3d)
        avg_confidence = np.mean(valid_keypoints[:, 3])
        
        # Depth consistency
        depths = valid_keypoints[:, 2]
        depth_consistency = 1.0 / (1.0 + np.var(depths) / 1000.0) if len(depths) > 1 else 0.5
        
        # Spatial coherence
        spatial_coherence = self._calculate_spatial_coherence(valid_keypoints)
        
        # Geometric plausibility
        geometric_plausibility = self._check_geometric_plausibility(valid_keypoints)
        
        # Incorporate depth map quality
        depth_factor = (depth_quality.get('edge_preservation', 0.5) + 
                       depth_quality.get('smoothness', 0.5) + 
                       depth_quality.get('coverage', 0.5)) / 3.0
        
        # Combined quality score
        quality = (valid_ratio * 0.2 + 
                  avg_confidence * 0.2 + 
                  depth_consistency * 0.2 + 
                  spatial_coherence * 0.2 + 
                  geometric_plausibility * 0.1 + 
                  depth_factor * 0.1)
        
        return float(quality)
    
    def _calculate_spatial_coherence(self, valid_keypoints: np.ndarray) -> float:
        """Calculate spatial coherence of 3D keypoints"""
        if len(valid_keypoints) < 3:
            return 0.0
        
        # Calculate distances between keypoints
        coords_3d = valid_keypoints[:, :3]
        distances = []
        
        for i in range(len(coords_3d)):
            for j in range(i + 1, len(coords_3d)):
                dist = np.linalg.norm(coords_3d[i] - coords_3d[j])
                distances.append(dist)
        
        if not distances:
            return 0.0
        
        # Coherence based on distance distribution
        distance_std = np.std(distances)
        distance_mean = np.mean(distances)
        
        if distance_mean == 0:
            return 0.0
        
        coherence = 1.0 / (1.0 + distance_std / distance_mean)
        return float(coherence)
    
    def _check_geometric_plausibility(self, valid_keypoints: np.ndarray) -> float:
        """Check geometric plausibility of 3D pose"""
        if not getattr(self.config, 'enable_geometric_consistency_check', False):
            return 1.0
        
        if len(valid_keypoints) < 5:
            return 0.0
        
        # Simple plausibility checks
        coords_3d = valid_keypoints[:, :3]
        
        # Check for reasonable pose dimensions
        x_range = np.ptp(coords_3d[:, 0])
        y_range = np.ptp(coords_3d[:, 1])
        z_range = np.ptp(coords_3d[:, 2])
        
        # Reasonable proportions (adjust for horse/human)
        if x_range == 0 or y_range == 0:
            return 0.0
        
        aspect_ratio = x_range / y_range
        reasonable_aspect = 0.3 <= aspect_ratio <= 3.0
        
        # Check depth variation is reasonable
        reasonable_depth = z_range < (x_range + y_range) * 2
        
        # Check for outliers
        centroid = np.mean(coords_3d, axis=0)
        distances_from_centroid = np.linalg.norm(coords_3d - centroid, axis=1)
        outlier_ratio = np.sum(distances_from_centroid > np.mean(distances_from_centroid) * 3) / len(distances_from_centroid)
        
        plausibility_score = (
            (1.0 if reasonable_aspect else 0.3) * 0.4 +
            (1.0 if reasonable_depth else 0.3) * 0.3 +
            (1.0 - outlier_ratio) * 0.3
        )
        
        return float(plausibility_score)
    
    def _extract_enhanced_geometric_features(self, keypoints_3d: np.ndarray) -> np.ndarray:
        """Extract enhanced geometric features for ReID"""
        valid_keypoints = keypoints_3d[keypoints_3d[:, 3] > 0]
        
        if len(valid_keypoints) < 3:
            return np.zeros(64)  # Return zero vector for invalid poses
        
        features = []
        coords_3d = valid_keypoints[:, :3]
        
        # Basic 3D statistics
        features.extend([
            np.mean(coords_3d[:, 0]),  # Mean X
            np.mean(coords_3d[:, 1]),  # Mean Y  
            np.mean(coords_3d[:, 2]),  # Mean Z
            np.std(coords_3d[:, 0]),   # Std X
            np.std(coords_3d[:, 1]),   # Std Y
            np.std(coords_3d[:, 2]),   # Std Z
            np.ptp(coords_3d[:, 0]),   # Range X
            np.ptp(coords_3d[:, 1]),   # Range Y
            np.ptp(coords_3d[:, 2]),   # Range Z
        ])
        
        # Volumetric features
        if getattr(self.config, 'enable_volumetric_features', True):
            volume = np.ptp(coords_3d[:, 0]) * np.ptp(coords_3d[:, 1]) * np.ptp(coords_3d[:, 2])
            surface_area = 2 * (np.ptp(coords_3d[:, 0]) * np.ptp(coords_3d[:, 1]) + 
                               np.ptp(coords_3d[:, 1]) * np.ptp(coords_3d[:, 2]) + 
                               np.ptp(coords_3d[:, 2]) * np.ptp(coords_3d[:, 0]))
            features.extend([volume / 1000000.0, surface_area / 100000.0])  # Normalized
        
        # Spatial distribution features
        if getattr(self.config, 'enable_spatial_distribution_features', True):
            centroid = np.mean(coords_3d, axis=0)
            distances_from_centroid = np.linalg.norm(coords_3d - centroid, axis=1)
            features.extend([
                np.mean(distances_from_centroid),
                np.std(distances_from_centroid),
                np.max(distances_from_centroid),
                np.min(distances_from_centroid)
            ])
        
        # Depth gradient features
        if getattr(self.config, 'enable_depth_gradient_features', True):
            if len(coords_3d) > 1:
                # Sort by Y coordinate and compute depth gradients
                sorted_indices = np.argsort(coords_3d[:, 1])
                sorted_depths = coords_3d[sorted_indices, 2]
                depth_gradients = np.gradient(sorted_depths)
                features.extend([
                    np.mean(depth_gradients),
                    np.std(depth_gradients),
                    np.max(np.abs(depth_gradients))
                ])
            else:
                features.extend([0, 0, 0])
        
        # Compactness features
        if getattr(self.config, 'enable_pose_compactness_features', True):
            if len(coords_3d) > 2:
                # Calculate convex hull volume approximation
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(coords_3d)
                    convex_volume = hull.volume
                    actual_volume = volume if 'volume' in locals() else 1.0
                    compactness = actual_volume / (convex_volume + 1e-6)
                    features.append(compactness)
                except:
                    features.append(0.5)  # Default compactness
            else:
                features.append(0.5)
        
        # Symmetry features (for horse poses)
        if len(coords_3d) > 4:
            # Simple symmetry check based on X coordinates
            left_side = coords_3d[coords_3d[:, 0] < np.mean(coords_3d[:, 0])]
            right_side = coords_3d[coords_3d[:, 0] >= np.mean(coords_3d[:, 0])]
            
            if len(left_side) > 0 and len(right_side) > 0:
                left_centroid = np.mean(left_side, axis=0)
                right_centroid = np.mean(right_side, axis=0)
                symmetry_score = 1.0 / (1.0 + np.linalg.norm(left_centroid - right_centroid))
                features.append(symmetry_score)
            else:
                features.append(0.5)
        else:
            features.append(0.5)
        
        # Pad or trim to exactly 64 features
        while len(features) < 64:
            features.append(0.0)
        features = features[:64]
        
        # Convert to numpy array and normalize
        features_array = np.array(features, dtype=np.float32)
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize
        norm = np.linalg.norm(features_array)
        if norm > 0:
            features_array = features_array / norm
        
        return features_array


class PoseEstimationManager:
    def __init__(self, config, superanimal_model=None):
        self.config = config
        self.superanimal = superanimal_model
        
        # Setup ViTPose models
        self.vitpose_processor = None
        self.vitpose_model = None
        self.vitpose_horse_processor = None
        self.vitpose_horse_model = None
        
        # GPU-accelerated 3D processing
        if getattr(config, 'enable_gpu_acceleration_3d', False):
            self.gpu_3d_processor = GPU3DPoseProcessor(config)
        else:
            self.gpu_3d_processor = None
        
        # 3D pose conversion parameters (legacy for fallback)
        self.depth_smoothing_factor = getattr(config, 'depth_smoothing_factor', 0.3)
        self.depth_outlier_threshold = getattr(config, 'depth_outlier_threshold', 3.0)
        self.min_valid_keypoints_3d = getattr(config, 'min_valid_keypoints_3d', 5)
        
        self.setup_vitpose_models()
    
    def setup_vitpose_models(self):
        # ViTPose for humans
        if self.config.human_pose_estimator == 'vitpose':
            if VITPOSE_AVAILABLE:
                try:
                    self.vitpose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
                    self.vitpose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")
                    self.vitpose_model.to(self.config.device)
                except Exception:
                    pass
        
        # ViTPose for horses (if needed)
        if self.config.horse_pose_estimator in ['vitpose', 'dual']:
            if VITPOSE_AVAILABLE:
                try:
                    self.vitpose_horse_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
                    self.vitpose_horse_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")
                    self.vitpose_horse_model.to(self.config.device)
                except Exception:
                    pass
    
    def estimate_human_poses(self, frame: np.ndarray, detections, depth_map=None):
        """Human pose estimation with GPU-accelerated 3D conversion"""
        if self.config.human_pose_estimator == 'none':
            return []
        
        if not sv or len(detections) == 0:
            return []
        
        if self.config.human_pose_estimator == 'vitpose':
            poses_2d = self._estimate_vitpose_human(frame, detections)
            
            # GPU-accelerated 3D conversion
            if (depth_map is not None and getattr(self.config, 'enable_3d_poses', False) 
                and self.gpu_3d_processor):
                bboxes = [detections.xyxy[pose.get('box_index', 0)] for pose in poses_2d]
                poses_3d = self.gpu_3d_processor.batch_convert_2d_to_3d(poses_2d, depth_map, bboxes)
                return self._ensure_one_pose_per_box(poses_3d, len(detections), "Human ViTPose 3D GPU")
            
            # Fallback to CPU 3D conversion
            elif depth_map is not None and getattr(self.config, 'enable_3d_poses', False):
                poses_3d = []
                for pose_2d in poses_2d:
                    bbox = detections.xyxy[pose_2d.get('box_index', 0)]
                    pose_3d = self.convert_2d_to_3d_pose(pose_2d, depth_map, bbox)
                    if pose_3d is not None:
                        poses_3d.append(pose_3d)
                    else:
                        poses_3d.append(pose_2d)
                return self._ensure_one_pose_per_box(poses_3d, len(detections), "Human ViTPose 3D CPU")
            
            return self._ensure_one_pose_per_box(poses_2d, len(detections), "Human ViTPose 2D")
        
        return []
    
    def estimate_horse_poses(self, frame: np.ndarray, detections, depth_map=None):
        """Horse pose estimation with GPU-accelerated 3D conversion"""
        if self.config.horse_pose_estimator == 'none':
            return []
        
        if not sv or len(detections) == 0:
            return []
        
        if self.config.horse_pose_estimator == 'superanimal':
            poses_2d = self._estimate_superanimal_only(frame, detections)
            
            # GPU-accelerated 3D conversion
            if (depth_map is not None and getattr(self.config, 'enable_3d_poses', False) 
                and self.gpu_3d_processor):
                bboxes = [pose['box'] for pose in poses_2d]
                poses_3d = self.gpu_3d_processor.batch_convert_2d_to_3d(poses_2d, depth_map, bboxes)
                return self._ensure_one_pose_per_box(poses_3d, len(detections), "SuperAnimal 3D GPU")
            
            return self._ensure_one_pose_per_box(poses_2d, len(detections), "SuperAnimal 2D")
            
        elif self.config.horse_pose_estimator == 'vitpose':
            poses_2d = self._estimate_vitpose_horse_only(frame, detections)
            
            # GPU-accelerated 3D conversion
            if (depth_map is not None and getattr(self.config, 'enable_3d_poses', False) 
                and self.gpu_3d_processor):
                bboxes = [detections.xyxy[pose.get('box_index', 0)] for pose in poses_2d]
                poses_3d = self.gpu_3d_processor.batch_convert_2d_to_3d(poses_2d, depth_map, bboxes)
                return self._ensure_one_pose_per_box(poses_3d, len(detections), "Horse ViTPose 3D GPU")
            
            return self._ensure_one_pose_per_box(poses_2d, len(detections), "Horse ViTPose 2D")
            
        elif self.config.horse_pose_estimator == 'dual':
            return self._estimate_dual_competition_3d(frame, detections, depth_map)
        
        return []
    
    def _estimate_dual_competition_3d(self, frame: np.ndarray, detections, depth_map=None):
        """Dual competition mode with GPU-accelerated 3D conversion"""
        if not sv or len(detections) == 0:
            return []
        
        # Get poses from both methods
        superanimal_poses_2d = self._estimate_superanimal_only(frame, detections)
        vitpose_poses_2d = self._estimate_vitpose_horse_only(frame, detections)
        
        # GPU-accelerated 3D conversion for both methods
        if (depth_map is not None and getattr(self.config, 'enable_3d_poses', False) 
            and self.gpu_3d_processor):
            
            # Convert SuperAnimal poses
            sa_bboxes = [pose['box'] for pose in superanimal_poses_2d]
            superanimal_poses_3d = self.gpu_3d_processor.batch_convert_2d_to_3d(
                superanimal_poses_2d, depth_map, sa_bboxes)
            
            # Convert ViTPose poses
            vp_bboxes = [detections.xyxy[pose.get('box_index', 0)] for pose in vitpose_poses_2d]
            vitpose_poses_3d = self.gpu_3d_processor.batch_convert_2d_to_3d(
                vitpose_poses_2d, depth_map, vp_bboxes)
        else:
            superanimal_poses_3d = superanimal_poses_2d
            vitpose_poses_3d = vitpose_poses_2d
        
        # Competition: Pick best pose per detection box (using 3D confidence if available)
        best_poses = []
        num_boxes = len(detections)
        
        for box_idx in range(num_boxes):
            superanimal_conf = 0.0
            vitpose_conf = 0.0
            superanimal_pose = None
            vitpose_pose = None
            
            # Find SuperAnimal pose for this box
            for pose in superanimal_poses_3d:
                if pose and pose.get('box_index') == box_idx:
                    superanimal_conf = pose.get('confidence_3d', pose.get('confidence', 0))
                    superanimal_pose = pose
                    break
            
            # Find ViTPose pose for this box
            for pose in vitpose_poses_3d:
                if pose and pose.get('box_index') == box_idx:
                    vitpose_conf = pose.get('confidence_3d', pose.get('confidence', 0))
                    vitpose_pose = pose
                    break
            
            # Pick the winner for this specific box (prefer 3D confidence)
            if superanimal_conf > vitpose_conf and superanimal_pose:
                best_poses.append(superanimal_pose)
            elif vitpose_conf > 0 and vitpose_pose:
                best_poses.append(vitpose_pose)
        
        return best_poses
    
    def extract_3d_pose_features(self, pose_3d):
        """Extract features from 3D pose for ReID (delegated to GPU processor)"""
        if self.gpu_3d_processor:
            return self.gpu_3d_processor._extract_enhanced_geometric_features(pose_3d['keypoints_3d'])
        else:
            # Fallback to basic feature extraction
            return self._extract_basic_3d_features(pose_3d)
    
    def _extract_basic_3d_features(self, pose_3d):
        """Basic 3D feature extraction for fallback"""
        if 'keypoints_3d' not in pose_3d:
            return np.zeros(32)
        
        keypoints_3d = pose_3d['keypoints_3d']
        valid_keypoints = keypoints_3d[keypoints_3d[:, 3] > 0]
        
        if len(valid_keypoints) < 3:
            return np.zeros(32)
        
        features = []
        
        # Basic 3D statistics
        x_coords = valid_keypoints[:, 0]
        y_coords = valid_keypoints[:, 1]
        z_coords = valid_keypoints[:, 2]
        
        features.extend([
            np.ptp(x_coords), np.ptp(y_coords), np.ptp(z_coords),
            np.mean(z_coords), np.std(z_coords)
        ])
        
        # Pad to 32 features
        while len(features) < 32:
            features.append(0.0)
        
        return np.array(features[:32])
    
    # Legacy 3D conversion method (CPU fallback)
    def convert_2d_to_3d_pose(self, pose_2d, depth_map, bbox):
        """Legacy CPU-based 2D to 3D conversion (fallback)"""
        if pose_2d is None or depth_map is None:
            return None
        
        keypoints_2d = pose_2d['keypoints']
        if hasattr(keypoints_2d, 'cpu'):
            keypoints_2d = keypoints_2d.cpu().numpy()
        
        keypoints_3d = []
        valid_depths = []
        
        # Extract depth values for each keypoint
        for i, kpt in enumerate(keypoints_2d):
            if len(kpt) == 2:  # ViTPose format [x, y]
                x, y = int(kpt[0]), int(kpt[1])
                confidence = pose_2d['scores'][i] if 'scores' in pose_2d else 1.0
            elif len(kpt) == 3:  # SuperAnimal format [x, y, confidence]
                x, y, confidence = int(kpt[0]), int(kpt[1]), kpt[2]
            else:
                keypoints_3d.append([0, 0, 0, 0])
                continue
            
            if x == -1 or y == -1 or confidence <= 0:
                keypoints_3d.append([0, 0, 0, 0])
                continue
            
            if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
                depth_window = depth_map[max(0, y-2):min(depth_map.shape[0], y+3),
                                       max(0, x-2):min(depth_map.shape[1], x+3)]
                
                if depth_window.size > 0:
                    depth_value = np.median(depth_window[depth_window > 0]) if np.any(depth_window > 0) else 0
                    valid_depths.append(depth_value)
                    keypoints_3d.append([x, y, depth_value, confidence])
                else:
                    keypoints_3d.append([0, 0, 0, 0])
            else:
                keypoints_3d.append([0, 0, 0, 0])
        
        if not valid_depths:
            return None
        
        keypoints_3d = np.array(keypoints_3d)
        pose_3d_quality = len([kpt for kpt in keypoints_3d if kpt[3] > 0]) / len(keypoints_3d)
        
        pose_3d = {
            'keypoints_3d': keypoints_3d,
            'keypoints_2d': keypoints_2d,
            'bbox': bbox,
            'method': pose_2d.get('method', 'unknown'),
            'confidence': pose_2d.get('confidence', 0.0),
            'confidence_3d': pose_3d_quality,
            'track_id': pose_2d.get('track_id', -1),
            'box_index': pose_2d.get('box_index', -1),
            'valid_keypoints_3d': len([kpt for kpt in keypoints_3d if kpt[3] > 0])
        }
        
        return pose_3d
    
    # Keep existing methods unchanged
    def select_best_pose_in_box(self, poses_in_box, method_name="Unknown"):
        """Select best pose if multiple candidates exist"""
        if not poses_in_box or len(poses_in_box) == 0:
            return None
        
        if len(poses_in_box) == 1:
            return poses_in_box[0]
        
        best_pose = None
        best_confidence = -1
        
        for pose in poses_in_box:
            confidence = pose.get('confidence_3d', pose.get('confidence', 0))
            if confidence > best_confidence:
                best_confidence = confidence
                best_pose = pose
        
        return best_pose if best_pose else poses_in_box[0]
    
    def _ensure_one_pose_per_box(self, poses, num_boxes, method_name):
        """Ensure exactly one pose per detection box"""
        if not poses:
            return []
        
        final_poses = []
        
        for box_idx in range(num_boxes):
            poses_for_this_box = [p for p in poses if p and p.get('box_index') == box_idx]
            
            if poses_for_this_box:
                best_pose = self.select_best_pose_in_box(poses_for_this_box, method_name)
                if best_pose:
                    final_poses.append(best_pose)
        
        return final_poses
    
    # Existing 2D estimation methods (unchanged)
    def _estimate_vitpose_human(self, frame: np.ndarray, detections):
        """ViTPose for humans - processes each detection box independently"""
        if not self.vitpose_model or not sv or len(detections) == 0:
            return []
        
        try:
            from PIL import Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            all_poses = []
            
            for box_idx, box in enumerate(detections.xyxy):
                x1, y1, x2, y2 = box
                coco_box = [x1, y1, x2-x1, y2-y1]
                
                inputs = self.vitpose_processor(pil_image, boxes=[[coco_box]], return_tensors="pt").to(self.config.device)
                
                with torch.no_grad():
                    outputs = self.vitpose_model(**inputs)
                
                pose_results = self.vitpose_processor.post_process_pose_estimation(outputs, boxes=[[coco_box]])
                
                if pose_results and len(pose_results[0]) > 0:
                    pose_result = pose_results[0][0]
                    
                    keypoints = pose_result['keypoints'].cpu().numpy() if hasattr(pose_result['keypoints'], 'cpu') else pose_result['keypoints']
                    scores = pose_result['scores'].cpu().numpy() if hasattr(pose_result['scores'], 'cpu') else pose_result['scores']
                    
                    conf_threshold = self.config.confidence_human_pose
                    filtered_keypoints = []
                    filtered_scores = []
                    valid_count = 0
                    total_confidence = 0.0
                    
                    for kpt, score in zip(keypoints, scores):
                        if score > conf_threshold:
                            filtered_keypoints.append(kpt)
                            filtered_scores.append(score)
                            valid_count += 1
                            total_confidence += score
                        else:
                            filtered_keypoints.append([-1.0, -1.0])
                            filtered_scores.append(0.0)
                    
                    avg_confidence = total_confidence / valid_count if valid_count > 0 else 0.0
                    
                    pose_with_confidence = {
                        'keypoints': torch.tensor(filtered_keypoints) if hasattr(pose_result['keypoints'], 'cpu') else np.array(filtered_keypoints),
                        'scores': torch.tensor(filtered_scores) if hasattr(pose_result['scores'], 'cpu') else np.array(filtered_scores),
                        'confidence': avg_confidence,
                        'box_index': box_idx
                    }
                    
                    all_poses.append(pose_with_confidence)
            
            return all_poses
            
        except Exception:
            return []
    
    def _estimate_superanimal_only(self, frame: np.ndarray, detections):
        """SuperAnimal pose estimation"""
        if not self.superanimal or not sv or len(detections) == 0:
            return []
        
        poses = self.superanimal.estimate_pose(frame, detections)
        
        for i, pose in enumerate(poses):
            pose['box_index'] = i
        
        return poses
    
    def _estimate_vitpose_horse_only(self, frame: np.ndarray, detections):
        """ViTPose for horses"""
        if not self.vitpose_horse_model or not sv or len(detections) == 0:
            return []
        
        try:
            from PIL import Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            all_poses = []
            
            for box_idx, box in enumerate(detections.xyxy):
                x1, y1, x2, y2 = box
                coco_box = [x1, y1, x2-x1, y2-y1]
                
                inputs = self.vitpose_horse_processor(pil_image, boxes=[[coco_box]], return_tensors="pt").to(self.config.device)
                
                with torch.no_grad():
                    outputs = self.vitpose_horse_model(**inputs)
                
                pose_results = self.vitpose_horse_processor.post_process_pose_estimation(outputs, boxes=[[coco_box]])
                
                if pose_results and len(pose_results[0]) > 0:
                    pose_result = pose_results[0][0]
                    
                    keypoints = pose_result['keypoints'].cpu().numpy() if hasattr(pose_result['keypoints'], 'cpu') else pose_result['keypoints']
                    scores = pose_result['scores'].cpu().numpy() if hasattr(pose_result['scores'], 'cpu') else pose_result['scores']
                    
                    conf_threshold = self.config.confidence_horse_pose_vitpose
                    filtered_keypoints = []
                    valid_count = 0
                    total_confidence = 0.0
                    
                    for kpt, score in zip(keypoints, scores):
                        if score > conf_threshold:
                            filtered_keypoints.append([kpt[0], kpt[1], score])
                            valid_count += 1
                            total_confidence += score
                        else:
                            filtered_keypoints.append([-1.0, -1.0, 0.0])
                    
                    avg_confidence = total_confidence / valid_count if valid_count > 0 else 0.0
                    
                    converted_pose = {
                        'keypoints': np.array(filtered_keypoints),
                        'box': box,
                        'method': 'ViTPose',
                        'confidence': avg_confidence,
                        'box_index': box_idx
                    }
                    
                    all_poses.append(converted_pose)
            
            return all_poses
            
        except Exception:
            return []