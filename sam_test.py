#!/usr/bin/env python3
"""
SAM Model Demo Script - Test both MobileSAM and SAM2 with center point prompts
This script demonstrates the new SAM model selection feature in the horse tracking system.
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Import your reid pipeline
from reid_pipeline import ReIDPipeline
from config import Config


def create_demo_config(sam_model='sam2', device='cuda'):
    """Create a demo configuration for testing SAM models"""
    config = Config()
    config.sam_model = sam_model
    config.device = device if torch.cuda.is_available() else 'cpu'
    config.enable_reid_pipeline = True
    config.enable_depth_anything = True
    config.enable_megadescriptor = False  # Skip for demo
    return config


def segment_image_with_both_sams(image_path: str, output_dir: str = "sam_comparison"):
    """
    Compare MobileSAM vs SAM2 segmentation on the same image using center point prompts
    
    Args:
        image_path (str): Path to the input image
        output_dir (str): Directory to save comparison results
    """
    
    # Load the image
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"Loaded image: {image_path}")
        print(f"Image shape: {image.shape}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Calculate center point
    height, width = image.shape[:2]
    center_x = width // 2
    center_y = height // 2
    center_point = np.array([[center_x, center_y]])
    center_labels = np.array([1])  # Foreground point
    
    print(f"Image dimensions: {width} x {height}")
    print(f"Center point: ({center_x}, {center_y})")
    
    # Create fake detection at center for pipeline testing
    class FakeDetection:
        def __init__(self, center_x, center_y, size=200):
            self.xyxy = np.array([[
                max(0, center_x - size//2),
                max(0, center_y - size//2), 
                min(width, center_x + size//2),
                min(height, center_y + size//2)
            ]])
    
    fake_detection = FakeDetection(center_x, center_y)
    
    results = {}
    
    # Test both SAM models
    for sam_model in ['mobilesam', 'sam2']:
        print(f"\n🔄 Testing {sam_model.upper()}...")
        
        try:
            # Create config for this SAM model
            config = create_demo_config(sam_model=sam_model)
            
            # Initialize pipeline
            pipeline = ReIDPipeline(config)
            
            if pipeline.sam_predictor is None:
                print(f"❌ {sam_model.upper()} not available, skipping...")
                continue
            
            # Set image
            pipeline.sam_predictor.set_image(image)
            
            # Generate masks using center point
            if sam_model == 'sam2':
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    masks, scores, logits = pipeline.sam_predictor.predict(
                        point_coords=center_point,
                        point_labels=center_labels,
                        multimask_output=True,
                    )
            else:  # mobilesam
                masks, scores, logits = pipeline.sam_predictor.predict(
                    point_coords=center_point,
                    point_labels=center_labels,
                    multimask_output=True,
                )
            
            print(f"✅ {sam_model.upper()}: Generated {len(masks)} masks")
            print(f"   Scores: {scores}")
            
            # Store results
            results[sam_model] = {
                'masks': masks,
                'scores': scores,
                'best_mask': masks[np.argmax(scores)]
            }
            
        except Exception as e:
            print(f"❌ {sam_model.upper()} failed: {e}")
            continue
    
    # Create comparison visualization
    if results:
        create_comparison_plot(image_rgb, center_x, center_y, results, output_path)
        print(f"\n✅ Comparison saved to {output_path}/")
    else:
        print("❌ No SAM models were successfully tested")


def create_comparison_plot(image_rgb, center_x, center_y, results, output_path):
    """Create a comprehensive comparison plot"""
    
    num_models = len(results)
    if num_models == 0:
        return
    
    # Create figure
    fig, axes = plt.subplots(2, max(3, num_models + 1), figsize=(18, 10))
    fig.suptitle('SAM Model Comparison: MobileSAM vs SAM2', fontsize=16, fontweight='bold')
    
    # Original image (top left)
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].plot(center_x, center_y, 'ro', markersize=15, label='Center Prompt')
    axes[0, 0].set_title('Original Image + Center Point', fontweight='bold')
    axes[0, 0].axis('off')
    axes[0, 0].legend()
    
    # Hide unused subplot
    if num_models < 2:
        axes[0, 2].axis('off')
    
    col_idx = 1
    for model_name, result in results.items():
        best_mask = result['best_mask']
        best_score = np.max(result['scores'])
        
        # Best mask overlay (top row)
        axes[0, col_idx].imshow(image_rgb)
        axes[0, col_idx].imshow(best_mask, alpha=0.6, cmap='Blues')
        axes[0, col_idx].plot(center_x, center_y, 'ro', markersize=12)
        axes[0, col_idx].set_title(f'{model_name.upper()}\nBest Mask (Score: {best_score:.3f})', fontweight='bold')
        axes[0, col_idx].axis('off')
        
        # Segmented subject (bottom row)
        masked_image = image_rgb.copy()
        masked_image[~best_mask] = [240, 240, 240]  # Light gray background
        axes[1, col_idx].imshow(masked_image)
        axes[1, col_idx].set_title(f'{model_name.upper()} Segmented Subject', fontweight='bold')
        axes[1, col_idx].axis('off')
        
        col_idx += 1
    
    # Side-by-side mask comparison (bottom left)
    if len(results) == 2:
        model_names = list(results.keys())
        mask1 = results[model_names[0]]['best_mask']
        mask2 = results[model_names[1]]['best_mask']
        
        # Create difference visualization
        diff_vis = np.zeros((*mask1.shape, 3))
        diff_vis[mask1 & ~mask2] = [1, 0, 0]  # Red: only in first model
        diff_vis[~mask1 & mask2] = [0, 0, 1]  # Blue: only in second model  
        diff_vis[mask1 & mask2] = [0, 1, 0]   # Green: in both models
        
        axes[1, 0].imshow(image_rgb)
        axes[1, 0].imshow(diff_vis, alpha=0.7)
        axes[1, 0].plot(center_x, center_y, 'yo', markersize=12)
        axes[1, 0].set_title('Mask Differences\n🔴 MobileSAM only  🔵 SAM2 only  🟢 Both', fontweight='bold')
        axes[1, 0].axis('off')
    else:
        axes[1, 0].axis('off')
    
    # Hide unused subplots
    for i in range(max(3, num_models + 1)):
        if i >= col_idx:
            axes[0, i].axis('off')
            if i > 0:  # Don't hide bottom-left comparison
                axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / 'sam_comparison.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    # Save individual masks
    for model_name, result in results.items():
        best_mask = result['best_mask']
        mask_image = (best_mask * 255).astype(np.uint8)
        cv2.imwrite(str(output_path / f'{model_name}_mask.png'), mask_image)
        
        # Save masked subject
        masked_subject = image_rgb.copy()
        masked_subject[~best_mask] = [255, 255, 255]  # White background
        masked_subject_bgr = cv2.cvtColor(masked_subject, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path / f'{model_name}_subject.png'), masked_subject_bgr)
    
    print(f"📁 Saved comparison plot and individual results")


def main():
    parser = argparse.ArgumentParser(description='Compare MobileSAM vs SAM2 segmentation')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('--output-dir', default='sam_comparison', 
                       help='Output directory for results (default: sam_comparison)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image_path).exists():
        print(f"❌ Error: Image file '{args.image_path}' not found")
        return
    
    print("🐴 SAM Model Comparison Demo")
    print("=" * 50)
    print(f"Input: {args.image_path}")
    print(f"Output: {args.output_dir}/")
    print(f"Device: {args.device}")
    
    segment_image_with_both_sams(args.image_path, args.output_dir)


if __name__ == "__main__":
    # Example usage if run directly
    # Make sure you have the required dependencies installed:
    # pip install torch torchvision
    # pip install git+https://github.com/ChaoningZhang/MobileSAM.git
    # pip install git+https://github.com/facebookresearch/segment-anything-2.git
    # pip install pillow matplotlib numpy opencv-python
    
    main()