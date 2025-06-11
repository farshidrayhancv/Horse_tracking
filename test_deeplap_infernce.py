
# STEP 1: Apply the COCO patch BEFORE importing DeepLabCut
print("🔧 Applying COCO evaluation patch...")

from pycocotools.coco import COCO

# Store the original loadRes method
original_loadRes = COCO.loadRes

def patched_loadRes(self, resFile):
    """Patched version that ensures dataset has info field"""
    
    # Ensure self.dataset has required fields
    if not hasattr(self, 'dataset') or self.dataset is None:
        self.dataset = {}
        
    if 'info' not in self.dataset:
        self.dataset['info'] = {
            "description": "DeepLabCut SuperAnimal Results",
            "url": "",
            "version": "1.0", 
            "year": 2025,
            "contributor": "DeepLabCut",
            "date_created": "2025-06-09"
        }
        
    if 'licenses' not in self.dataset:
        self.dataset['licenses'] = [
            {
                "url": "",
                "id": 1,
                "name": "Unknown License"
            }
        ]
    
    # Call the original method
    return original_loadRes(self, resFile)

# Apply the patch
COCO.loadRes = patched_loadRes
print("✅ COCO evaluation patched successfully!")

# STEP 2: Now import and run DeepLabCut with the patch active
print("🚀 Starting DeepLabCut with patched evaluation...")

import deeplabcut
import torch

# See options
# print(help(deeplabcut.video_inference_superanimal))

# Clear GPU cache before starting
torch.cuda.empty_cache()

video_path = "horse.mp4"
superanimal_name = "superanimal_quadruped"

# Run with patched COCO evaluation
deeplabcut.video_inference_superanimal(
    [video_path], 
    superanimal_name, 
    model_name="hrnet_w32", 
    detector_name="fasterrcnn_resnet50_fpn_v2",
#    detector_name="fasterrcnn_mobilenet_v3_fpn",    
    video_adapt=True,
    scale_list=range(500, 1200, 100),
    
    # Your preferred settings
    video_adapt_batch_size=2,      # Your target setting
    batch_size=3,                  # Inference batch
    detector_batch_size=5,         # Detection batch
    
    # Full training with patch
    detector_epochs=43,            # Full detector training
    pose_epochs=13,                 # Pose training
    
    # Keep full functionality
    max_individuals=8,             # All 8 horses
    
    # Conservative thresholds for high-quality results
    pseudo_threshold=0.1,          # More selective training data
    bbox_threshold=0.8,            # Standard bounding box confidence  
    pcutoff=0.1                    # Only confident keypoints in output
)

print("🎉 SUCCESS! Training completed with patched evaluation!")
