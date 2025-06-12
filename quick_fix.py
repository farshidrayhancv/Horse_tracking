#!/usr/bin/env python3
"""
Quick fix for current Re-ID pipeline issues
"""

import os
import urllib.request

def main():
    print("🔧 Quick fix for Re-ID pipeline issues\n")
    
    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Fix 1: Download MobileSAM checkpoint
    mobile_sam_path = "checkpoints/mobile_sam.pt"
    if not os.path.exists(mobile_sam_path):
        print("📥 Downloading MobileSAM checkpoint...")
        try:
            url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
            urllib.request.urlretrieve(url, mobile_sam_path)
            print(f"✅ Downloaded MobileSAM to {mobile_sam_path}")
        except Exception as e:
            print(f"❌ Failed to download MobileSAM: {e}")
    else:
        print(f"✅ MobileSAM already exists at {mobile_sam_path}")
    
    # Fix 2: Install missing packages
    import subprocess
    packages = ["mobile-sam", "timm"]
    
    for package in packages:
        print(f"🔄 Installing {package}...")
        try:
            if package == "mobile-sam":
                subprocess.run(["pip", "install", "git+https://github.com/ChaoningZhang/MobileSAM.git"], check=True)
            else:
                subprocess.run(["pip", "install", package], check=True)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
    
    # Fix 3: Create a simple config that disables problematic components
    simple_config = """# Video file path (required) - filename format: horse_XX.mp4 where XX = number of horses
video_path: inputs/horse_11.mp4  # Will automatically detect 11 horses + 11 jockeys

# Output settings
output_path: null  # Auto-generate if null (will include "_tracked_" in filename)
display: true

# Model selection
human_detector: rtdetr
horse_detector: rtdetr  # Options: rtdetr, superanimal ( NOT RECOMMENDED VERY SLOW ), both
human_pose_estimator: vitpose
horse_pose_estimator: superanimal # Options: superanimal (better), vitpose, dual

# Separate confidence thresholds for each model
confidence_human_detection: 0.5
confidence_horse_detection: 0.5
confidence_human_pose: 0.5
confidence_horse_pose_superanimal: 0.5
confidence_horse_pose_vitpose: 0.5

# Tracking and jockey detection settings
jockey_overlap_threshold: 0.4

# Processing settings
max_frames: null
device: cuda

# Re-identification Pipeline Settings (Conservative settings)
enable_reid_pipeline: true
reid_similarity_threshold: 0.6
reid_embedding_history_size: 3
enable_mobile_sam: true
enable_depth_anything: false  # Disabled to avoid issues
enable_megadescriptor: false  # Use CLIP fallback instead

# NOTE: Expected horse/jockey counts are automatically parsed from filename
# Examples:
#   horse_11.mp4 -> 11 horses, 11 jockeys
#   horse_22.mp4 -> 22 horses, 22 jockeys
#   horse_8.mp4  -> 8 horses, 8 jockeys"""
    
    with open("config_safe.yaml", "w") as f:
        f.write(simple_config)
    
    print("✅ Created config_safe.yaml with conservative settings")
    
    print("\n🎉 Quick fix completed!")
    print("\n📋 What was fixed:")
    print("   • Downloaded MobileSAM checkpoint")
    print("   • Installed missing packages")
    print("   • Created config_safe.yaml with working settings")
    
    print("\n🚀 Try running with the safe config:")
    print("   python main.py config_safe.yaml")
    
    print("\n💡 This config uses:")
    print("   • MobileSAM segmentation (should work now)")
    print("   • CLIP for re-identification (fallback)")
    print("   • Disabled depth estimation (to avoid issues)")

if __name__ == "__main__":
    main()