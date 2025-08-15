#!/usr/bin/env python3
"""
Test script to debug Roboflow integration - match working approach
"""

import cv2
import numpy as np
import supervision as sv
from pathlib import Path
from config import Config

def test_roboflow_detection(config_file: str, test_frames: int = 10):
    """Test Roboflow detection using same approach as working code"""
    
    # Load config
    config = Config(config_file)
    
    print(f"🧪 Testing Roboflow Detection (Matching Working Code)")
    print(f"   Model: {config.roboflow_model_id}")
    print(f"   API Key: {config.roboflow_api_key[:10]}..." if config.roboflow_api_key else "   API Key: NOT SET")
    print(f"   Confidence: {config.roboflow_confidence}")
    print(f"   Video: {config.video_path}")
    
    # Test direct approach like working code
    try:
        from inference import get_model
        print(f"✅ Inference SDK available")
        
        # Load model exactly like working code
        model = get_model(model_id=config.roboflow_model_id, api_key=config.roboflow_api_key)
        print(f"✅ Model loaded successfully")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Open video
    cap = cv2.VideoCapture(config.video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {config.video_path}")
        return
    
    print(f"\n📹 Testing first {test_frames} frames with direct inference...")
    
    for frame_num in range(test_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Could not read frame {frame_num}")
            break
        
        print(f"\n--- Frame {frame_num} ---")
        print(f"Frame shape: {frame.shape}")
        
        # Test detection exactly like working code
        try:
            # Step 1: Run inference (like working code)
            results = model.infer(frame)[0]  # Take first result
            print(f"✅ Inference successful, got results type: {type(results)}")
            
            # Step 2: Convert to detections (like working code)  
            detections = sv.Detections.from_inference(results)
            print(f"✅ Conversion successful")
            
            print(f"✅ Detection Results:")
            print(f"   Count: {len(detections)}")
            
            if len(detections) > 0:
                print(f"   Boxes shape: {detections.xyxy.shape}")
                print(f"   Confidences: {detections.confidence}")
                if hasattr(detections, 'class_id') and detections.class_id is not None:
                    print(f"   Class IDs: {detections.class_id}")
                if hasattr(detections, 'mask') and detections.mask is not None:
                    print(f"   Masks shape: {detections.mask.shape}")
                    print(f"   Any true mask pixels: {detections.mask.any()}")
                
                # Test individual detection details
                for i, (bbox, conf) in enumerate(zip(detections.xyxy, detections.confidence)):
                    x1, y1, x2, y2 = bbox
                    w, h = x2 - x1, y2 - y1
                    print(f"   Detection {i}: conf={conf:.3f}, size={w:.0f}x{h:.0f}, pos=({x1:.0f},{y1:.0f})")
            else:
                print(f"   ❌ No detections found")
                print(f"   Try lowering confidence threshold from {config.roboflow_confidence}")
                
        except Exception as e:
            print(f"❌ Detection failed: {e}")
            import traceback
            traceback.print_exc()
    
    cap.release()
    
    print(f"\n🏁 Direct Test Complete!")
    print(f"\n💡 Comparison with your working code:")
    print(f"   Working video: /home/farshid/Downloads/del_mar_pan_poc_1.mp4")  
    print(f"   Test video: {config.video_path}")
    print(f"   Working model: del_mar_pan_seg-mdxam/1")
    print(f"   Test model: {config.roboflow_model_id}")
    
    if config.video_path != "/home/farshid/Downloads/del_mar_pan_poc_1.mp4":
        print(f"\n🎯 SUGGESTION: Try testing with the same video file:")
        print(f"   Change video_path in config.yaml to: '/home/farshid/Downloads/del_mar_pan_poc_1.mp4'")

def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python test_roboflow.py config.yaml")
        sys.exit(1)
    
    test_roboflow_detection(sys.argv[1])

if __name__ == "__main__":
    main()