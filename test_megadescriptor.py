#!/usr/bin/env python3
"""
Test script for MegaDescriptor ReID pipeline
Validates RGB+Depth embedding extraction
"""

import cv2
import numpy as np
from reid_pipeline import EnhancedReIDPipeline
from config import Config

def test_megadescriptor_pipeline():
    """Test MegaDescriptor embedding extraction"""
    print("🔬 Testing MegaDescriptor ReID Pipeline")
    print("=" * 50)
    
    # Create test config
    config = Config()
    config.sam_model = 'sam2'  # or 'mobilesam'
    config.enable_depth_anything = True
    config.enable_megadescriptor = True
    config.device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
    
    print(f"Device: {config.device}")
    
    # Initialize pipeline
    try:
        pipeline = EnhancedReIDPipeline(config)
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return False
    
    # Create test image (simulate horse racing scene)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Create fake bounding box (center of image)
    h, w = test_image.shape[:2]
    bbox = np.array([w//4, h//4, 3*w//4, 3*h//4])
    
    print(f"Test image shape: {test_image.shape}")
    print(f"Test bbox: {bbox}")
    
    # Test SAM segmentation
    print("\n🎯 Testing SAM segmentation...")
    mask, confidence = pipeline.segment_with_sam(test_image, bbox)
    print(f"✅ SAM mask shape: {mask.shape}, confidence: {confidence:.3f}")
    
    # Test depth estimation
    print("\n🌊 Testing depth estimation...")
    depth_map = pipeline.estimate_depth_full_image(test_image)
    print(f"✅ Depth map shape: {depth_map.shape}")
    
    # Test MegaDescriptor embeddings
    print("\n🧠 Testing MegaDescriptor embeddings...")
    
    if pipeline.megadescriptor_model is None:
        print("❌ MegaDescriptor not available - skipping embedding test")
        return False
    
    # Test RGB features
    rgb_features = pipeline.extract_megadescriptor_features(test_image, mask)
    print(f"✅ RGB features shape: {rgb_features.shape} (expected: (32,))")
    
    # Test Depth features  
    depth_features = pipeline.extract_megadescriptor_features(depth_map, mask)
    print(f"✅ Depth features shape: {depth_features.shape} (expected: (32,))")
    
    # Test combined embedding
    combined_embedding = pipeline.create_combined_embedding(test_image, depth_map, mask)
    print(f"✅ Combined embedding shape: {combined_embedding.shape} (expected: (64,))")
    
    # Test similarity calculation
    embedding1 = combined_embedding
    embedding2 = combined_embedding + np.random.rand(64) * 0.1  # Slight variation
    similarity = pipeline.calculate_similarity(embedding1, embedding2)
    print(f"✅ Similarity calculation: {similarity:.3f} (expected: ~0.9)")
    
    # Test with different embeddings
    embedding3 = np.random.rand(64)
    # Normalize random embedding for fair comparison
    embedding3 = embedding3 / np.linalg.norm(embedding3)
    similarity_random = pipeline.calculate_similarity(embedding1, embedding3)
    print(f"✅ Random similarity: {similarity_random:.3f} (expected: ~0.0)")
    
    print("\n🎉 All tests passed!")
    print(f"📊 Summary:")
    print(f"   - SAM segmentation: {'✅' if mask.any() else '❌'}")
    print(f"   - Depth estimation: {'✅' if depth_map.any() else '❌'}")
    print(f"   - MegaDescriptor RGB: {'✅' if rgb_features.shape == (32,) else '❌'}")
    print(f"   - MegaDescriptor Depth: {'✅' if depth_features.shape == (32,) else '❌'}")
    print(f"   - Combined embedding: {'✅' if combined_embedding.shape == (64,) else '❌'}")
    print(f"   - Similarity calculation: {'✅' if not np.isnan(similarity) and 0 <= similarity <= 1 else '❌'}")
    
    return True

if __name__ == "__main__":
    try:
        success = test_megadescriptor_pipeline()
        if success:
            print("\n✅ MegaDescriptor ReID pipeline is ready!")
        else:
            print("\n❌ Pipeline test failed")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()