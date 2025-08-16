#!/usr/bin/env python3
import os
import cv2
import numpy as np
import json
import supervision as sv
from pathlib import Path
# from inference import get_model
from boxmot import DeepOcSort



# # CACHE OPTION - Set to False to use live model inference
# use_cache = True
# cache_file = "detection_cache/horse_9_del_mar_pan_seg_mdxam_1_0.7_2000.json"

# # Load cached detections if requested
# cached_detections = []
# if use_cache:
#     print(f"Loading cached detections from {cache_file}")
#     with open(cache_file, 'r') as f:
#         cache_data = json.load(f)
    
#     for frame_data in cache_data['detections']:
#         detection_info = frame_data['detections']
        
#         # Handle empty detections properly
#         if not detection_info['xyxy'] or len(detection_info['xyxy']) == 0:
#             detections = sv.Detections.empty()
#         else:
#             xyxy = np.array(detection_info['xyxy'], dtype=np.float32)
#             confidence = np.array(detection_info['confidence'], dtype=np.float32)
#             class_id = np.array(detection_info['class_id'], dtype=np.int32)
            
#             detections = sv.Detections(
#                 xyxy=xyxy,
#                 confidence=confidence,
#                 class_id=class_id
#             )
        
#         cached_detections.append(detections)
    
#     print(f"Loaded {len(cached_detections)} cached detection frames")
# else:
#     # Load model only if not using cache
#     model = get_model(model_id="del_mar_pan_seg-mdxam/1", api_key="wFo5HAaMOWxmTBCubny1")

# Load video
cap = cv2.VideoCapture("/home/farshid/Downloads/del_mar_pan_poc_1.mp4")

# # Setup annotators (your exact code)
# box_annotator = sv.BoxAnnotator()
# mask_annotator = sv.MaskAnnotator(opacity=0.4)
# label_annotator = sv.LabelAnnotator()
# edge_annotator = sv.EdgeAnnotator(color=sv.Color.GREEN, thickness=2)

# # REPLACED: Your ByteTracker with champion ReID tracker
# reid_weight_pth = Path("reid_test_weights/osnet_ibn_x1_0_imagenet.pth")
# reid_weight_pt = Path("reid_test_weights/osnet_ibn_x1_0_imagenet.pt")
# if reid_weight_pth.exists() and not reid_weight_pt.exists():
#     import shutil
#     shutil.copy2(reid_weight_pth, reid_weight_pt)


# print("About to create tracker...")
# tracker = DeepOcSort(reid_weights=reid_weight_pt, device=0, half=True)
# print("Tracker created successfully")


# print("Warming up tracker...")
# dummy_det = np.array([[100, 100, 200, 200, 0.9, 0]], dtype=np.float64)
# dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
# _ = tracker.update(dummy_det, dummy_frame)
# print("Tracker warmed up!")


# track_smoother = sv.DetectionsSmoother(length=10)
frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_num += 1
    
    # # Get detections - from cache or model inference
    # if use_cache:
    #     if frame_num < len(cached_detections):
    #         detections = cached_detections[frame_num]
    #     else:
    #         detections = sv.Detections.empty()
    # else:
    #     # Run inference (your exact code)
    #     results = model.infer(frame)[0]
    #     detections = sv.Detections.from_inference(results)
    
    # # REPLACED: Update with DeepOcSort instead of ByteTrack
    # if len(detections) > 0:
    #     xyxy = np.array(detections.xyxy, dtype=np.float64)
    #     confidence = np.array(detections.confidence, dtype=np.float64)
    #     class_ids = np.zeros(len(detections), dtype=np.float64)
    #     dets_np = np.column_stack((xyxy, confidence, class_ids))
        
    #     print(f"Frame {frame_num}: About to track {len(detections)} detections")
    #     tracks = tracker.update(dets_np, frame)
    #     print(f"Frame {frame_num}: Tracking complete")
        
    #     if tracks is not None and len(tracks) > 0:
    #         detections = sv.Detections(
    #             xyxy=tracks[:, :4],
    #             confidence=tracks[:, 5],
    #             class_id=tracks[:, 6].astype(np.int32),
    #             tracker_id=tracks[:, 4].astype(np.int32)
    #         )
    
    # # detections = track_smoother.update_with_detections(detections)
    
    # # Build labels (your exact code)
    # labels = []
    # if len(detections) > 0:
    #     print(detections)
    #     for i, (class_id, confidence) in enumerate(zip(detections.class_id, detections.confidence)):
    #         if not use_cache and hasattr(results, 'class_names') and class_id is not None and class_id < len(results.class_names):
    #             class_name = results.class_names[class_id]
    #         else:
    #             class_name = f"Class_{class_id}"
            
    #         # Add tracker ID to label
    #         track_id = detections.tracker_id[i] if hasattr(detections, 'tracker_id') and detections.tracker_id is not None else "N/A"
    #         labels.append(f"{class_name}: {confidence:.2f} ID:{track_id}")
    
    #     annotated = frame.copy()
    #     if detections.mask is not None:
    #         annotated = mask_annotator.annotate(annotated, detections)
    #     annotated = box_annotator.annotate(annotated, detections)
    #     annotated = label_annotator.annotate(annotated, detections, labels=labels)
        
        # print("Chekcing for keypoints...")
        # # Keypoint annotation (your exact code)
        # if hasattr(detections, "keypoints") and detections.keypoints is not None:
        #     try:
        #         key_points = sv.KeyPoints(
        #             xy=detections.keypoints[..., :2],
        #             confidence=detections.keypoints[..., 2] if detections.keypoints.shape[-1] > 2 else None
        #         )
        #         annotated = edge_annotator.annotate(scene=annotated, key_points=key_points)
        #     except Exception as e:
        #         print(f"[WARN] Could not annotate keypoints: {e}")
        
        # cv2.namedWindow('Roboflow Detections', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('Roboflow Detections', annotated)
        # cv2.namedWindow('Roboflow Detections', cv2.WINDOW_AUTOSIZE)
        
    cv2.imshow('Roboflow Detections', frame)  # Not annotated
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()