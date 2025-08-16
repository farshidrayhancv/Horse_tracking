#!/usr/bin/env python3
import os
import cv2
import numpy as np

def check_display():
    """Check if display is available"""
    try:
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imshow('test', test_img)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        return True
    except:
        return False

def main():
    print("DEBUG: Starting main()")
    
    # Check display FIRST before importing heavy libraries
    display_available = check_display()
    print(f"DEBUG: Display check completed: {display_available}")
    
    # Import libraries
    print("DEBUG: About to import heavy libraries")
    import supervision as sv
    from pathlib import Path
    from boxmot import DeepOcSort
    from inference import get_model
    print("DEBUG: Heavy libraries imported")
    
    # Setup Roboflow model
    print("DEBUG: Setting up Roboflow model")
    try:
        model = get_model(model_id="del_mar_pan_seg-mdxam/2", api_key="wFo5HAaMOWxmTBCubny1")
        print("DEBUG: Roboflow model loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load Roboflow model: {e}")
        return

    # Setup annotators
    print("DEBUG: Setting up annotators")
    mask_annotator = sv.MaskAnnotator(opacity=0.4)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    color_palette = sv.ColorPalette.DEFAULT
    print("DEBUG: Annotators created")

    # Setup tracker
    reid_weight_pth = Path("reid_test_weights/osnet_ibn_x1_0_imagenet.pth")
    reid_weight_pt = Path("reid_test_weights/osnet_ibn_x1_0_imagenet.pt")
    if reid_weight_pth.exists() and not reid_weight_pt.exists():
        import shutil
        shutil.copy2(reid_weight_pth, reid_weight_pt)

    print("About to create tracker...")
    tracker = DeepOcSort(reid_weights=reid_weight_pt, device=0, half=True)
    print("Tracker created successfully")

    print("Warming up tracker...")
    dummy_det = np.array([[100, 100, 200, 200, 0.9, 0]], dtype=np.float64)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _ = tracker.update(dummy_det, dummy_frame)
    print("Tracker warmed up!")

    # Load video
    cap = cv2.VideoCapture("/home/farshid/Downloads/del_mar_pan_poc_1.mp4")

    frame_num = 0
    print("DEBUG: Starting main loop")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_num += 1
        
        # Run live inference with Roboflow model
        try:
            # Convert BGR to RGB for Roboflow
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference with correct parameters
            results = model.infer(
                rgb_frame,
                confidence=0.4,  # Fixed: was 40, should be 0.4
                iou_threshold=0.5  # Fixed: was overlap=30
            )[0]  # Take first result
            
            # Use supervision's built-in conversion
            detections = sv.Detections.from_inference(results)
            
            if frame_num % 30 == 0:
                print(f"DEBUG: Frame {frame_num} - Found {len(detections)} detections")
                if len(detections) > 0:
                    print(f"DEBUG: Confidences: {detections.confidence}")
                    print(f"DEBUG: Bounding boxes: {detections.xyxy}")
                    
        except Exception as e:
            print(f"ERROR: Live inference failed on frame {frame_num}: {e}")
            detections = sv.Detections.empty()
        
        # Update tracker
        if len(detections) > 0:
            xyxy = np.array(detections.xyxy, dtype=np.float64)
            confidence = np.array(detections.confidence, dtype=np.float64)
            class_ids = np.zeros(len(detections), dtype=np.float64)
            dets_np = np.column_stack((xyxy, confidence, class_ids))
            
            tracks = tracker.update(dets_np, frame)
            
            if tracks is not None and len(tracks) > 0:
                detections = sv.Detections(
                    xyxy=tracks[:, :4],
                    confidence=tracks[:, 5],
                    class_id=tracks[:, 6].astype(np.int32),
                    tracker_id=tracks[:, 4].astype(np.int32)
                )
                
                if frame_num % 30 == 0:
                    print(f"DEBUG: Tracker assigned IDs: {detections.tracker_id}")
        
        # Annotate frame with unique colors and labels
        annotated = frame.copy()
        if len(detections) > 0:
            # Apply mask annotation first
            if hasattr(detections, 'mask') and detections.mask is not None:
                annotated = mask_annotator.annotate(annotated, detections)
            
            # Create unique colors for each tracker ID
            if hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
                # Generate unique colors for each tracker ID
                unique_colors = []
                labels = []
                for i, (conf, track_id) in enumerate(zip(detections.confidence, detections.tracker_id)):
                    # Get unique color based on tracker ID
                    color = color_palette.by_idx(track_id % len(color_palette.colors))
                    unique_colors.append(color)
                    # Create label with confidence and tracker ID
                    labels.append(f"ID:{track_id} {conf:.2f}")
                
                # Create box annotator with unique colors
                colored_box_annotator = sv.BoxAnnotator(color=sv.ColorPalette(unique_colors))
                annotated = colored_box_annotator.annotate(annotated, detections)
                
                # Add labels
                annotated = label_annotator.annotate(annotated, detections, labels=labels)
            else:
                # Fallback to regular box annotation if no tracker IDs
                annotated = box_annotator.annotate(annotated, detections)
                labels = [f"{conf:.2f}" for conf in detections.confidence]
                annotated = label_annotator.annotate(annotated, detections, labels=labels)
        
        if display_available:
            cv2.imshow('Roboflow Detections', annotated)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        else:
            if frame_num % 100 == 0:
                print(f"Frame {frame_num} - No display available")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()