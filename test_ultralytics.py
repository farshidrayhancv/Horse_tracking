import json
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import supervision as sv
from pathlib import Path
import gzip
import base64
from typing import Optional

def compress_mask(mask: np.ndarray) -> str:
    if mask is None or mask.size == 0:
        return ""
    mask_uint8 = (mask * 255).astype(np.uint8) if mask.dtype != np.uint8 else mask
    compressed = gzip.compress(mask_uint8.tobytes())
    return base64.b64encode(compressed).decode('utf-8')

def decompress_mask(compressed_str: str, shape: tuple) -> Optional[np.ndarray]:
    if not compressed_str:
        return None
    try:
        compressed = base64.b64decode(compressed_str.encode('utf-8'))
        decompressed = gzip.decompress(compressed)
        mask = np.frombuffer(decompressed, dtype=np.uint8).reshape(shape)
        return mask > 127
    except Exception as e:
        print(f"Error decompressing mask: {e}")
        return None

def process_video_with_cache(video_path, output_path, cache_path, model_path='yolo11s_seg_best.pt', 
                           confidence=0.8, overwrite_cache=False, resize_image_square=False,
                           show_masks=True, show_boxes=True, show_labels=True):
    if Path(cache_path).exists() and not overwrite_cache:
        print(f"Loading from cache: {cache_path}")
        return create_video_from_cache(video_path, output_path, cache_path, resize_image_square,
                                     show_masks, show_boxes, show_labels)
    if overwrite_cache and Path(cache_path).exists():
        print(f"Overwriting existing cache: {cache_path}")
    else:
        print(f"No cache found. Running inference...")
    return process_and_cache(video_path, output_path, cache_path, model_path, confidence, resize_image_square,
                           show_masks, show_boxes, show_labels)

def process_and_cache(video_path, output_path, cache_path, model_path, confidence, resize_image_square,
                     show_masks, show_boxes, show_labels):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    video_info = sv.VideoInfo.from_video_path(video_path)
    if resize_image_square:
        output_size = (1280, 1280)
    else:
        output_size = (video_info.width, video_info.height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_info.fps, output_size)
    mask_annotator = sv.MaskAnnotator(opacity=0.8) if show_masks else None
    box_annotator = sv.BoundingBoxAnnotator() if show_boxes else None
    label_annotator = sv.LabelAnnotator() if show_labels else None
    cache_data = []
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if resize_image_square:
            input_frame = cv2.resize(frame, (1280, 1280))
        else:
            input_frame = frame
        results = model.predict(input_frame, conf=confidence, retina_masks=True)[0]
        detections = sv.Detections.from_ultralytics(results)
        annotated = input_frame.copy()
        if show_masks and mask_annotator:
            annotated = mask_annotator.annotate(annotated, detections)
        if show_boxes and box_annotator:
            annotated = box_annotator.annotate(annotated, detections)
        if show_labels and label_annotator:
            labels = [f"{confidence:.2f}" for confidence in detections.confidence]
            annotated = label_annotator.annotate(annotated, detections, labels=labels)
        out.write(annotated)
        compressed_masks = []
        mask_shape = []
        if detections.mask is not None and len(detections.mask) > 0:
            mask_shape = list(detections.mask[0].shape)
            for mask in detections.mask:
                compressed_masks.append(compress_mask(mask))
        frame_data = {
            'frame_num': frame_num,
            'detections': {
                'xyxy': detections.xyxy.tolist(),
                'confidence': detections.confidence.tolist(),
                'class_id': detections.class_id.tolist() if detections.class_id is not None else [],
                'masks_compressed': compressed_masks,
                'mask_shape': mask_shape
            }
        }
        cache_data.append(frame_data)
        frame_num += 1
        if frame_num % 100 == 0:
            print(f"Processed {frame_num} frames")
    cap.release()
    out.release()
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f)
    print(f"✅ Processed {frame_num} frames and saved cache to {cache_path}")
    return frame_num

def create_video_from_cache(video_path, output_path, cache_path, resize_image_square,
                          show_masks, show_boxes, show_labels):
    with open(cache_path, 'r') as f:
        cache_data = json.load(f)
    cap = cv2.VideoCapture(video_path)
    video_info = sv.VideoInfo.from_video_path(video_path)
    if resize_image_square:
        output_size = (1280, 1280)
    else:
        output_size = (video_info.width, video_info.height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_info.fps, output_size)
    mask_annotator = sv.MaskAnnotator(opacity=0.4) if show_masks else None
    box_annotator = sv.BoundingBoxAnnotator() if show_boxes else None
    label_annotator = sv.LabelAnnotator() if show_labels else None
    frame_num = 0
    while cap.isOpened() and frame_num < len(cache_data):
        ret, frame = cap.read()
        if not ret:
            break
        frame_data = cache_data[frame_num]
        det_data = frame_data['detections']
        masks = None
        if det_data['masks_compressed'] and det_data['mask_shape']:
            mask_shape = tuple(det_data['mask_shape'])
            masks = []
            for compressed_mask in det_data['masks_compressed']:
                mask = decompress_mask(compressed_mask, mask_shape)
                if mask is not None:
                    masks.append(mask)
            masks = np.array(masks) if masks else None
        xyxy = np.array(det_data['xyxy'])
        if xyxy.size == 0:
            xyxy = np.empty((0, 4))
        detections = sv.Detections(
            xyxy=xyxy,
            confidence=np.array(det_data['confidence']),
            class_id=np.array(det_data['class_id']) if det_data['class_id'] else None,
            mask=masks
        )
        if resize_image_square:
            input_frame = cv2.resize(frame, (1280, 1280))
        else:
            input_frame = frame
        annotated = input_frame.copy()
        if show_masks and mask_annotator:
            annotated = mask_annotator.annotate(annotated, detections)
        if show_boxes and box_annotator:
            annotated = box_annotator.annotate(annotated, detections)
        if show_labels and label_annotator:
            labels = [f"{confidence:.2f}" for confidence in detections.confidence]
            annotated = label_annotator.annotate(annotated, detections, labels=labels)
        out.write(annotated)
        frame_num += 1
        if frame_num % 100 == 0:
            print(f"Processed {frame_num} frames from cache")
    cap.release()
    out.release()
    print(f"✅ Created video from cache: {output_path}")
    return frame_num

if __name__ == "__main__":
    video_path = "/home/farshid/proj/Horse_tracking/inputs/horse_9.mp4"
    output_path = "/home/farshid/proj/Horse_tracking/inputs/output_video.mp4" 
    cache_path = "/home/farshid/proj/Horse_tracking/detection_cache/del_mar_pan_poc_1__gate_dets.json"
    # model_path = "/home/farshid/Downloads/runs/segment/train_1280_6/weights/best.pt"
    model_path = "/home/farshid/proj/Horse_tracking/inputs/yolo11n_gate_det_obj_det_v4_conf_0.7.pt"
    

    confidence = 0.7
    overwrite_cache = False
    resize_image_square = True
    show_masks = False
    show_boxes = True
    show_labels = True
    process_video_with_cache(video_path, output_path, cache_path, model_path, 
                           confidence, overwrite_cache, resize_image_square,
                           show_masks, show_boxes, show_labels)