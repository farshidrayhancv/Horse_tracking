#!/usr/bin/env python3
"""
Analyze horse movement in your video to determine optimal motion_distance_threshold
WITH options for full video analysis and side-by-side comparison video
"""

import cv2
import numpy as np
from pathlib import Path
import argparse

def analyze_horse_movement(video_path: str, sample_frames: int = 50, full_video: bool = False, 
                          save_comparison: bool = False, output_path: str = None):
    """Analyze typical horse movement distances in video"""
    print(f"🔍 Analyzing horse movement in {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 Video info: {total_frames} frames, {fps:.1f} FPS, {width}x{height}")
    
    # Determine frame processing strategy
    if full_video:
        print("🎬 Processing FULL VIDEO (may take a while...)")
        frame_indices = list(range(100, total_frames-100, 1))  # Skip every N frames for large videos
        print(f"   Processing {len(frame_indices)} frames from full video")
    else:
        frame_indices = np.linspace(100, total_frames-100, sample_frames).astype(int)
        print(f"   Processing {len(frame_indices)} sample frames")
    
    # Setup video writer for comparison if requested
    video_writer = None
    if save_comparison and output_path:
        output_path = Path(output_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Side-by-side video will be twice the width
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width * 2, height))
        print(f"📹 Saving comparison video to: {output_path}")
    
    # Analysis variables
    prev_frame = None
    movement_magnitudes = []
    frame_movement_data = []  # For video annotation
    
    # Processing loop
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Resize for processing (4K is too large)
        frame_small = cv2.resize(frame, (960, 540))  # Quarter resolution
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        
        # Current frame movements
        current_movements = []
        annotated_frame = frame.copy()
        
        if prev_frame is not None:
            # Calculate frame difference
            diff = cv2.absdiff(prev_frame, gray)
            
            # Find regions of significant change
            _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
            
            # Find contours of moving objects
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 10000:  # Filter for horse-sized objects
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Estimate movement (simple heuristic)
                    movement = max(w, h) * 0.3  # Conservative estimate
                    
                    # Scale back to original resolution
                    movement_original = movement * 4  # We downscaled by 4x
                    movement_magnitudes.append(movement_original)
                    current_movements.append(movement_original)
                    
                    # Annotate original frame if saving comparison
                    if save_comparison:
                        # Scale bounding box back to original resolution
                        x_orig, y_orig = x * 4, y * 4
                        w_orig, h_orig = w * 4, h * 4
                        
                        # Draw bounding box
                        color = (0, 255, 0) if movement_original < 50 else (0, 255, 255) if movement_original < 100 else (0, 0, 255)
                        cv2.rectangle(annotated_frame, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), color, 3)
                        
                        # Add movement text
                        cv2.putText(annotated_frame, f"{movement_original:.0f}px", 
                                   (x_orig, y_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Store frame data for video
        frame_movement_data.append({
            'frame_idx': frame_idx,
            'movements': current_movements,
            'avg_movement': np.mean(current_movements) if current_movements else 0
        })
        
        # Save comparison frame if requested
        if save_comparison and video_writer:
            # Add overall statistics to frames
            avg_movement = np.mean(current_movements) if current_movements else 0
            max_movement = max(current_movements) if current_movements else 0
            
            # Add info overlay to original frame
            cv2.putText(frame, f"Frame: {frame_idx}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(frame, f"Avg Movement: {avg_movement:.1f}px", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(frame, f"Max Movement: {max_movement:.1f}px", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            # Add info overlay to annotated frame
            cv2.putText(annotated_frame, f"Movement Analysis", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(annotated_frame, f"Green: <50px, Yellow: 50-100px, Red: >100px", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Detected Objects: {len(current_movements)}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            # Create side-by-side frame
            side_by_side = np.hstack([frame, annotated_frame])
            video_writer.write(side_by_side)
        
        prev_frame = gray
        
        # Progress updates
        if full_video:
            if i % max(1, len(frame_indices)//10) == 0:
                progress = (i / len(frame_indices)) * 100
                print(f"  Progress: {progress:.1f}% ({i}/{len(frame_indices)} frames)")
        else:
            if i % 10 == 0:
                print(f"  Processed {i}/{len(frame_indices)} frames...")
    
    cap.release()
    if video_writer:
        video_writer.release()
        print(f"✅ Comparison video saved to: {output_path}")
    
    # Analysis results
    if movement_magnitudes:
        movements = np.array(movement_magnitudes)
        
        print(f"\n📈 Movement Analysis Results:")
        print(f"  Total movement samples: {len(movements)}")
        print(f"  Average movement: {np.mean(movements):.1f} pixels/frame")
        print(f"  Median movement: {np.median(movements):.1f} pixels/frame")
        print(f"  75th percentile: {np.percentile(movements, 75):.1f} pixels/frame")
        print(f"  90th percentile: {np.percentile(movements, 90):.1f} pixels/frame")
        print(f"  95th percentile: {np.percentile(movements, 95):.1f} pixels/frame")
        print(f"  99th percentile: {np.percentile(movements, 99):.1f} pixels/frame")
        print(f"  Max movement: {np.max(movements):.1f} pixels/frame")
        
        # Recommendations
        conservative = np.percentile(movements, 75)
        moderate = np.percentile(movements, 90) 
        permissive = np.percentile(movements, 95)
        
        print(f"\n💡 Recommended motion_distance_threshold values:")
        print(f"  Conservative (strict): {conservative:.0f} pixels")
        print(f"  Moderate (balanced): {moderate:.0f} pixels") 
        print(f"  Permissive (loose): {permissive:.0f} pixels")
        
        print(f"\n🎯 For your excessive reassignment problem, try:")
        print(f"  motion_distance_threshold: {conservative:.0f}  # Start here")
        
        # Frame size context
        frame_percent = (moderate / width) * 100
        print(f"\n📏 Context: {moderate:.0f} pixels = {frame_percent:.1f}% of frame width")
        
        # Frame-by-frame analysis summary
        if full_video:
            frame_avgs = [fd['avg_movement'] for fd in frame_movement_data if fd['avg_movement'] > 0]
            if frame_avgs:
                print(f"\n📊 Frame-by-frame analysis:")
                print(f"  Frames with movement: {len(frame_avgs)}/{len(frame_movement_data)}")
                print(f"  Avg movement per moving frame: {np.mean(frame_avgs):.1f} pixels")
                print(f"  Most active frame: {np.max(frame_avgs):.1f} pixels")
        
        return {
            'conservative': conservative,
            'moderate': moderate,
            'permissive': permissive,
            'mean': np.mean(movements),
            'max': np.max(movements),
            'frame_data': frame_movement_data
        }
    else:
        print("❌ No movement data collected")
        return None

def estimate_from_video_properties(video_path: str):
    """Quick estimation based on video resolution"""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()
    
    print(f"📐 Quick estimates based on video properties:")
    print(f"  Video: {width}x{height} @ {fps:.1f} FPS")
    
    # Rule of thumb estimates for racing footage
    if width >= 3840:  # 4K
        conservative = 60
        moderate = 100  
        permissive = 150
    elif width >= 1920:  # HD
        conservative = 30
        moderate = 50
        permissive = 80
    else:  # SD
        conservative = 15
        moderate = 25
        permissive = 40
    
    # Adjust for frame rate
    fps_factor = fps / 25.0  # Normalize to 25 FPS
    conservative *= fps_factor
    moderate *= fps_factor
    permissive *= fps_factor
    
    print(f"  Conservative: {conservative:.0f} pixels")
    print(f"  Moderate: {moderate:.0f} pixels")
    print(f"  Permissive: {permissive:.0f} pixels")
    
    print(f"\n🎯 For 4K racing footage with excessive reassignments:")
    print(f"  motion_distance_threshold: {conservative:.0f}")
    
    return conservative, moderate, permissive

def main():
    parser = argparse.ArgumentParser(description='Analyze horse movement for motion threshold tuning')
    parser.add_argument('video_path', help='Path to input video')
    parser.add_argument('--quick', action='store_true', help='Skip analysis, use resolution-based estimate')
    parser.add_argument('--full-video', action='store_true', help='Process entire video instead of sampling')
    parser.add_argument('--save-comparison', action='store_true', help='Save side-by-side comparison video')
    parser.add_argument('--output', '-o', help='Output path for comparison video (default: auto-generated)')
    parser.add_argument('--sample-frames', type=int, default=50, help='Number of frames to sample (default: 50)')
    
    args = parser.parse_args()
    
    if not Path(args.video_path).exists():
        print(f"❌ Video not found: {args.video_path}")
        return
    
    print("🐴 Horse Movement Analysis for Motion Threshold")
    print("=" * 50)
    
    # Generate output path if saving comparison
    output_path = None
    if args.save_comparison:
        if args.output:
            output_path = args.output
        else:
            input_path = Path(args.video_path)
            output_path = input_path.parent / f"{input_path.stem}_movement_analysis.mp4"
    
    if args.quick:
        conservative, moderate, permissive = estimate_from_video_properties(args.video_path)
        print(f"\n📝 Update your config.yaml:")
        print(f"motion_distance_threshold: {conservative:.0f}")
    else:
        mode = "full video" if args.full_video else f"{args.sample_frames} sample frames"
        print(f"🔍 Running detailed movement analysis on {mode}...")
        
        result = analyze_horse_movement(
            args.video_path, 
            sample_frames=args.sample_frames,
            full_video=args.full_video,
            save_comparison=args.save_comparison,
            output_path=output_path
        )
        
        if result:
            print(f"\n📝 Update your config.yaml:")
            print(f"motion_distance_threshold: {result['conservative']:.0f}")
            
            if args.save_comparison:
                print(f"\n🎥 Watch the comparison video to see movement patterns:")
                print(f"   {output_path}")
                print(f"   Left side: Original video")
                print(f"   Right side: Movement detection (Green <50px, Yellow 50-100px, Red >100px)")
        else:
            print("\n⚠️ Analysis failed, using quick estimate...")
            conservative, moderate, permissive = estimate_from_video_properties(args.video_path)
            print(f"\n📝 Update your config.yaml:")
            print(f"motion_distance_threshold: {conservative:.0f}")

if __name__ == "__main__":
    main()