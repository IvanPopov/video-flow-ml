import cv2
import os
import argparse
from tqdm import tqdm

def get_video_info(video_path):
    """Get video information"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    return width, height, fps, total_frames

def calculate_16_9_crop(width, height):
    """Calculate crop parameters to achieve 16:9 aspect ratio"""
    target_ratio = 16.0 / 9.0
    current_ratio = width / height
    
    if abs(current_ratio - target_ratio) < 0.01:
        # Already 16:9, no cropping needed
        return 0, 0, width, height
    
    if current_ratio > target_ratio:
        # Video is wider than 16:9, crop width
        new_width = int(height * target_ratio)
        new_height = height
        x_offset = (width - new_width) // 2
        y_offset = 0
    else:
        # Video is taller than 16:9, crop height
        new_width = width
        new_height = int(width / target_ratio)
        x_offset = 0
        y_offset = (height - new_height) // 2
    
    return x_offset, y_offset, new_width, new_height

def crop_to_16_9(input_path, output_path=None):
    """
    Crop video to 16:9 aspect ratio
    
    Args:
        input_path: Path to input video
        output_path: Path to output video (optional, auto-generated if None)
    """
    # Get video info
    width, height, fps, total_frames = get_video_info(input_path)
    
    print(f"Input: {input_path}")
    print(f"Original size: {width}x{height}")
    print(f"Original aspect ratio: {width/height:.3f}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps:.2f}")
    
    # Calculate crop parameters
    x_offset, y_offset, new_width, new_height = calculate_16_9_crop(width, height)
    
    if x_offset == 0 and y_offset == 0 and new_width == width and new_height == height:
        print("Video is already 16:9, no cropping needed!")
        return
    
    print(f"New size: {new_width}x{new_height}")
    print(f"New aspect ratio: {new_width/new_height:.3f} (16:9 = {16/9:.3f})")
    print(f"Crop area: x={x_offset}, y={y_offset}, w={new_width}, h={new_height}")
    
    # Generate output filename if not provided
    if output_path is None:
        input_dir = os.path.dirname(input_path)
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        input_ext = os.path.splitext(input_path)[1]
        output_path = os.path.join(input_dir, f"{input_name}_16x9{input_ext}")
    
    print(f"Output: {output_path}")
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    if not out.isOpened():
        raise ValueError(f"Cannot create output video: {output_path}")
    
    # Process frames
    pbar = tqdm(total=total_frames, desc="Cropping to 16:9", unit="frame")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop frame
        cropped_frame = frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width]
        
        # Write frame
        out.write(cropped_frame)
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f"✓ Video cropped to 16:9 successfully!")
    print(f"✓ Processed {frame_count} frames")
    print(f"✓ Output saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Crop video to 16:9 aspect ratio')
    parser.add_argument('--input', required=True, help='Input video file path')
    parser.add_argument('--output', help='Output video file path (optional, auto-generated if not specified)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
        return
    
    try:
        crop_to_16_9(args.input, args.output)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 