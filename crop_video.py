import cv2
import os
import argparse
from tqdm import tqdm

def crop_video(input_path, crop_fraction=0.75):
    """
    Crop video by removing a fraction of width from the right side
    
    Args:
        input_path: Path to input video
        crop_fraction: Fraction of width to remove (0.25 = remove 1/4 from right)
    """
    # Generate output filename
    input_dir = os.path.dirname(input_path)
    input_name = os.path.splitext(os.path.basename(input_path))[0]
    input_ext = os.path.splitext(input_path)[1]
    
    crop_percent = int(crop_fraction * 100)
    output_name = f"{input_name}_crop{crop_percent}{input_ext}"
    output_path = os.path.join(input_dir, output_name)
    
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Cropping: {crop_percent}% from right side")
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate new width after cropping
    new_width = int(width * (1 - crop_fraction))
    
    print(f"Original size: {width}x{height}")
    print(f"New size: {new_width}x{height}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps:.2f}")
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use XVID codec for AVI
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, height))
    
    if not out.isOpened():
        raise ValueError(f"Cannot create output video: {output_path}")
    
    # Process frames
    pbar = tqdm(total=total_frames, desc="Cropping video", unit="frame")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop frame: remove right portion
        cropped_frame = frame[:, :new_width]
        
        # Write frame
        out.write(cropped_frame)
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f"✓ Video cropped successfully!")
    print(f"✓ Processed {frame_count} frames")
    print(f"✓ Output saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Crop video by removing width from right side')
    parser.add_argument('--input', required=True, help='Input video file path')
    parser.add_argument('--crop', type=float, default=0.25, 
                       help='Fraction of width to crop from right (default: 0.25 = 1/4)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
        return
    
    try:
        crop_video(args.input, args.crop)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 