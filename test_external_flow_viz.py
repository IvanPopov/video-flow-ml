#!/usr/bin/env python3

import subprocess
import sys
import os

def test_external_flow_viz():
    """Test external flow visualization feature"""
    
    # Test files
    main_video = "VideoFlow/demo_input_images/frame_0001.png"
    flow_video = "VideoFlow/demo_input_images/frame_0001_flow_cache_seq5_start0_frames1_fast/flow_frame_000000.npz"
    
    # Check if test files exist
    if not os.path.exists(main_video):
        print(f"Main video file not found: {main_video}")
        return False
    
    if not os.path.exists(flow_video):
        print(f"Flow video file not found: {flow_video}")
        return False
    
    output_dir = "test_external_flow_viz"
    
    # Command to test
    cmd = [
        sys.executable, "flow_processor.py",
        "--input", main_video,
        "--output", output_dir,
        "--frames", "5",
        "--taa",
        "--flow-input", flow_video,
        "--device", "cpu"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ Test passed successfully!")
            print("Output:")
            print(result.stdout)
            return True
        else:
            print("✗ Test failed!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Test timed out!")
        return False
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = test_external_flow_viz()
    sys.exit(0 if success else 1) 