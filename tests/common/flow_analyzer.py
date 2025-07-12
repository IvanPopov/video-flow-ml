"""
Optical Flow Analyzer for Ground Truth Comparison

Analyzes optical flow accuracy against mathematically precise ground truth data.
"""

import os
import numpy as np
import math
from typing import Dict, Any


class OpticalFlowAnalyzer:
    """Analyze optical flow accuracy against ground truth"""
    
    def __init__(self, ground_truth: Dict[str, Any]):
        """
        Initialize analyzer
        
        Args:
            ground_truth: Ground truth data from video generation
        """
        self.ground_truth = ground_truth
        self.width = ground_truth['video_info']['width']
        self.height = ground_truth['video_info']['height']
        
    def create_ball_mask(self, frame_idx: int, dilation: int = 2) -> np.ndarray:
        """
        Create mask for ball pixels in given frame
        
        Args:
            frame_idx: Frame index
            dilation: Additional pixels around ball to include
            
        Returns:
            Binary mask where ball pixels are True
        """
        mask = np.zeros((self.height, self.width), dtype=bool)
        
        frame_data = self.ground_truth['frames'][frame_idx]
        center_x, center_y = frame_data['ball_center']
        radius = frame_data['ball_radius'] + dilation
        
        # Create circular mask
        y, x = np.ogrid[:self.height, :self.width]
        mask_circle = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        return mask_circle
    
    def load_flow_data(self, cache_dir: str, frame_idx: int) -> np.ndarray:
        """
        Load optical flow data for given frame
        
        Args:
            cache_dir: Directory containing flow cache
            frame_idx: Frame index
            
        Returns:
            Flow array [H, W, 2] with [dx, dy] per pixel
        """
        flow_file = os.path.join(cache_dir, f"flow_frame_{frame_idx:06d}.npz")
        
        if not os.path.exists(flow_file):
            raise FileNotFoundError(f"Flow file not found: {flow_file}")
        
        data = np.load(flow_file)
        flow = data['flow']
        
        return flow
    
    def calculate_direction_error(self, predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Calculate angle difference between predicted and ground truth vectors
        
        Args:
            predicted: Predicted flow vector [x, y]
            ground_truth: Ground truth velocity vector [x, y]
            
        Returns:
            Angle difference in degrees [0, 180]
        """
        # Calculate angles
        pred_angle = math.atan2(predicted[1], predicted[0])
        gt_angle = math.atan2(ground_truth[1], ground_truth[0])
        
        # Calculate difference
        diff = abs(pred_angle - gt_angle)
        
        # Ensure we get the smaller angle (0 to π)
        if diff > math.pi:
            diff = 2 * math.pi - diff
        
        # Convert to degrees
        return math.degrees(diff)
    
    def analyze_flow_accuracy(self, cache_dir: str, model_name: str) -> Dict[str, Any]:
        """
        Analyze optical flow accuracy for all frames
        
        Args:
            cache_dir: Directory containing optical flow cache
            model_name: Name of the model being analyzed
            
        Returns:
            Dictionary with analysis results
        """
        print(f"\nAnalyzing flow accuracy for {model_name}...")
        
        total_frames = self.ground_truth['video_info']['total_frames']
        
        # Statistics collections
        velocity_errors = []
        direction_errors = []
        magnitude_errors = []
        ball_pixel_counts = []
        
        # Process each frame (skip first frame as it has no flow)
        for frame_idx in range(1, total_frames):
            try:
                # Load flow data
                flow = self.load_flow_data(cache_dir, frame_idx)
                
                # Get ground truth for this frame
                frame_data = self.ground_truth['frames'][frame_idx]
                gt_velocity = frame_data['ball_velocity']
                
                # Create ball mask
                ball_mask = self.create_ball_mask(frame_idx)
                
                if not np.any(ball_mask):
                    print(f"Warning: No ball pixels found in frame {frame_idx}")
                    continue
                
                # Extract flow vectors from ball region
                ball_flow = flow[ball_mask]  # [N, 2] where N is number of ball pixels
                
                # Calculate statistics
                mean_flow = np.mean(ball_flow, axis=0)
                std_flow = np.std(ball_flow, axis=0)
                
                # Compare with ground truth
                velocity_error = np.linalg.norm(mean_flow - gt_velocity)
                direction_error = self.calculate_direction_error(mean_flow, gt_velocity)
                magnitude_error = abs(np.linalg.norm(mean_flow) - np.linalg.norm(gt_velocity))
                
                velocity_errors.append(velocity_error)
                direction_errors.append(direction_error)
                magnitude_errors.append(magnitude_error)
                ball_pixel_counts.append(np.sum(ball_mask))
                
            except Exception as e:
                print(f"Error analyzing frame {frame_idx}: {e}")
                continue
        
        if not velocity_errors:
            return {
                'model_name': model_name,
                'error': 'No valid frames analyzed',
                'total_frames_analyzed': 0
            }
        
        # Calculate aggregate statistics
        velocity_errors = np.array(velocity_errors)
        direction_errors = np.array(direction_errors)
        magnitude_errors = np.array(magnitude_errors)
        
        # Calculate accuracy at different thresholds
        accuracy_1px = np.mean(velocity_errors < 1.0) * 100
        accuracy_2px = np.mean(velocity_errors < 2.0) * 100
        accuracy_5px = np.mean(velocity_errors < 5.0) * 100
        
        results = {
            'model_name': model_name,
            'total_frames_analyzed': len(velocity_errors),
            'mean_velocity_error': float(np.mean(velocity_errors)),
            'std_velocity_error': float(np.std(velocity_errors)),
            'mean_direction_error': float(np.mean(direction_errors)),
            'std_direction_error': float(np.std(direction_errors)),
            'mean_magnitude_error': float(np.mean(magnitude_errors)),
            'std_magnitude_error': float(np.std(magnitude_errors)),
            'mean_ball_pixels': float(np.mean(ball_pixel_counts)),
            'accuracy_threshold_1px': float(accuracy_1px),
            'accuracy_threshold_2px': float(accuracy_2px),
            'accuracy_threshold_5px': float(accuracy_5px),
            'detailed_errors': {
                'velocity_errors': velocity_errors.tolist(),
                'direction_errors': direction_errors.tolist(),
                'magnitude_errors': magnitude_errors.tolist()
            }
        }
        
        return results 