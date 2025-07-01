"""
Adaptive Kalman Filter for optical flow smoothing.

This module contains the AdaptiveOpticalFlowKalmanFilter class which provides
adaptive Kalman filtering for optical flow with support for sudden motion changes.

WARNING: This module is NOT thread-safe. The AdaptiveOpticalFlowKalmanFilter 
maintains internal state that cannot be safely shared between threads without 
external synchronization.
"""

import cv2
import numpy as np
from scipy.interpolate import griddata


class AdaptiveOpticalFlowKalmanFilter:
    """
    Adaptive Kalman Filter for optical flow smoothing with support for sudden motion changes
    """
    def __init__(self, process_noise=0.01, measurement_noise=0.1, prediction_confidence=0.7, 
                 motion_model='constant_acceleration', outlier_threshold=3.0, min_track_length=3):
        self.base_process_noise = process_noise
        self.base_measurement_noise = measurement_noise
        self.prediction_confidence = prediction_confidence
        self.motion_model = motion_model
        self.outlier_threshold = outlier_threshold
        self.min_track_length = min_track_length
        
        # Adaptive parameters
        self.adaptation_factor = 2.0  # How much to increase noise during sudden changes
        self.motion_threshold = 5.0   # Threshold for detecting sudden motion changes
        self.adaptation_decay = 0.9   # How quickly to return to normal after adaptation
        
        # State tracking
        self.kalman_filters = {}      # Dict of Kalman filters per pixel
        self.motion_history = {}      # Motion history for adaptation
        self.frame_count = 0
        self.is_initialized = False
        
        # Statistics for monitoring
        self.adaptation_map = None    # Map of current adaptation levels
        self.outlier_count = 0
        self.total_pixels = 0
        
    def _create_kalman_filter(self, initial_position, initial_velocity):
        """Create a new Kalman filter for a pixel"""
        if self.motion_model == 'constant_velocity':
            # State: [x, y, vx, vy]
            state_size = 4
            measurement_size = 2
            
            # State transition matrix (constant velocity model)
            dt = 1.0
            transition_matrix = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
            
        else:  # constant_acceleration
            # State: [x, y, vx, vy, ax, ay]
            state_size = 6
            measurement_size = 2
            
            # State transition matrix (constant acceleration model)
            dt = 1.0
            dt2 = dt * dt / 2
            transition_matrix = np.array([
                [1, 0, dt, 0, dt2, 0],
                [0, 1, 0, dt, 0, dt2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ], dtype=np.float32)
        
        # Create Kalman filter
        kf = cv2.KalmanFilter(state_size, measurement_size)
        
        # Set matrices
        kf.transitionMatrix = transition_matrix
        
        # Measurement matrix (we observe position)
        kf.measurementMatrix = np.zeros((measurement_size, state_size), dtype=np.float32)
        kf.measurementMatrix[0, 0] = 1  # x
        kf.measurementMatrix[1, 1] = 1  # y
        
        # Process noise covariance
        kf.processNoiseCov = np.eye(state_size, dtype=np.float32) * self.base_process_noise
        
        # Measurement noise covariance
        kf.measurementNoiseCov = np.eye(measurement_size, dtype=np.float32) * self.base_measurement_noise
        
        # Error covariance
        kf.errorCovPost = np.eye(state_size, dtype=np.float32)
        
        # Initialize state
        if self.motion_model == 'constant_velocity':
            kf.statePre = np.array([initial_position[0], initial_position[1], 
                                   initial_velocity[0], initial_velocity[1]], dtype=np.float32)
        else:  # constant_acceleration
            kf.statePre = np.array([initial_position[0], initial_position[1], 
                                   initial_velocity[0], initial_velocity[1], 0, 0], dtype=np.float32)
        
        kf.statePost = kf.statePre.copy()
        
        return kf
    
    def _detect_sudden_motion(self, pixel_key, current_velocity, predicted_velocity):
        """Detect sudden changes in motion for adaptive filtering"""
        if pixel_key not in self.motion_history:
            self.motion_history[pixel_key] = {'velocities': [], 'adaptations': []}
            return False, 1.0
        
        history = self.motion_history[pixel_key]
        
        # Calculate velocity change
        velocity_change = np.linalg.norm(current_velocity - predicted_velocity)
        
        # Calculate recent average velocity change
        if len(history['velocities']) > 0:
            recent_changes = history['velocities'][-3:]  # Last 3 frames
            avg_change = np.mean(recent_changes)
            std_change = np.std(recent_changes) if len(recent_changes) > 1 else 1.0
            
            # Detect sudden motion if change is significantly larger than recent average
            threshold = avg_change + self.motion_threshold * max(std_change, 0.1)
            sudden_motion = velocity_change > threshold
            
            # Calculate adaptation factor
            if sudden_motion:
                adaptation = min(self.adaptation_factor, 1.0 + velocity_change / max(avg_change, 0.1))
            else:
                # Gradually return to normal
                last_adaptation = history['adaptations'][-1] if history['adaptations'] else 1.0
                adaptation = max(1.0, last_adaptation * self.adaptation_decay)
        else:
            sudden_motion = False
            adaptation = 1.0
        
        # Update history
        history['velocities'].append(velocity_change)
        history['adaptations'].append(adaptation)
        
        # Keep only recent history
        if len(history['velocities']) > 10:
            history['velocities'] = history['velocities'][-10:]
            history['adaptations'] = history['adaptations'][-10:]
        
        return sudden_motion, adaptation
    
    def _is_outlier(self, measurement, prediction, covariance):
        """Detect outlier measurements using Mahalanobis distance"""
        if covariance is None:
            return False
        
        diff = measurement - prediction
        
        # Calculate Mahalanobis distance
        try:
            inv_cov = np.linalg.inv(covariance + np.eye(2) * 1e-6)  # Add small regularization
            mahal_dist = np.sqrt(diff.T @ inv_cov @ diff)
            return mahal_dist > self.outlier_threshold
        except:
            # Fallback to Euclidean distance
            euclidean_dist = np.linalg.norm(diff)
            return euclidean_dist > self.outlier_threshold * 2
    
    def update(self, flow_field, frame_idx):
        """Update Kalman filters for entire flow field"""
        if flow_field is None:
            return flow_field
        
        h, w = flow_field.shape[:2]
        smoothed_flow = np.zeros_like(flow_field)
        
        # Initialize adaptation map
        if self.adaptation_map is None or self.adaptation_map.shape[:2] != (h, w):
            self.adaptation_map = np.ones((h, w), dtype=np.float32)
        
        self.frame_count += 1
        self.outlier_count = 0
        self.total_pixels = 0
        
        # Process pixels with subsampling for performance
        step = max(1, min(8, max(h, w) // 100))  # Adaptive step size
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                pixel_key = (y, x)
                current_flow = flow_field[y, x]
                current_position = np.array([x, y], dtype=np.float32)
                current_velocity = current_flow.astype(np.float32)
                
                self.total_pixels += 1
                
                # Skip zero flow during initialization
                if self.frame_count <= self.min_track_length and np.linalg.norm(current_velocity) < 0.1:
                    smoothed_flow[y, x] = current_flow
                    continue
                
                # Create new filter if needed
                if pixel_key not in self.kalman_filters:
                    self.kalman_filters[pixel_key] = self._create_kalman_filter(current_position, current_velocity)
                    smoothed_flow[y, x] = current_flow
                    continue
                
                kf = self.kalman_filters[pixel_key]
                
                # Predict
                predicted_state = kf.predict()
                predicted_position = predicted_state[:2]
                predicted_velocity = predicted_state[2:4] if self.motion_model == 'constant_velocity' else predicted_state[2:4]
                
                # Detect sudden motion and adapt
                sudden_motion, adaptation = self._detect_sudden_motion(pixel_key, current_velocity, predicted_velocity)
                self.adaptation_map[y, x] = adaptation
                
                # Adjust noise based on adaptation
                if adaptation > 1.1:  # Significant adaptation needed
                    # Temporarily increase process noise to allow faster adaptation
                    kf.processNoiseCov *= adaptation
                    kf.measurementNoiseCov /= np.sqrt(adaptation)  # Trust measurements more during sudden changes
                
                # Check for outliers
                measurement_position = current_position + current_velocity  # Where pixel moved to
                prediction_position = predicted_position + predicted_velocity
                
                if self._is_outlier(measurement_position, prediction_position, kf.errorCovPost[:2, :2]):
                    # Outlier detected - skip this measurement
                    self.outlier_count += 1
                    if self.motion_model == 'constant_velocity':
                        smoothed_velocity = predicted_state[2:4]
                    else:
                        smoothed_velocity = predicted_state[2:4]
                    smoothed_flow[y, x] = smoothed_velocity
                else:
                    # Normal measurement - update filter
                    measurement = current_position + current_velocity  # End position
                    kf.correct(measurement)
                    
                    # Get smoothed velocity from updated state
                    if self.motion_model == 'constant_velocity':
                        smoothed_velocity = kf.statePost[2:4]
                    else:
                        smoothed_velocity = kf.statePost[2:4]
                    
                    # Blend with original based on confidence
                    confidence = self.prediction_confidence * (2.0 - adaptation)  # Lower confidence during adaptation
                    confidence = np.clip(confidence, 0.1, 0.9)
                    
                    smoothed_flow[y, x] = (confidence * smoothed_velocity + 
                                         (1 - confidence) * current_velocity)
                
                # Restore normal noise levels
                if adaptation > 1.1:
                    kf.processNoiseCov /= adaptation
                    kf.measurementNoiseCov *= np.sqrt(adaptation)
        
        # Interpolate smoothed flow to full resolution
        if step > 1:
            smoothed_flow = self._interpolate_flow(smoothed_flow, flow_field, step)
        
        # Print statistics
        if self.frame_count % 30 == 0:  # Every 30 frames
            outlier_rate = self.outlier_count / max(self.total_pixels, 1) * 100
            avg_adaptation = np.mean(self.adaptation_map)
            print(f"Kalman Filter Stats - Frame {self.frame_count}: "
                  f"Outliers: {outlier_rate:.1f}%, Avg Adaptation: {avg_adaptation:.2f}")
        
        self.is_initialized = True
        return smoothed_flow
    
    def _interpolate_flow(self, sparse_flow, full_flow, step):
        """Interpolate sparse smoothed flow to full resolution"""
        h, w = full_flow.shape[:2]
        
        # Create coordinate grids
        y_sparse, x_sparse = np.mgrid[0:h:step, 0:w:step]
        y_full, x_full = np.mgrid[0:h, 0:w]
        
        # Interpolate each flow component
        points = np.column_stack([y_sparse.ravel(), x_sparse.ravel()])
        
        # Interpolate flow_x
        values_x = sparse_flow[::step, ::step, 0].ravel()
        flow_x_interp = griddata(points, values_x, (y_full, x_full), method='linear', fill_value=0)
        
        # Interpolate flow_y  
        values_y = sparse_flow[::step, ::step, 1].ravel()
        flow_y_interp = griddata(points, values_y, (y_full, x_full), method='linear', fill_value=0)
        
        # Combine and blend with original
        interpolated = np.stack([flow_x_interp, flow_y_interp], axis=2)
        
        # Blend with original flow in areas without sparse coverage
        mask = np.zeros((h, w), dtype=np.float32)
        mask[::step, ::step] = 1
        mask = cv2.GaussianBlur(mask, (step*2+1, step*2+1), step/2)
        
        result = full_flow.copy()
        for c in range(2):
            result[:, :, c] = mask * interpolated[:, :, c] + (1 - mask) * full_flow[:, :, c]
        
        return result 