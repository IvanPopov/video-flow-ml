#!/usr/bin/env python3
"""
Interactive Optical Flow Visualizer

This script provides an interactive GUI for visualizing optical flow data:
- Shows two consecutive video frames vertically
- Slider to navigate through frame pairs
- Mouse hover shows flow vector as arrow pointing to next frame pixel
- Supports both .flo and .npz flow formats
"""

import os
import sys
import argparse
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import glob
from pathlib import Path
import threading
import queue
from scipy import signal
from scipy.ndimage import rotate

# Add flow_processor to path for loading flow data
sys.path.insert(0, os.getcwd())
from flow_processor import VideoFlowProcessor

class FlowVisualizer:
    def __init__(self, video_path, flow_dir, start_frame=0, max_frames=None):
        self.video_path = video_path
        self.flow_dir = flow_dir
        self.start_frame = start_frame
        self.max_frames = max_frames
        self.processor = VideoFlowProcessor()
        
        # Load video frames
        self.frames = self.load_video_frames()
        self.flow_files = self.find_flow_files()
        
        if len(self.frames) == 0:
            raise ValueError("No frames loaded from video")
        if len(self.flow_files) == 0:
            raise ValueError("No flow files found in directory")
            
        print(f"Loaded {len(self.frames)} frames and {len(self.flow_files)} flow files")
        
        # Current state
        self.current_pair = 0
        self.max_pairs = min(len(self.frames) - 1, len(self.flow_files))
        
        # Initialize quality maps storage
        self.quality_maps = {}
        self.quality_map_queue = queue.Queue()
        self.computing_quality = set()  # Track which frames are being computed
        self.computation_threads = {}  # Track active threads
        self.stop_computation = threading.Event()  # Signal to stop computations
        self.slider_dragging = False  # Track if slider is being dragged
        
        # LOD support
        self.lod_data = {}  # Cache for LOD data
        self.current_lod_level = 0  # Current LOD level
        self.use_lod_for_vectors = False  # Whether to use LOD for flow vectors
        self.max_lod_levels = 5  # Maximum number of LOD levels
        
        # UI state
        self.show_quality_map = True  # Whether to show quality map
        
        # Detail analysis state
        self.detail_analysis_mode = False
        self.detail_analysis_data = None
        self.detail_analysis_region_size = 25
        
        # Check for LOD data availability
        self.check_lod_availability()
        
        # Pre-compute only the first quality map
        print("Computing initial quality map...")
        if self.max_pairs > 0:
            frame1 = self.frames[0]
            frame2 = self.frames[1]
            flow = self.load_flow_data(0)
            self.quality_maps[0] = self.generate_quality_frame_fast(frame1, frame2, flow)
            print("Initial quality map ready!")
        
        # Zoom state
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        self.zoom_step = 0.1
        
        # Pan state for dragging
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.is_panning = False
        self.last_pan_x = 0
        self.last_pan_y = 0
        
        # UI setup
        self.setup_ui()
        
        # Start background quality computation checker
        self.check_quality_queue()
        
        self.update_display()
        
    def load_video_frames(self):
        """Load frames from video starting at start_frame"""
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Check if we've reached max frames
            if self.max_frames is not None and frame_count >= self.max_frames:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_count += 1
            
        cap.release()
        return frames
    
    def find_flow_files(self):
        """Find all flow files in directory"""
        flow_files = []
        
        # Look for .flo files
        flo_pattern = os.path.join(self.flow_dir, "*.flo")
        flo_files = sorted(glob.glob(flo_pattern))
        
        # Look for .npz files if no .flo found
        if not flo_files:
            npz_pattern = os.path.join(self.flow_dir, "*.npz")
            npz_files = sorted(glob.glob(npz_pattern))
            flow_files = npz_files
        else:
            flow_files = flo_files
            
        return flow_files
    
    def check_lod_availability(self):
        """Check if LOD data is available in the flow directory"""
        lod_dir = os.path.join(self.flow_dir, 'lods')
        if os.path.exists(lod_dir):
            # Check if we have LOD 0 for the first frame to validate availability
            lod0_file = os.path.join(lod_dir, f"flow_frame_{0:06d}_lod0.npz")
            if os.path.exists(lod0_file):
                print("LOD data found! LOD-based vector rendering available.")
                return True
        print("LOD data not available. Using original flow data only.")
        return False
    
    def load_lod_data(self, frame_idx, lod_level):
        """Load LOD data for specific frame and level"""
        cache_key = (frame_idx, lod_level)
        if cache_key in self.lod_data:
            return self.lod_data[cache_key]
        
        lod_dir = os.path.join(self.flow_dir, 'lods')
        lod_file = os.path.join(lod_dir, f"flow_frame_{frame_idx:06d}_lod{lod_level}.npz")
        
        if os.path.exists(lod_file):
            try:
                npz_data = self.processor.load_flow_npz(lod_file)
                lod_flow = npz_data['flow']
                self.lod_data[cache_key] = lod_flow
                return lod_flow
            except Exception as e:
                print(f"Error loading LOD {lod_level} for frame {frame_idx}: {e}")
                return None
        return None
    
    def load_flow_data(self, frame_idx):
        """Load flow data for specific frame"""
        if frame_idx >= len(self.flow_files):
            return None
            
        flow_file = self.flow_files[frame_idx]
        
        try:
            if flow_file.endswith('.flo'):
                return self.processor.load_flow_flo(flow_file)
            elif flow_file.endswith('.npz'):
                npz_data = self.processor.load_flow_npz(flow_file)
                return npz_data['flow']
        except Exception as e:
            print(f"Error loading flow file {flow_file}: {e}")
            return None
    
    def compute_quality_map_background(self, frame_idx):
        """Compute quality map for a specific frame in background thread"""
        def worker():
            try:
                # Check if computation should be stopped
                if self.stop_computation.is_set():
                    return
                
                # Load frames and flow
                frame1 = self.frames[frame_idx]
                frame2 = self.frames[frame_idx + 1]
                flow = self.load_flow_data(frame_idx)
                
                # Check again before heavy computation
                if self.stop_computation.is_set():
                    return
                
                # Generate quality map
                quality_map = self.generate_quality_frame_fast(frame1, frame2, flow)
                
                # Check if result is still needed
                if not self.stop_computation.is_set():
                    # Put result in queue
                    self.quality_map_queue.put((frame_idx, quality_map))
                
            except Exception as e:
                if not self.stop_computation.is_set():
                    print(f"Error computing quality map for frame {frame_idx}: {e}")
                    # Put error result
                    self.quality_map_queue.put((frame_idx, None))
            finally:
                # Remove from computing set and thread tracking
                self.computing_quality.discard(frame_idx)
                self.computation_threads.pop(frame_idx, None)
        
        # Start background thread
        thread = threading.Thread(target=worker, daemon=True)
        self.computation_threads[frame_idx] = thread
        thread.start()
    
    def check_quality_queue(self):
        """Check for completed quality map computations"""
        try:
            while True:
                frame_idx, quality_map = self.quality_map_queue.get_nowait()
                if quality_map is not None:
                    self.quality_maps[frame_idx] = quality_map
                    # Update display if this is the current frame
                    if frame_idx == self.current_pair:
                        self.update_display()
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.check_quality_queue)
    
    def stop_all_computations(self):
        """Stop all background computations"""
        self.stop_computation.set()
        # Clear computing set
        self.computing_quality.clear()
        # Clear thread tracking
        self.computation_threads.clear()
        # Clear queue
        while not self.quality_map_queue.empty():
            try:
                self.quality_map_queue.get_nowait()
            except queue.Empty:
                break
    
    def resume_computations(self):
        """Resume background computations"""
        self.stop_computation.clear()
    
    def request_quality_map(self, frame_idx):
        """Request quality map for a frame (compute if not available)"""
        if (frame_idx not in self.quality_maps and 
            frame_idx not in self.computing_quality and 
            not self.stop_computation.is_set() and
            not self.slider_dragging):
            self.computing_quality.add(frame_idx)
            self.compute_quality_map_background(frame_idx)
    
    def preload_adjacent_frames(self):
        """Preload quality maps for adjacent frames"""
        # Preload previous and next frames
        for offset in [-1, 1]:
            frame_idx = self.current_pair + offset
            if 0 <= frame_idx < self.max_pairs:
                self.request_quality_map(frame_idx)
    
    def phase_correlation_with_rotation(self, img1, img2):
        """
        Compute phase correlation between two images to find translation.
        Rotation is not calculated and is returned as 0.
        Returns: (dx, dy, angle, confidence)
        """
        # Ensure images are grayscale and float32
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)

        # Use phase correlation to find the translation
        (dx, dy), confidence = cv2.phaseCorrelate(img1, img2)

        # Angle is not calculated, return 0 as per requirement
        angle = 0.0

        return dx, dy, angle, confidence
    
    def get_highest_available_lod(self, frame_idx):
        """Get the highest available LOD level for a frame"""
        for lod_level in range(self.max_lod_levels - 1, -1, -1):
            lod_data = self.load_lod_data(frame_idx, lod_level)
            if lod_data is not None:
                return lod_level, lod_data
        return None, None
    
    def extract_region(self, image, center_x, center_y, radius):
        """Extract a square region around a center point"""
        h, w = image.shape[:2]
        
        # Calculate bounds
        x1 = max(0, int(center_x - radius))
        y1 = max(0, int(center_y - radius))
        x2 = min(w, int(center_x + radius))
        y2 = min(h, int(center_y + radius))
        
        # Extract region
        region = image[y1:y2, x1:x2]
        
        # Pad if necessary to make square
        target_size = 2 * radius
        if region.shape[0] < target_size or region.shape[1] < target_size:
            pad_h = max(0, target_size - region.shape[0])
            pad_w = max(0, target_size - region.shape[1])
            if len(image.shape) == 3:
                region = np.pad(region, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            else:
                region = np.pad(region, ((0, pad_h), (0, pad_w)), mode='constant')
        
        return region, (x1, y1, x2, y2)
    
    def on_left_click(self, event):
        """Handle left mouse click for detail analysis"""
        if self.current_flow is None:
            return
            
        # Check if click is on first frame
        if (self.frame1_x <= event.x <= self.frame1_x + self.display_width and
            self.frame1_y <= event.y <= self.frame1_y + self.display_height):
            
            # Convert display coordinates to original frame coordinates
            display_x = event.x - self.frame1_x
            display_y = event.y - self.frame1_y
            
            orig_x = int(display_x / self.display_scale)
            orig_y = int(display_y / self.display_scale)
            
            # Clamp to frame bounds
            orig_x = max(0, min(orig_x, self.orig_width - 1))
            orig_y = max(0, min(orig_y, self.orig_height - 1))
            
            # Check if this is a different pixel while in detail mode
            if (self.detail_analysis_mode and 
                self.detail_analysis_data and
                (orig_x, orig_y) != self.detail_analysis_data['source_pixel']):
                
                if not self.check_exit_detail_mode("Клик на другой пиксель"):
                    return
            
            # Check if this pixel has poor flow quality
            if self.current_flow is not None:
                # Get target position using current flow
                fh, fw = self.current_flow.shape[:2]
                scale_x = fw / self.orig_width
                scale_y = fh / self.orig_height
                
                flow_x_coord = int(orig_x * scale_x)
                flow_y_coord = int(orig_y * scale_y)
                flow_x_coord = max(0, min(flow_x_coord, fw - 1))
                flow_y_coord = max(0, min(flow_y_coord, fh - 1))
                
                flow_x = self.current_flow[flow_y_coord, flow_x_coord, 0] / scale_x
                flow_y = self.current_flow[flow_y_coord, flow_x_coord, 1] / scale_y
                
                target_orig_x = orig_x - flow_x
                target_orig_y = orig_y - flow_y
                
                # Check color quality
                arrow_color = self.get_arrow_color(orig_x, orig_y, target_orig_x, target_orig_y)
                
                if arrow_color == "red":
                    # Poor quality - start detail analysis, pass original flow vector
                    self.perform_detail_analysis(orig_x, orig_y, (flow_x, flow_y))
                else:
                    messagebox.showinfo("Информация", 
                        f"Пиксель ({orig_x}, {orig_y}) имеет хорошее качество потока. "
                        "Детальный анализ выполняется только для пикселей с плохим качеством (красные стрелки).")
    
    def perform_detail_analysis(self, orig_x, orig_y, original_flow):
        """Perform detailed analysis on a pixel with poor flow quality"""
        print(f"Starting detail analysis at pixel ({orig_x}, {orig_y})")
        
        # Get highest available LOD
        lod_level, lod_flow = self.get_highest_available_lod(self.current_pair)
        if lod_flow is None:
            messagebox.showerror("Error", "No LOD data available for detail analysis")
            return
        
        print(f"Using LOD level {lod_level}")
        
        # Get LOD flow vector
        lod_h, lod_w = lod_flow.shape[:2]
        lod_scale_x = lod_w / self.orig_width
        lod_scale_y = lod_h / self.orig_height
        
        lod_x = int(orig_x * lod_scale_x)
        lod_y = int(orig_y * lod_scale_y)
        lod_x = max(0, min(lod_x, lod_w - 1))
        lod_y = max(0, min(lod_y, lod_h - 1))
        
        # Get LOD flow vector and scale back to frame coordinates
        lod_flow_x = lod_flow[lod_y, lod_x, 0] / lod_scale_x
        lod_flow_y = lod_flow[lod_y, lod_x, 1] / lod_scale_y
        
        print(f"LOD flow vector: ({lod_flow_x:.2f}, {lod_flow_y:.2f})")
        
        # Calculate target position using LOD flow
        lod_target_x = orig_x - lod_flow_x
        lod_target_y = orig_y - lod_flow_y
        
        # Extract regions around source and LOD target
        region1, bounds1 = self.extract_region(self.frame1, orig_x, orig_y, self.detail_analysis_region_size)
        region2, bounds2 = self.extract_region(self.frame2, lod_target_x, lod_target_y, self.detail_analysis_region_size)
        
        # Perform phase correlation
        dx, dy, angle, confidence = self.phase_correlation_with_rotation(region1, region2)
        
        print(f"Phase correlation result: dx={dx:.2f}, dy={dy:.2f}, angle={angle:.2f}, confidence={confidence:.3f}")
        
        # Calculate corrected flow vector
        corrected_flow_x = lod_flow_x - dx
        corrected_flow_y = lod_flow_y - dy
        
        # Calculate final target position
        final_target_x = orig_x - corrected_flow_x
        final_target_y = orig_y - corrected_flow_y
        
        print(f"Corrected flow vector: ({corrected_flow_x:.2f}, {corrected_flow_y:.2f})")
        
        # Store analysis data
        self.detail_analysis_data = {
            'source_pixel': (orig_x, orig_y),
            'original_flow': original_flow,
            'lod_flow': (lod_flow_x, lod_flow_y),
            'lod_target': (lod_target_x, lod_target_y),
            'corrected_flow': (corrected_flow_x, corrected_flow_y),
            'final_target': (final_target_x, final_target_y),
            'region1': region1,
            'region2': region2,
            'bounds1': bounds1,
            'bounds2': bounds2,
            'phase_shift': (dx, dy),
            'angle': angle,
            'confidence': confidence,
            'lod_level': lod_level
        }
        
        # Enter detail analysis mode
        self.detail_analysis_mode = True
        
        # Show exit button
        self.exit_detail_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Update display
        self.update_display()
    
    def check_exit_detail_mode(self, action_description):
        """Check if user wants to exit detail analysis mode"""
        if self.detail_analysis_mode:
            result = messagebox.askyesno(
                "Выход из режима анализа",
                f"{action_description} приведет к сбросу всех дополнительных визуализаций. Продолжить?"
            )
            if result:
                self.exit_detail_analysis_mode()
                return True
            else:
                return False
        return True
    
    def exit_detail_analysis_mode(self):
        """Exit detail analysis mode and return to normal operation"""
        self.detail_analysis_mode = False
        self.detail_analysis_data = None
        # Hide exit button
        self.exit_detail_btn.pack_forget()
        # Clear detail analysis graphics
        self.canvas.delete("detail_analysis")
        # Update display
        self.update_display()
    
    def setup_ui(self):
        """Setup the GUI"""
        self.root = tk.Tk()
        self.root.title("Interactive Optical Flow Visualizer")
        self.root.geometry("1200x1700")
        
        # Bind window resize event
        self.root.bind('<Configure>', self.on_window_resize)
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Info label
        video_name = os.path.basename(self.video_path)
        cache_name = os.path.basename(self.flow_dir)
        frame_range = f"Frames {self.start_frame}-{self.start_frame + len(self.frames) - 1}"
        info_text = f"Video: {video_name} | Cache: {cache_name} | {frame_range}"
        self.info_label = ttk.Label(main_frame, text=info_text, font=("Arial", 9))
        self.info_label.pack(pady=(0, 5))
        
        # Statistics label
        stats_text = f"Total frames: {len(self.frames)} | Flow files: {len(self.flow_files)} | Start frame: {self.start_frame}"
        self.stats_label = ttk.Label(main_frame, text=stats_text, font=("Arial", 8), foreground="gray")
        self.stats_label.pack(pady=(0, 10))
        
        # Frame display area
        self.canvas_frame = ttk.Frame(main_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for frames
        self.canvas = tk.Canvas(self.canvas_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events
        self.canvas.bind('<Motion>', self.on_mouse_move)
        self.canvas.bind('<Leave>', self.on_mouse_leave)
        self.canvas.bind('<Button-1>', self.on_left_click)  # Left click for detail analysis
        self.canvas.bind('<MouseWheel>', self.on_mouse_wheel)
        self.canvas.bind('<Button-4>', self.on_mouse_wheel)  # Linux scroll up
        self.canvas.bind('<Button-5>', self.on_mouse_wheel)  # Linux scroll down
        self.canvas.bind('<Double-Button-1>', self.on_double_click)  # Double-click to reset zoom
        self.canvas.bind('<Button-2>', self.on_middle_click)  # Middle mouse button press
        self.canvas.bind('<ButtonRelease-2>', self.on_middle_release)  # Middle mouse button release
        self.canvas.bind('<B2-Motion>', self.on_middle_drag)  # Middle mouse drag
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Frame pair slider
        slider_frame = ttk.Frame(control_frame)
        slider_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(slider_frame, text="Frame Pair:").pack(side=tk.LEFT)
        
        self.frame_var = tk.IntVar(value=0)
        self.frame_slider = ttk.Scale(
            slider_frame, 
            from_=0, 
            to=self.max_pairs-1,
            orient=tk.HORIZONTAL,
            variable=self.frame_var,
            command=self.on_slider_change
        )
        
        # Bind slider drag events
        self.frame_slider.bind('<Button-1>', self.on_slider_press)
        self.frame_slider.bind('<ButtonRelease-1>', self.on_slider_release)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))
        
        self.frame_label = ttk.Label(slider_frame, text="0/0")
        self.frame_label.pack(side=tk.RIGHT)
        
        # Zoom controls
        zoom_frame = ttk.Frame(control_frame)
        zoom_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT)
        
        zoom_out_btn = ttk.Button(zoom_frame, text="-", width=3, command=self.zoom_out)
        zoom_out_btn.pack(side=tk.LEFT, padx=(10, 5))
        
        zoom_reset_btn = ttk.Button(zoom_frame, text="100%", width=6, command=self.zoom_reset)
        zoom_reset_btn.pack(side=tk.LEFT, padx=5)
        
        zoom_in_btn = ttk.Button(zoom_frame, text="+", width=3, command=self.zoom_in)
        zoom_in_btn.pack(side=tk.LEFT, padx=(5, 10))
        
        reset_pos_btn = ttk.Button(zoom_frame, text="Center", width=8, command=self.reset_position)
        reset_pos_btn.pack(side=tk.LEFT, padx=(10, 10))
        
        ttk.Label(zoom_frame, text="(Mouse wheel: zoom, Middle button: drag, Double-click: reset, Left click on red arrow: detail analysis)").pack(side=tk.LEFT, padx=(20, 0))
        
        # Quality map and vector controls
        vector_frame = ttk.Frame(control_frame)
        vector_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Quality map checkbox
        self.quality_map_var = tk.BooleanVar(value=True)
        quality_check = ttk.Checkbutton(vector_frame, text="Show Quality Map", 
                                       variable=self.quality_map_var, command=self.on_quality_toggle)
        quality_check.pack(side=tk.LEFT)
        
        # Vector mode controls
        ttk.Label(vector_frame, text="Flow Vectors:").pack(side=tk.LEFT, padx=(20, 5))
        
        self.vector_mode_var = tk.StringVar(value="original")
        original_radio = ttk.Radiobutton(vector_frame, text="Original", variable=self.vector_mode_var, 
                                        value="original", command=self.on_vector_mode_change)
        original_radio.pack(side=tk.LEFT, padx=(0, 5))
        
        # Check if LOD data is available before showing LOD option
        lod_available = self.check_lod_availability()
        if lod_available:
            lod_radio = ttk.Radiobutton(vector_frame, text="LOD", variable=self.vector_mode_var, 
                                       value="lod", command=self.on_vector_mode_change)
            lod_radio.pack(side=tk.LEFT, padx=(0, 10))
            
            # LOD level selection
            ttk.Label(vector_frame, text="LOD Level:").pack(side=tk.LEFT, padx=(10, 5))
            
            self.lod_level_var = tk.IntVar(value=0)
            lod_scale = ttk.Scale(vector_frame, from_=0, to=self.max_lod_levels-1, 
                                 orient=tk.HORIZONTAL, variable=self.lod_level_var,
                                 command=self.on_lod_level_change, length=100)
            lod_scale.pack(side=tk.LEFT, padx=(0, 5))
            
            self.lod_level_label = ttk.Label(vector_frame, text="0")
            self.lod_level_label.pack(side=tk.LEFT)
        
        # Status frame
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(fill=tk.X)
        
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT)
        
        self.mouse_label = ttk.Label(status_frame, text="")
        self.mouse_label.pack(side=tk.RIGHT)
        
        self.zoom_label = ttk.Label(status_frame, text="Zoom: 100%")
        self.zoom_label.pack(side=tk.RIGHT, padx=(20, 10))
        
        # Exit detail analysis button (initially hidden)
        self.exit_detail_btn = ttk.Button(status_frame, text="Exit Detail Analysis", 
                                         command=self.exit_detail_analysis_mode)
        # Don't pack initially - will be shown only in detail mode
        
    def resize_frame_for_display(self, frame):
        """Resize frame to fit display with zoom while maintaining aspect ratio"""
        h, w = frame.shape[:2]
        
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not initialized yet, use default
            canvas_width = 800
            canvas_height = 600
        
        # Reserve space for controls and spacing (three frames + spacing + margins)
        available_width = canvas_width - 40  # 20px margin on each side
        available_height = (canvas_height - 150) // 3  # Space for three frames plus controls
        
        # Calculate base scale to fit width
        base_scale_w = available_width / w
        base_scale_h = available_height / h
        base_scale = min(base_scale_w, base_scale_h)
        
        # Apply zoom factor
        final_scale = base_scale * self.zoom_factor
        
        new_w = int(w * final_scale)
        new_h = int(h * final_scale)
        
        # Ensure minimum size
        if new_w < 50 or new_h < 50:
            min_scale = max(50 / w, 50 / h)
            final_scale = min_scale
            new_w = int(w * final_scale)
            new_h = int(h * final_scale)
        
        resized = cv2.resize(frame, (new_w, new_h))
        return resized, final_scale
    
    def update_display(self):
        """Update the display with current frame pair"""
        if self.current_pair >= self.max_pairs:
            return
            
        # Load frames
        frame1 = self.frames[self.current_pair]
        frame2 = self.frames[self.current_pair + 1]
        
        # Store original frames for color comparison
        self.frame1 = frame1
        self.frame2 = frame2
        
        # Load flow data
        self.current_flow = self.load_flow_data(self.current_pair)
        
        # Get quality frame (compute if not available and enabled)
        if self.show_quality_map:
            if self.current_pair in self.quality_maps:
                quality_frame = self.quality_maps[self.current_pair]
            else:
                self.request_quality_map(self.current_pair)
                quality_frame = np.zeros_like(frame1)
        else:
            quality_frame = np.zeros_like(frame1)
        
        # Resize frames for display
        frame1_display, self.display_scale = self.resize_frame_for_display(frame1)
        frame2_display, _ = self.resize_frame_for_display(frame2)
        quality_display, _ = self.resize_frame_for_display(quality_frame)
        
        # Store original frame dimensions for coordinate mapping
        self.orig_height, self.orig_width = frame1.shape[:2]
        self.display_height, self.display_width = frame1_display.shape[:2]
        
        # Convert to PIL Images
        pil_frame1 = Image.fromarray(frame1_display)
        pil_frame2 = Image.fromarray(frame2_display)
        pil_quality = Image.fromarray(quality_display)
        
        # Convert to PhotoImage
        self.photo1 = ImageTk.PhotoImage(pil_frame1)
        self.photo2 = ImageTk.PhotoImage(pil_frame2)
        self.photo_quality = ImageTk.PhotoImage(pil_quality)
        
        # Clear canvas
        self.canvas.delete("all")
        
        # Calculate positions for centering with pan offset
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1:  # Canvas not initialized yet
            self.root.after(100, self.update_display)
            return
            
        x_offset = max(0, (canvas_width - self.display_width) // 2) + self.pan_offset_x
        
        # Position frames vertically with pan offset
        y1_offset = 10 + self.pan_offset_y
        y2_offset = y1_offset + self.display_height + 20
        y3_offset = y2_offset + self.display_height + 20
        
        # Store positions for mouse coordinate mapping
        self.frame1_x = x_offset
        self.frame1_y = y1_offset
        self.frame2_x = x_offset
        self.frame2_y = y2_offset
        self.frame3_x = x_offset
        self.frame3_y = y3_offset
        
        # Draw frames
        self.canvas.create_image(x_offset, y1_offset, anchor=tk.NW, image=self.photo1, tags="frame1")
        self.canvas.create_image(x_offset, y2_offset, anchor=tk.NW, image=self.photo2, tags="frame2")
        self.canvas.create_image(x_offset, y3_offset, anchor=tk.NW, image=self.photo_quality, tags="frame3")
        
        # Draw flow vectors as white lines over first frame
        self.draw_flow_vectors(x_offset, y1_offset)
        
        # Add frame labels (top-left corner)
        frame1_num = self.start_frame + self.current_pair
        frame2_num = self.start_frame + self.current_pair + 1
        
        self.canvas.create_text(x_offset, y1_offset - 5, anchor=tk.SW, 
                               text=f"Frame {self.current_pair} (#{frame1_num})", fill="white", font=("Arial", 10))
        self.canvas.create_text(x_offset, y2_offset - 5, anchor=tk.SW, 
                               text=f"Frame {self.current_pair + 1} (#{frame2_num})", fill="white", font=("Arial", 10))
        self.canvas.create_text(x_offset, y3_offset - 5, anchor=tk.SW, 
                               text=f"Flow Quality Map", fill="white", font=("Arial", 10))
        
        # Add frame numbers in corners
        # Top-left corners
        self.canvas.create_text(x_offset + 5, y1_offset + 5, anchor=tk.NW, 
                               text=str(frame1_num), fill="yellow", font=("Arial", 12, "bold"))
        self.canvas.create_text(x_offset + 5, y2_offset + 5, anchor=tk.NW, 
                               text=str(frame2_num), fill="yellow", font=("Arial", 12, "bold"))
        self.canvas.create_text(x_offset + 5, y3_offset + 5, anchor=tk.NW, 
                               text="Quality", fill="yellow", font=("Arial", 12, "bold"))
        
        # Top-right corners  
        self.canvas.create_text(x_offset + self.display_width - 5, y1_offset + 5, anchor=tk.NE, 
                               text=f"Pair {self.current_pair}", fill="cyan", font=("Arial", 10, "bold"))
        self.canvas.create_text(x_offset + self.display_width - 5, y2_offset + 5, anchor=tk.NE, 
                               text=f"Next", fill="cyan", font=("Arial", 10, "bold"))
        
        if self.detail_analysis_mode:
            quality_text = "Detail Analysis Mode"
        else:
            quality_text = f"Red=Bad, Green=Good" if self.show_quality_map else "Quality Map Disabled"
        
        self.canvas.create_text(x_offset + self.display_width - 5, y3_offset + 5, anchor=tk.NE, 
                               text=quality_text, fill="cyan", font=("Arial", 10, "bold"))
        
        # Draw detail analysis overlays if in detail mode
        if self.detail_analysis_mode:
            self.draw_detail_analysis_overlays(x_offset, y1_offset, x_offset, y2_offset)
            self.draw_detail_analysis_view()
        
        # Update labels
        self.frame_label.config(text=f"{self.current_pair}/{self.max_pairs-1}")
        
        # Update status with current mode info
        if self.current_flow is not None:
            flow_status = f"Flow loaded: {self.current_flow.shape}"
            
            # Detail analysis mode status
            if self.detail_analysis_mode:
                data = self.detail_analysis_data
                px, py = data['source_pixel']
                analysis_status = f"Detail Analysis: Pixel ({px},{py}), LOD{data['lod_level']}, Confidence={data['confidence']:.3f}"
                self.status_label.config(text=f"{flow_status} | {analysis_status}")
                return
            
            # Quality map status
            if self.show_quality_map:
                if self.current_pair in self.quality_maps:
                    quality_status = "Quality map: On"
                elif self.current_pair in self.computing_quality:
                    quality_status = "Quality map: Computing..."
                else:
                    quality_status = "Quality map: Computing..."
            else:
                quality_status = "Quality map: Off"
            
            # Vector mode status
            if self.use_lod_for_vectors:
                vector_status = f"Vectors: LOD{self.current_lod_level}"
            else:
                vector_status = "Vectors: Original"
            
            self.status_label.config(text=f"{flow_status} | {quality_status} | {vector_status}")
        else:
            self.status_label.config(text="No flow data")
    
    def update_display_quick(self):
        """Quick update display without quality map computation (for dragging)"""
        if self.current_pair >= self.max_pairs:
            return
            
        # Load frames
        frame1 = self.frames[self.current_pair]
        frame2 = self.frames[self.current_pair + 1]
        
        # Store original frames for color comparison
        self.frame1 = frame1
        self.frame2 = frame2
        
        # Load flow data
        self.current_flow = self.load_flow_data(self.current_pair)
        
        # Use black frame for quality (no computation during drag, or if disabled)
        quality_frame = np.zeros_like(frame1)
        
        # Resize frames for display
        frame1_display, self.display_scale = self.resize_frame_for_display(frame1)
        frame2_display, _ = self.resize_frame_for_display(frame2)
        quality_display, _ = self.resize_frame_for_display(quality_frame)
        
        # Store original frame dimensions for coordinate mapping
        self.orig_height, self.orig_width = frame1.shape[:2]
        self.display_height, self.display_width = frame1_display.shape[:2]
        
        # Convert to PIL Images
        pil_frame1 = Image.fromarray(frame1_display)
        pil_frame2 = Image.fromarray(frame2_display)
        pil_quality = Image.fromarray(quality_display)
        
        # Convert to PhotoImage
        self.photo1 = ImageTk.PhotoImage(pil_frame1)
        self.photo2 = ImageTk.PhotoImage(pil_frame2)
        self.photo_quality = ImageTk.PhotoImage(pil_quality)
        
        # Clear canvas
        self.canvas.delete("all")
        
        # Calculate positions for centering with pan offset
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1:  # Canvas not initialized yet
            self.root.after(100, self.update_display_quick)
            return
            
        x_offset = max(0, (canvas_width - self.display_width) // 2) + self.pan_offset_x
        
        # Position frames vertically with pan offset
        y1_offset = 10 + self.pan_offset_y
        y2_offset = y1_offset + self.display_height + 20
        y3_offset = y2_offset + self.display_height + 20
        
        # Store positions for mouse coordinate mapping
        self.frame1_x = x_offset
        self.frame1_y = y1_offset
        self.frame2_x = x_offset
        self.frame2_y = y2_offset
        self.frame3_x = x_offset
        self.frame3_y = y3_offset
        
        # Draw frames
        self.canvas.create_image(x_offset, y1_offset, anchor=tk.NW, image=self.photo1, tags="frame1")
        self.canvas.create_image(x_offset, y2_offset, anchor=tk.NW, image=self.photo2, tags="frame2")
        self.canvas.create_image(x_offset, y3_offset, anchor=tk.NW, image=self.photo_quality, tags="frame3")
        
        # Draw flow vectors as white lines over first frame
        self.draw_flow_vectors(x_offset, y1_offset)
        
        # Add frame labels (top-left corner)
        frame1_num = self.start_frame + self.current_pair
        frame2_num = self.start_frame + self.current_pair + 1
        
        self.canvas.create_text(x_offset, y1_offset - 5, anchor=tk.SW, 
                               text=f"Frame {self.current_pair} (#{frame1_num})", fill="white", font=("Arial", 10))
        self.canvas.create_text(x_offset, y2_offset - 5, anchor=tk.SW, 
                               text=f"Frame {self.current_pair + 1} (#{frame2_num})", fill="white", font=("Arial", 10))
        self.canvas.create_text(x_offset, y3_offset - 5, anchor=tk.SW, 
                               text=f"Flow Quality Map", fill="white", font=("Arial", 10))
        
        # Add frame numbers in corners
        # Top-left corners
        self.canvas.create_text(x_offset + 5, y1_offset + 5, anchor=tk.NW, 
                               text=str(frame1_num), fill="yellow", font=("Arial", 12, "bold"))
        self.canvas.create_text(x_offset + 5, y2_offset + 5, anchor=tk.NW, 
                               text=str(frame2_num), fill="yellow", font=("Arial", 12, "bold"))
        self.canvas.create_text(x_offset + 5, y3_offset + 5, anchor=tk.NW, 
                               text="Quality", fill="yellow", font=("Arial", 12, "bold"))
        
        # Top-right corners  
        self.canvas.create_text(x_offset + self.display_width - 5, y1_offset + 5, anchor=tk.NE, 
                               text=f"Pair {self.current_pair}", fill="cyan", font=("Arial", 10, "bold"))
        self.canvas.create_text(x_offset + self.display_width - 5, y2_offset + 5, anchor=tk.NE, 
                               text=f"Next", fill="cyan", font=("Arial", 10, "bold"))
        drag_text = f"Dragging... (Quality Map {'On' if self.show_quality_map else 'Off'})"
        self.canvas.create_text(x_offset + self.display_width - 5, y3_offset + 5, anchor=tk.NE, 
                               text=drag_text, fill="orange", font=("Arial", 10, "bold"))
        
        # Update labels
        self.frame_label.config(text=f"{self.current_pair}/{self.max_pairs-1}")
        
        # Update status for dragging
        if self.current_flow is not None:
            vector_status = f"Vectors: LOD{self.current_lod_level}" if self.use_lod_for_vectors else "Vectors: Original"
            self.status_label.config(text=f"Flow loaded: {self.current_flow.shape} | Quality map: Paused (dragging) | {vector_status}")
        else:
            self.status_label.config(text="No flow data")
    
    def on_slider_press(self, event):
        """Handle slider press (start of drag)"""
        self.slider_dragging = True
        self.stop_all_computations()
    
    def on_slider_release(self, event):
        """Handle slider release (end of drag)"""
        self.slider_dragging = False
        self.resume_computations()
        # Trigger update and preloading after drag ends
        self.update_display()
        self.preload_adjacent_frames()
    
    def on_slider_change(self, value):
        """Handle slider change"""
        new_pair = int(float(value))
        
        # Check if we need to exit detail mode
        if new_pair != self.current_pair:
            if not self.check_exit_detail_mode("Смена кадра"):
                # Revert slider to current position
                self.frame_var.set(self.current_pair)
                return
        
        self.current_pair = new_pair
        
        # Only update display during dragging, no computations
        if self.slider_dragging:
            # Quick update without quality map computation
            self.update_display_quick()
        else:
            # Full update with quality map computation
            self.update_display()
            # Preload adjacent frames for smoother navigation
            self.preload_adjacent_frames()
    
    def on_mouse_move(self, event):
        """Handle mouse movement over canvas"""
        # Clear previous arrow (but keep flow vectors)
        self.canvas.delete("flow_arrow")
        
        if self.current_flow is None:
            return
            
        # Check if mouse is over first frame (accounting for pan offset)
        if (self.frame1_x <= event.x <= self.frame1_x + self.display_width and
            self.frame1_y <= event.y <= self.frame1_y + self.display_height):
            
            # Convert display coordinates to original frame coordinates
            display_x = event.x - self.frame1_x
            display_y = event.y - self.frame1_y
            
            orig_x = int(display_x / self.display_scale)
            orig_y = int(display_y / self.display_scale)
            
            # Clamp to frame bounds
            orig_x = max(0, min(orig_x, self.orig_width - 1))
            orig_y = max(0, min(orig_y, self.orig_height - 1))
            
            # Get flow vector, handling different flow and frame resolutions
            if self.current_flow is not None:
                fh, fw = self.current_flow.shape[:2]
                
                # Calculate scale factors
                scale_x = fw / self.orig_width
                scale_y = fh / self.orig_height
                
                # Map to flow coordinates
                flow_x_coord = int(orig_x * scale_x)
                flow_y_coord = int(orig_y * scale_y)
                
                # Clamp to flow bounds
                flow_x_coord = max(0, min(flow_x_coord, fw - 1))
                flow_y_coord = max(0, min(flow_y_coord, fh - 1))
                
                # Get flow vector and scale it back to frame resolution
                flow_x = self.current_flow[flow_y_coord, flow_x_coord, 0] / scale_x
                flow_y = self.current_flow[flow_y_coord, flow_x_coord, 1] / scale_y
                
                # Calculate target position in original coordinates
                target_orig_x = orig_x - flow_x
                target_orig_y = orig_y - flow_y
                
                # Convert source pixel position to canvas coordinates on first frame
                source_display_x = orig_x * self.display_scale
                source_display_y = orig_y * self.display_scale
                canvas_source_x = self.frame1_x + source_display_x
                canvas_source_y = self.frame1_y + source_display_y
                
                # Convert target position to display coordinates on second frame
                target_display_x = target_orig_x * self.display_scale
                target_display_y = target_orig_y * self.display_scale
                canvas_target_x = self.frame2_x + target_display_x
                canvas_target_y = self.frame2_y + target_display_y
                
                # Check color consistency to determine arrow color and get details
                arrow_color, src_color, target_color, similarity = self.get_arrow_color_details(orig_x, orig_y, target_orig_x, target_orig_y)
                
                # Draw arrow from exact pixel on first frame to corresponding pixel on second frame
                self.draw_arrow(canvas_source_x, canvas_source_y, canvas_target_x, canvas_target_y, arrow_color)
                
                # Update mouse info in status bar (general info)
                self.mouse_label.config(
                    text=(f"Pixel ({orig_x},{orig_y}) → Flow ({flow_x:.1f},{flow_y:.1f}) | Similarity: {similarity:.3f}")
                )
                
                # --- Draw detailed color info with squares next to the arrow ---
                info_box_x = canvas_source_x + 20
                info_box_y = canvas_source_y
                square_size = 32

                # Convert RGB to hex for tkinter
                src_hex_color = f"#{src_color[0]:02x}{src_color[1]:02x}{src_color[2]:02x}"
                tgt_hex_color = f"#{target_color[0]:02x}{target_color[1]:02x}{target_color[2]:02x}"
                
                # Draw source color square
                self.canvas.create_rectangle(
                    info_box_x, info_box_y, 
                    info_box_x + square_size, info_box_y + square_size,
                    fill=src_hex_color, outline="white", width=1, tags="flow_arrow"
                )

                # Draw target color square
                self.canvas.create_rectangle(
                    info_box_x + square_size + 5, info_box_y, 
                    info_box_x + square_size * 2 + 5, info_box_y + square_size,
                    fill=tgt_hex_color, outline="white", width=1, tags="flow_arrow"
                )
                
                # Draw similarity text below the squares
                text_y = info_box_y + square_size + 5
                text_x_centered = info_box_x + (square_size * 2 + 5) / 2
                info_text = f"{similarity:.3f}"
                
                # Background for text
                self.canvas.create_text(text_x_centered + 1, text_y + 1, text=info_text, 
                                        anchor=tk.N, fill="black", font=("Arial", 10, "bold"), 
                                        tags="flow_arrow")
                # Foreground text
                self.canvas.create_text(text_x_centered, text_y, text=info_text, 
                                        anchor=tk.N, fill="white", font=("Arial", 10, "bold"), 
                                        tags="flow_arrow")
                
                self.canvas.tag_raise("flow_arrow") # Ensure it's drawn on top

            else:
                self.mouse_label.config(text="")
        else:
            self.mouse_label.config(text="")
    
    def on_mouse_leave(self, event):
        """Handle mouse leaving canvas"""
        self.canvas.delete("flow_arrow")
        self.mouse_label.config(text="")
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel for zooming"""
        # Determine scroll direction
        if event.delta > 0 or event.num == 4:  # Scroll up / zoom in
            zoom_change = self.zoom_step
        elif event.delta < 0 or event.num == 5:  # Scroll down / zoom out
            zoom_change = -self.zoom_step
        else:
            return
        
        # Update zoom factor
        new_zoom = self.zoom_factor + zoom_change
        new_zoom = max(self.min_zoom, min(new_zoom, self.max_zoom))
        
        if new_zoom != self.zoom_factor:
            self.zoom_factor = new_zoom
            self.zoom_label.config(text=f"Zoom: {int(self.zoom_factor * 100)}%")
            self.update_display()
    
    def on_double_click(self, event):
        """Handle double-click to reset zoom and position"""
        changed = False
        if self.zoom_factor != 1.0:
            self.zoom_factor = 1.0
            self.zoom_label.config(text="Zoom: 100%")
            changed = True
        if self.pan_offset_x != 0 or self.pan_offset_y != 0:
            self.pan_offset_x = 0
            self.pan_offset_y = 0
            changed = True
        if changed:
            self.update_display()
    
    def zoom_in(self):
        """Zoom in by zoom_step"""
        new_zoom = min(self.zoom_factor + self.zoom_step, self.max_zoom)
        if new_zoom != self.zoom_factor:
            self.zoom_factor = new_zoom
            self.zoom_label.config(text=f"Zoom: {int(self.zoom_factor * 100)}%")
            self.update_display()
    
    def zoom_out(self):
        """Zoom out by zoom_step"""
        new_zoom = max(self.zoom_factor - self.zoom_step, self.min_zoom)
        if new_zoom != self.zoom_factor:
            self.zoom_factor = new_zoom
            self.zoom_label.config(text=f"Zoom: {int(self.zoom_factor * 100)}%")
            self.update_display()
    
    def zoom_reset(self):
        """Reset zoom to 100%"""
        if self.zoom_factor != 1.0:
            self.zoom_factor = 1.0
            self.zoom_label.config(text="Zoom: 100%")
            self.update_display()
    
    def reset_position(self):
        """Reset pan position to center"""
        if self.pan_offset_x != 0 or self.pan_offset_y != 0:
            self.pan_offset_x = 0
            self.pan_offset_y = 0
            self.update_display()
    
    def on_quality_toggle(self):
        """Handle quality map checkbox toggle"""
        self.show_quality_map = self.quality_map_var.get()
        self.update_display()
    
    def on_vector_mode_change(self):
        """Handle vector mode radio button change"""
        mode = self.vector_mode_var.get()
        self.use_lod_for_vectors = (mode == "lod")
        self.update_display()
    
    def on_lod_level_change(self, value):
        """Handle LOD level slider change"""
        self.current_lod_level = int(float(value))
        if hasattr(self, 'lod_level_label'):
            self.lod_level_label.config(text=str(self.current_lod_level))
        if self.use_lod_for_vectors:
            self.update_display()
    
    def on_window_resize(self, event):
        """Handle window resize to update frame display"""
        # Only handle resize events for the main window, not child widgets
        if event.widget == self.root:
            # Delay update to avoid too frequent redraws during resize
            if hasattr(self, '_resize_timer'):
                self.root.after_cancel(self._resize_timer)
            self._resize_timer = self.root.after(200, self.update_display)
    
    def on_middle_click(self, event):
        """Handle middle mouse button press to start panning"""
        self.is_panning = True
        self.last_pan_x = event.x
        self.last_pan_y = event.y
        self.canvas.config(cursor="fleur")  # Change cursor to indicate panning
    
    def on_middle_release(self, event):
        """Handle middle mouse button release to stop panning"""
        self.is_panning = False
        self.canvas.config(cursor="")  # Reset cursor
    
    def on_middle_drag(self, event):
        """Handle middle mouse drag for panning"""
        if self.is_panning:
            # Calculate drag distance
            dx = event.x - self.last_pan_x
            dy = event.y - self.last_pan_y
            
            # Update pan offset
            self.pan_offset_x += dx
            self.pan_offset_y += dy
            
            # Update last position
            self.last_pan_x = event.x
            self.last_pan_y = event.y
            
            # Update display
            self.update_display()
    
    def calculate_pixel_quality(self, src_color, target_color):
        """Calculate quality score for a single pixel pair"""
        src_f = src_color.astype(float)
        tgt_f = target_color.astype(float)

        # 1. Euclidean distance in RGB space
        rgb_distance = np.sqrt(np.sum((src_f - tgt_f) ** 2))
        rgb_max_distance = np.sqrt(3 * 255**2)  # Maximum possible distance
        rgb_similarity = 1.0 - (rgb_distance / rgb_max_distance)
        
        # 2. Normalized absolute difference
        abs_diff = np.mean(np.abs(src_f - tgt_f)) / 255.0
        abs_similarity = 1.0 - abs_diff
        
        # 3. Cosine similarity (treating colors as vectors)
        src_norm = np.linalg.norm(src_f)
        target_norm = np.linalg.norm(tgt_f)
        
        # Use a small epsilon to avoid division by zero and handle near-black colors robustly
        if src_norm > 1e-6 and target_norm > 1e-6:
            cosine_sim = np.dot(src_f, tgt_f) / (src_norm * target_norm)
            cosine_similarity = (cosine_sim + 1.0) / 2.0  # Normalize to [0,1]
        else:
            # If one or both colors are black/near-black, angle is not a good metric.
            # Instead, base similarity on the difference in their brightness (norms).
            norm_diff = np.abs(src_norm - target_norm)
            cosine_similarity = 1.0 - (norm_diff / rgb_max_distance)
        
        # Use the average of all similarity metrics
        overall_similarity = (rgb_similarity + abs_similarity + cosine_similarity) / 3.0
        return overall_similarity
    
    def generate_quality_frame(self, frame1, frame2, flow):
        """Generate a quality visualization frame"""
        if flow is None:
            # Return black frame if no flow data
            return np.zeros_like(frame1)
        
        h, w = frame1.shape[:2]
        fh, fw = flow.shape[:2]
        quality_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Calculate scale factors if flow and frame dimensions don't match
        scale_x = w / fw if fw > 0 else 1.0
        scale_y = h / fh if fh > 0 else 1.0
        
        # Resize flow to match frame dimensions if needed
        if (fh != h or fw != w):
            flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
            # Scale flow vectors by the resize ratio
            flow[:, :, 0] *= scale_x
            flow[:, :, 1] *= scale_y
        
        # Vectorized approach for better performance
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Get flow vectors
        flow_x = flow[:, :, 0]
        flow_y = flow[:, :, 1]
        
        # Calculate target positions
        target_x = x_coords - flow_x
        target_y = y_coords - flow_y
        
        # Create masks for valid targets
        valid_mask = ((target_x >= 0) & (target_x < w) & 
                     (target_y >= 0) & (target_y < h))
        
        # Process valid pixels
        valid_y, valid_x = np.where(valid_mask)
        
        if len(valid_y) > 0:
            # Get source colors
            src_colors = frame1[valid_y, valid_x]
            
            # Get target coordinates (rounded to integers)
            tgt_x = np.round(target_x[valid_y, valid_x]).astype(int)
            tgt_y = np.round(target_y[valid_y, valid_x]).astype(int)
            
            # Ensure target coordinates are within bounds
            tgt_x = np.clip(tgt_x, 0, w-1)
            tgt_y = np.clip(tgt_y, 0, h-1)
            
            # Get target colors
            tgt_colors = frame2[tgt_y, tgt_x]
            
            # Calculate similarities for all valid pixels at once
            similarities = []
            for i in range(len(valid_y)):
                sim = self.calculate_pixel_quality(src_colors[i], tgt_colors[i])
                similarities.append(sim)
            similarities = np.array(similarities)
            
            # Color coding
            good_mask = similarities > 0.8
            
            # Good quality pixels - green
            good_indices = np.where(good_mask)[0]
            if len(good_indices) > 0:
                intensities = (255 * (similarities[good_indices] - 0.5) * 2).astype(int) # Scale green intensity
                quality_frame[valid_y[good_indices], valid_x[good_indices]] = np.column_stack([
                    np.zeros_like(intensities), np.clip(intensities, 0, 255), np.zeros_like(intensities)
                ])
            
            # Bad quality pixels - red
            bad_indices = np.where(~good_mask)[0]
            if len(bad_indices) > 0:
                intensities = (255 * (1.0 - similarities[bad_indices])).astype(int)
                quality_frame[valid_y[bad_indices], valid_x[bad_indices]] = np.column_stack([
                    intensities, np.zeros_like(intensities), np.zeros_like(intensities)
                ])
        
        # Out of bounds pixels - red
        invalid_y, invalid_x = np.where(~valid_mask)
        if len(invalid_y) > 0:
            quality_frame[invalid_y, invalid_x] = [255, 0, 0]
        
        return quality_frame
    
    def generate_quality_frame_fast(self, frame1, frame2, flow):
        """Generate a quality visualization frame - optimized version"""
        if flow is None:
            return np.zeros_like(frame1)
        
        h, w = frame1.shape[:2]
        fh, fw = flow.shape[:2]
        quality_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Calculate scale factors if flow and frame dimensions don't match
        scale_x = w / fw if fw > 0 else 1.0
        scale_y = h / fh if fh > 0 else 1.0
        
        # Resize flow to match frame dimensions if needed
        if (fh != h or fw != w):
            flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
            # Scale flow vectors by the resize ratio
            flow[:, :, 0] *= scale_x
            flow[:, :, 1] *= scale_y
        
        # Downsample for faster computation if image is large
        downsample_factor = 1
        if h * w > 500000:  # If more than 500k pixels
            downsample_factor = 2
        elif h * w > 1000000:  # If more than 1M pixels
            downsample_factor = 4
        
        # Process with downsampling
        step = downsample_factor
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                # Get flow vector
                flow_x = flow[y, x, 0]
                flow_y = flow[y, x, 1]
                
                # Calculate target position
                target_x = x - flow_x
                target_y = y - flow_y
                
                # Check bounds
                if (0 <= target_x < w and 0 <= target_y < h):
                    # Get colors
                    src_color = frame1[y, x]
                    target_color = frame2[int(target_y), int(target_x)]
                    
                    # Use the consistent, robust quality metric
                    similarity = self.calculate_pixel_quality(src_color, target_color)
                    
                    # Color coding
                    if similarity > 0.8:
                        # Good quality - green, scaled for better visibility
                        intensity = int(255 * (similarity - 0.5) * 2)
                        color = [0, np.clip(intensity, 0, 255), 0]
                    else:
                        # Bad quality - red
                        intensity = int(255 * (1.0 - similarity))
                        color = [intensity, 0, 0]
                else:
                    # Out of bounds - red
                    color = [255, 0, 0]
                
                # Fill block if downsampled
                y_end = min(y + step, h)
                x_end = min(x + step, w)
                quality_frame[y:y_end, x:x_end] = color
        
        return quality_frame
    
    def get_arrow_color(self, src_x, src_y, target_x, target_y):
        """Determine arrow color based on color consistency between source and target pixels"""
        color, _, _, _ = self.get_arrow_color_details(src_x, src_y, target_x, target_y)
        return color

    def get_arrow_color_details(self, src_x, src_y, target_x, target_y):
        """Get arrow color and detailed color information for the hover label."""
        # Default values for out-of-bounds cases
        default_color = "red"
        default_src_color = (0,0,0)
        default_tgt_color = (0,0,0)
        default_similarity = 0.0

        # Check bounds for both frames
        if (src_x < 0 or src_x >= self.orig_width or src_y < 0 or src_y >= self.orig_height or
            target_x < 0 or target_x >= self.orig_width or target_y < 0 or target_y >= self.orig_height):
            return default_color, default_src_color, default_tgt_color, default_similarity
        
        # Get pixel colors from both frames
        src_color = self.frame1[int(src_y), int(src_x)]
        target_color = self.frame2[int(target_y), int(target_x)]
        
        # Calculate quality using shared method
        overall_similarity = self.calculate_pixel_quality(src_color, target_color)
        
        # Threshold: green if similarity > 80%, red otherwise
        arrow_color = "green" if overall_similarity > 0.8 else "red"
        
        return arrow_color, tuple(src_color), tuple(target_color), overall_similarity

    def draw_flow_vectors(self, frame_x_offset, frame_y_offset):
        """Draw flow vectors as white lines over the first frame"""
        # Choose flow data source based on current mode
        if self.use_lod_for_vectors:
            # Use LOD data
            lod_flow = self.load_lod_data(self.current_pair, self.current_lod_level)
            if lod_flow is None:
                return  # No LOD data available
            flow_data = lod_flow
        else:
            # Use original flow data
            if self.current_flow is None:
                return
            flow_data = self.current_flow
        
        # Step size based on mode
        if self.use_lod_for_vectors:
            # For LOD: use step of 20 pixels from frame size
            frame_step = 20
            # Get LOD flow dimensions
            lod_h, lod_w = flow_data.shape[:2]
            # Calculate step in LOD coordinates
            lod_scale_x = lod_w / self.orig_width
            lod_scale_y = lod_h / self.orig_height
            step_x = max(1, int(frame_step * lod_scale_x))
            step_y = max(1, int(frame_step * lod_scale_y))
        else:
            # For original flow: use uniform step in flow coordinates
            step_x = step_y = 25
        
        # Get flow dimensions
        flow_h, flow_w = flow_data.shape[:2]
        
        # Calculate scale factors between flow and frame
        flow_to_frame_scale_x = self.orig_width / flow_w
        flow_to_frame_scale_y = self.orig_height / flow_h
        
        # Draw vectors with step size
        for y in range(0, flow_h, step_y):
            for x in range(0, flow_w, step_x):
                # Get flow vector
                flow_x = flow_data[y, x, 0]
                flow_y = flow_data[y, x, 1]
                
                # Skip very small vectors
                magnitude = np.sqrt(flow_x**2 + flow_y**2)
                if magnitude < 0.3:
                    continue
                
                # Convert flow coordinates to frame coordinates
                frame_x = x * flow_to_frame_scale_x
                frame_y = y * flow_to_frame_scale_y
                
                # Convert to display coordinates
                start_x = frame_x_offset + frame_x * self.display_scale
                start_y = frame_y_offset + frame_y * self.display_scale
                
                # Scale flow vector to frame coordinates, then to display coordinates
                scaled_flow_x = flow_x * flow_to_frame_scale_x
                scaled_flow_y = flow_y * flow_to_frame_scale_y
                end_x = start_x + scaled_flow_x * self.display_scale
                end_y = start_y + scaled_flow_y * self.display_scale
                
                # Draw flow vector as white line
                self.canvas.create_line(start_x, start_y, end_x, end_y, 
                                      fill="white", width=1, tags="flow_vectors")
                
                # Draw small circle at start point
                self.canvas.create_oval(start_x-1, start_y-1, start_x+1, start_y+1, 
                                      fill="white", outline="white", tags="flow_vectors")

    def draw_arrow(self, x1, y1, x2, y2, color="red", tags="flow_arrow"):
        """Draw arrow from (x1,y1) to (x2,y2) with specified color and tags"""
        # Clamp target to canvas bounds
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        x2 = max(0, min(x2, canvas_width))
        y2 = max(0, min(y2, canvas_height))
        
        # Calculate arrow properties
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx*dx + dy*dy)
        
        if length < 2:  # Too short to draw
            return
            
        # Normalize
        if length > 0:
            dx /= length
            dy /= length
        
        # Arrow head size
        head_length = min(10, length * 0.2)
        head_angle = 0.5  # radians
        
        # Arrow head points
        head_x1 = x2 - head_length * (dx * np.cos(head_angle) - dy * np.sin(head_angle))
        head_y1 = y2 - head_length * (dx * np.sin(head_angle) + dy * np.cos(head_angle))
        head_x2 = x2 - head_length * (dx * np.cos(-head_angle) - dy * np.sin(-head_angle))
        head_y2 = y2 - head_length * (dx * np.sin(-head_angle) + dy * np.cos(-head_angle))
        
        # Draw arrow line
        self.canvas.create_line(x1, y1, x2, y2, fill=color, width=2, tags=tags)
        
        # Draw arrow head
        self.canvas.create_line(x2, y2, head_x1, head_y1, fill=color, width=2, tags=tags)
        self.canvas.create_line(x2, y2, head_x2, head_y2, fill=color, width=2, tags=tags)
        
        # Draw circle at start point
        self.canvas.create_oval(x1-3, y1-3, x1+3, y1+3, fill=color, outline="white", tags=tags)
    
    def draw_detail_analysis_view(self):
        """Draw the detail analysis panel directly on the canvas."""
        if not self.detail_analysis_data:
            return

        canvas = self.canvas
        data = self.detail_analysis_data
        
        # --- 1. Sizing and Scaling ---
        display_size = 200  # We want to display them as 200x200
        border_size = 2
        
        # Scale 50x50 regions to 200x200
        scale_factor = display_size / (self.detail_analysis_region_size * 2)

        region1_scaled = cv2.resize(data['region1'], (display_size, display_size), interpolation=cv2.INTER_NEAREST)
        region2_scaled = cv2.resize(data['region2'], (display_size, display_size), interpolation=cv2.INTER_NEAREST)

        # --- 2. Marker Drawing ---
        def draw_marker(image, coords_in_region, color):
            marker_x = int(coords_in_region[0] * scale_factor)
            marker_y = int(coords_in_region[1] * scale_factor)
            cv2.rectangle(image, (marker_x - 2, marker_y - 2), (marker_x + 2, marker_y + 2), color, -1)

        draw_marker(region1_scaled, (25, 25), (255, 255, 0))

        orig_flow_x, orig_flow_y = data['original_flow']
        orig_target_x = data['source_pixel'][0] - orig_flow_x
        orig_target_y = data['source_pixel'][1] - orig_flow_y
        # Original Target (from original, bad flow) -> Orange
        draw_marker(region2_scaled, (orig_target_x - data['bounds2'][0], orig_target_y - data['bounds2'][1]), (255, 165, 0))
        
        # LOD Target (center of patch, from reliable flow) -> Red
        draw_marker(region2_scaled, (25, 25), (255, 0, 0))

        final_target_x, final_target_y = data['final_target']
        draw_marker(region2_scaled, (final_target_x - data['bounds2'][0], final_target_y - data['bounds2'][1]), (0, 255, 0))

        # --- 3. Difference View Calculation ---
        h_s, w_s = region2_scaled.shape[:2]
        center_scaled = (w_s / 2, h_s / 2)
        dx, dy, angle = data['phase_shift'][0], data['phase_shift'][1], data.get('angle', 0)
        
        # --- 3a. Create transformation matrices ---
        M_rot = cv2.getRotationMatrix2D(center_scaled, -angle, 1.0)
        shift_dx = dx * scale_factor
        shift_dy = dy * scale_factor
        # FIX 1: Correct sign for transformation matrix. It should be a positive shift.
        M_trans = np.float32([[1, 0, shift_dx], [0, 1, shift_dy]])
        
        # --- 3b. Apply transformations ---
        # Apply to the target region itself
        rotated_region2 = cv2.warpAffine(region2_scaled, M_rot, (w_s, h_s))
        transformed_region2 = cv2.warpAffine(rotated_region2, M_trans, (w_s, h_s))

        # FIX 2: Create and apply a mask to calculate difference only on the overlapping area
        # Create a white mask and apply the same transformations to it
        mask = np.ones_like(region2_scaled, dtype=np.uint8) * 255
        rotated_mask = cv2.warpAffine(mask, M_rot, (w_s, h_s), flags=cv2.INTER_NEAREST)
        transformed_mask = cv2.warpAffine(rotated_mask, M_trans, (w_s, h_s), flags=cv2.INTER_NEAREST)
        
        # Calculate difference and then mask it
        diff = cv2.absdiff(region1_scaled, transformed_region2)
        diff[transformed_mask < 255] = 0 # Black out areas outside the transformed region

        overlay_content = np.clip(diff * 2, 0, 255).astype(np.uint8)

        # --- 3c. Draw alignment frames ---
        # Source frame (static white box)
        cv2.rectangle(overlay_content, (0, 0), (w_s - 1, h_s - 1), (255, 255, 255), 1)
        
        # Target frame (transformed green box)
        src_pts = np.float32([[0,0], [w_s,0], [w_s,h_s], [0,h_s]]).reshape(-1,1,2)
        
        # Create a combined 3x3 transformation matrix for transforming the points of the polygon
        M_rot_3x3 = np.vstack([M_rot, [0, 0, 1]])
        M_trans_3x3 = np.float32([[1, 0, shift_dx], [0, 1, shift_dy], [0, 0, 1]])
        M_combined = np.dot(M_trans_3x3, M_rot_3x3)
        
        # Apply the combined transform to the corners of the frame
        transformed_pts = cv2.transform(src_pts, M_combined[:2, :]).astype(np.int32)
        cv2.polylines(overlay_content, [transformed_pts], True, (0, 255, 0), 1)
        
        # --- 4. Add Borders & Finalize Images ---
        region1_bordered = cv2.copyMakeBorder(region1_scaled, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[255,255,255])
        region2_bordered = cv2.copyMakeBorder(region2_scaled, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[255,255,255])
        overlay_bordered = cv2.copyMakeBorder(overlay_content, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[255,255,255])

        # --- 5. Canvas Drawing ---
        base_y = self.frame3_y + self.display_height + 40
        spacing = 20
        region_h, region_w = region1_bordered.shape[:2]

        x1 = spacing
        x2 = x1 + region_w + spacing
        x3 = x2 + region_w + spacing

        # Convert to PhotoImage and store references
        self.analysis_photo1 = ImageTk.PhotoImage(Image.fromarray(region1_bordered))
        self.analysis_photo2 = ImageTk.PhotoImage(Image.fromarray(region2_bordered))
        self.analysis_photo_diff = ImageTk.PhotoImage(Image.fromarray(overlay_bordered))

        canvas.create_image(x1, base_y, anchor=tk.NW, image=self.analysis_photo1, tags="detail_analysis")
        canvas.create_image(x2, base_y, anchor=tk.NW, image=self.analysis_photo2, tags="detail_analysis")
        canvas.create_image(x3, base_y, anchor=tk.NW, image=self.analysis_photo_diff, tags="detail_analysis")
        
        # --- 6. Labels and Legends ---
        font_details = ("Arial", 8)
        legend_y_offset = base_y + region_h + 15
        
        canvas.create_text(x1, base_y - 5, text="Source", anchor=tk.SW, fill="white", tags="detail_analysis")
        canvas.create_text(x1, legend_y_offset, text="■ Yellow: Selected Pixel", anchor=tk.NW, fill="yellow", font=font_details, tags="detail_analysis")
        
        canvas.create_text(x2, base_y - 5, text="Target", anchor=tk.SW, fill="white", tags="detail_analysis")
        canvas.create_text(x2, legend_y_offset, text="■ Orange: Original Target", anchor=tk.NW, fill="orange", font=font_details, tags="detail_analysis")
        canvas.create_text(x2, legend_y_offset + 12, text="■ Red: LOD Target", anchor=tk.NW, fill="red", font=font_details, tags="detail_analysis")
        canvas.create_text(x2, legend_y_offset + 24, text="■ Green: Corrected Target", anchor=tk.NW, fill="lime", font=font_details, tags="detail_analysis")

        canvas.create_text(x3, base_y - 5, text="Difference & Alignment", anchor=tk.SW, fill="white", tags="detail_analysis")
        canvas.create_text(x3, legend_y_offset, text="White: Source Frame", anchor=tk.NW, fill="white", font=font_details, tags="detail_analysis")
        canvas.create_text(x3, legend_y_offset + 12, text="Green: Target Frame (Aligned)", anchor=tk.NW, fill="lime", font=font_details, tags="detail_analysis")
    
    def draw_detail_analysis_overlays(self, frame1_x, frame1_y, frame2_x, frame2_y):
        """Draw overlay vectors for detail analysis mode on the main frames."""
        if not self.detail_analysis_data:
            return

        data = self.detail_analysis_data
        source_x, source_y = data['source_pixel']

        # Calculate original target
        orig_flow_x, orig_flow_y = data['original_flow']
        orig_target_x = source_x - orig_flow_x
        orig_target_y = source_y - orig_flow_y
        
        # Get LOD target
        lod_target_x, lod_target_y = data['lod_target']
        
        def to_display_coords(orig_x, orig_y, frame_x_offset, frame_y_offset):
            return (frame_x_offset + orig_x * self.display_scale,
                    frame_y_offset + orig_y * self.display_scale)
        
        # Convert points to canvas coordinates
        src_display_x, src_display_y = to_display_coords(source_x, source_y, frame1_x, frame1_y)
        orig_tgt_display_x, orig_tgt_display_y = to_display_coords(orig_target_x, orig_target_y, frame2_x, frame2_y)
        lod_tgt_display_x, lod_tgt_display_y = to_display_coords(lod_target_x, lod_target_y, frame2_x, frame2_y)
        
        # Draw Original Flow Vector (Red)
        self.draw_arrow(src_display_x, src_display_y, orig_tgt_display_x, orig_tgt_display_y, 
                        color="red", tags="detail_analysis")
        
        # Draw LOD Flow Vector (Orange)
        self.draw_arrow(src_display_x, src_display_y, lod_tgt_display_x, lod_tgt_display_y, 
                        color="orange", tags="detail_analysis")

        # Draw source point circle
        self.canvas.create_oval(src_display_x-3, src_display_y-3, src_display_x+3, src_display_y+3, 
                               fill="yellow", outline="white", tags="detail_analysis")

    def run(self):
        """Run the GUI"""
        self.root.mainloop()

def main():
    parser = argparse.ArgumentParser(description='Interactive Optical Flow Visualizer')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--flow-dir', required=True, help='Directory containing optical flow files')
    parser.add_argument('--start-frame', type=int, default=0, help='Starting frame number')
    parser.add_argument('--max-frames', type=int, default=None, help='Maximum number of frames')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
        
    if not os.path.exists(args.flow_dir):
        print(f"Error: Flow directory not found: {args.flow_dir}")
        return
    
    try:
        visualizer = FlowVisualizer(args.video, args.flow_dir, args.start_frame, args.max_frames)
        visualizer.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 