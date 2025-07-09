import sys
from PyQt6.QtWidgets import QApplication

# Create the application instance first to avoid initialization errors
app = QApplication(sys.argv)

import os
import cv2
import subprocess
import json
from pathlib import Path
from threading import Thread
from queue import Queue, Empty

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QSettings
from PyQt6.QtWidgets import (QWidget, QLabel, QLineEdit,
                             QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QCheckBox, QComboBox, QDoubleSpinBox, QSpinBox, QFrame,
                             QScrollArea, QSizePolicy, QProgressBar, QTextEdit, QSlider)
from PyQt6.QtGui import QPixmap, QImage, QIcon, QFont

from qfluentwidgets import (setTheme, Theme, TitleLabel, SubtitleLabel, LineEdit, PushButton,
                            InfoBar, InfoBarPosition, CheckBox, ComboBox, DoubleSpinBox, SpinBox,
                            BodyLabel, CardWidget, HyperlinkButton, ProgressBar, TextEdit)
import qtawesome as qta

# Do not import this at module level
# from flow_processor import VideoFlowProcessor


class VideoThread(QThread):
    """Thread for reading video frames to prevent UI freezing"""
    frame_loaded = pyqtSignal(object, int, float)  # frame, total_frames, fps
    
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        
    def run(self):
        cap = cv2.VideoCapture(str(self.video_path))
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Load first frame
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_loaded.emit(frame, total_frames, fps)
        
        cap.release()


class ProcessThread(QThread):
    """Thread for running external processes to prevent UI freezing"""
    output_received = pyqtSignal(str)
    process_finished = pyqtSignal(int)
    
    def __init__(self, command):
        super().__init__()
        self.command = command
        
    def run(self):
        try:
            process = subprocess.Popen(
                self.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    self.output_received.emit(line.strip())
            
            process.wait()
            self.process_finished.emit(process.returncode)
            
        except Exception as e:
            self.output_received.emit(f"Error: {str(e)}")
            self.process_finished.emit(1)


class FlowRunnerApp(QWidget):
    def __init__(self):
        super().__init__()
        
        # Settings for persistence
        self.settings = QSettings('VideoFlow', 'FlowRunner')
        
        # --- State Variables ---
        self.video_path = ""
        self.output_path = ""
        self.flow_cache_path = ""
        self.current_frame = None
        self.total_frames = 0
        self.fps = 0
        self.video_thread = None
        self.process_thread = None
        
        # Import and create instance only when needed
        self.flow_processor_instance = None

        # --- UI Initialization ---
        self._initialization_complete = False
        self.init_ui()
        self.load_settings()
        
        # Mark initialization as complete
        self._initialization_complete = True
        
        # Update command preview now that initialization is complete
        self.update_command_preview()

    def init_ui(self):
        setTheme(Theme.DARK)
        self.setWindowTitle("VideoFlow Processor GUI")
        self.setMinimumSize(1400, 900)
        
        # Set smaller font for compact interface
        font = QFont()
        font.setPointSize(8)
        self.setFont(font)

        # Main layout
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # Left panel - Settings
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)

        # Right panel - Video preview and controls
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 1)

        # Bottom panel - Command and execution
        self.create_bottom_panel(main_layout)

        # Timer for real-time command updates
        self.command_timer = QTimer()
        self.command_timer.timeout.connect(self.update_command_preview)
        self.command_timer.setSingleShot(True)

    def create_left_panel(self):
        """Create the left panel with all flow processor settings"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(450)
        
        container = QWidget()
        layout = QVBoxLayout(container)
        
        # Title
        title = SubtitleLabel("Processing Settings")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Boolean flags card
        flags_card = CardWidget()
        flags_layout = QGridLayout(flags_card)
        
        flags_title = BodyLabel("Processing Options")
        flags_layout.addWidget(flags_title, 0, 0, 1, 2)
        
        # Boolean options with tooltips
        self.flag_widgets = {}
        flag_configs = [
            ('fast', False, 'Fast mode', 'Use fast processing mode with reduced quality'),
            ('tile', True, 'Tile mode', 'Enable tile-based processing for large videos'),
            ('vertical', False, 'Vertical', 'Process video in vertical orientation'),
            ('flow_only', False, 'Flow only', 'Output only optical flow without video composition'),
            ('taa', True, 'TAA', 'Apply Temporal Anti-Aliasing for smoother results'),
            ('taa_emulate_compression', False, 'TAA Emulate Compression', 'Emulate motion vectors compression/decompression in TAA processing'),
            ('lossless', False, 'Lossless', 'Use lossless compression (FFV1)'),
            ('uncompressed', False, 'Uncompressed', 'Save without any compression (raw format)'),
            ('skip_lods', False, 'Skip LODs', 'Skip Level of Detail (LOD) generation for faster processing'),
            ('force_recompute', False, 'Force recompute', 'Delete existing cache and force recomputation of optical flow')
        ]
        
        row = 1
        for i, (key, default, label, tooltip) in enumerate(flag_configs):
            checkbox = CheckBox()
            checkbox.setChecked(default)
            checkbox.setText(label)
            checkbox.setToolTip(tooltip)
            checkbox.stateChanged.connect(self.on_setting_changed)
            self.flag_widgets[key] = checkbox
            flags_layout.addWidget(checkbox, row + i // 2, i % 2)
        
        layout.addWidget(flags_card)

        # Frame/Time controls card
        time_card = CardWidget()
        time_layout = QGridLayout(time_card)
        
        time_title = BodyLabel("Time Controls")
        time_layout.addWidget(time_title, 0, 0, 1, 2)
        
        # Control type selector
        time_layout.addWidget(BodyLabel("Control type:"), 1, 0)
        self.time_control_combo = ComboBox()
        self.time_control_combo.addItems(['Control by frame', 'Control by time'])
        self.time_control_combo.setToolTip("Choose whether to control by frame numbers or time")
        self.time_control_combo.currentTextChanged.connect(self.on_time_control_changed)
        time_layout.addWidget(self.time_control_combo, 1, 1)
        
        # Frame controls (visible by default)
        self.frame_start_label = BodyLabel("Start frame:")
        time_layout.addWidget(self.frame_start_label, 2, 0)
        self.start_frame_spin = SpinBox()
        self.start_frame_spin.setRange(0, 99999)
        self.start_frame_spin.setToolTip("Starting frame number for processing")
        self.start_frame_spin.valueChanged.connect(self.on_setting_changed)
        time_layout.addWidget(self.start_frame_spin, 2, 1)
        
        self.frame_max_label = BodyLabel("Max frames:")
        time_layout.addWidget(self.frame_max_label, 3, 0)
        self.max_frames_spin = SpinBox()
        self.max_frames_spin.setRange(0, 99999)
        self.max_frames_spin.setValue(0)
        self.max_frames_spin.setToolTip("Maximum number of frames to process (0 = all)")
        self.max_frames_spin.valueChanged.connect(self.on_setting_changed)
        time_layout.addWidget(self.max_frames_spin, 3, 1)
        
        # Time controls (hidden by default)
        self.time_start_label = BodyLabel("Start time (s):")
        time_layout.addWidget(self.time_start_label, 2, 0)
        self.start_time_spin = DoubleSpinBox()
        self.start_time_spin.setRange(0, 99999)
        self.start_time_spin.setDecimals(2)
        self.start_time_spin.setToolTip("Starting time in seconds")
        self.start_time_spin.valueChanged.connect(self.on_setting_changed)
        time_layout.addWidget(self.start_time_spin, 2, 1)
        
        self.time_duration_label = BodyLabel("Duration (s):")
        time_layout.addWidget(self.time_duration_label, 3, 0)
        self.duration_spin = DoubleSpinBox()
        self.duration_spin.setRange(0, 99999)
        self.duration_spin.setDecimals(2)
        self.duration_spin.setToolTip("Duration in seconds (0 = until end)")
        self.duration_spin.valueChanged.connect(self.on_setting_changed)
        time_layout.addWidget(self.duration_spin, 3, 1)
        
        # Initially show frame controls
        self.time_start_label.hide()
        self.start_time_spin.hide()
        self.time_duration_label.hide()
        self.duration_spin.hide()
        
        layout.addWidget(time_card)

        # Flow parameters card
        flow_card = CardWidget()
        flow_layout = QGridLayout(flow_card)
        
        flow_title = BodyLabel("Flow Parameters")
        flow_layout.addWidget(flow_title, 0, 0, 1, 2)
        
        # Model selection
        flow_layout.addWidget(BodyLabel("Model:"), 1, 0)
        self.model_combo = ComboBox()
        self.model_combo.addItems(['videoflow', 'memflow'])
        self.model_combo.setCurrentText('videoflow')
        self.model_combo.setToolTip("Optical flow model: VideoFlow (MOF) or MemFlow")
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        flow_layout.addWidget(self.model_combo, 1, 1)
        
        # Dataset selection (for both models)
        flow_layout.addWidget(BodyLabel("Dataset:"), 2, 0)
        self.dataset_combo = ComboBox()
        self.dataset_combo.addItems(['sintel', 'things', 'kitti'])
        self.dataset_combo.setCurrentText('sintel')
        self.dataset_combo.setToolTip("Training dataset for the model")
        self.dataset_combo.currentTextChanged.connect(self.on_setting_changed)
        flow_layout.addWidget(self.dataset_combo, 2, 1)
        
        # VideoFlow architecture selection (only for VideoFlow)
        flow_layout.addWidget(BodyLabel("VF Architecture:"), 3, 0)
        self.vf_architecture_combo = ComboBox()
        self.vf_architecture_combo.addItems(['mof', 'bof'])
        self.vf_architecture_combo.setCurrentText('mof')
        self.vf_architecture_combo.setToolTip("VideoFlow architecture: MOF (MOFNet) or BOF (BOFNet)")
        self.vf_architecture_combo.currentTextChanged.connect(self.on_setting_changed)
        flow_layout.addWidget(self.vf_architecture_combo, 3, 1)
        
        # VideoFlow variant selection (only for VideoFlow)
        flow_layout.addWidget(BodyLabel("VF Variant:"), 4, 0)
        self.vf_variant_combo = ComboBox()
        self.vf_variant_combo.addItems(['standard', 'noise'])
        self.vf_variant_combo.setCurrentText('standard')
        self.vf_variant_combo.setToolTip("VideoFlow variant: standard or noise (things_288960noise)")
        self.vf_variant_combo.currentTextChanged.connect(self.on_setting_changed)
        flow_layout.addWidget(self.vf_variant_combo, 4, 1)
        
        # Device
        flow_layout.addWidget(BodyLabel("Device:"), 5, 0)
        self.device_combo = ComboBox()
        self.device_combo.addItems(['cuda', 'cpu'])
        self.device_combo.setCurrentText('cuda')
        self.device_combo.setToolTip("Processing device (CPU or CUDA GPU)")
        self.device_combo.currentTextChanged.connect(self.on_setting_changed)
        flow_layout.addWidget(self.device_combo, 5, 1)
        
        # Flow format
        flow_layout.addWidget(BodyLabel("Flow format:"), 6, 0)
        self.flow_format_combo = ComboBox()
        self.flow_format_combo.addItems(['gamedev', 'hsv', 'torchvision', 'motion-vectors-rg8', 'motion-vectors-rgb8'])
        self.flow_format_combo.setToolTip("Output format for optical flow visualization")
        self.flow_format_combo.currentTextChanged.connect(self.on_flow_format_changed)
        flow_layout.addWidget(self.flow_format_combo, 6, 1)
        
        # Motion vectors clamp range (initially hidden)
        flow_layout.addWidget(BodyLabel("MV clamp range:"), 7, 0)
        self.mv_clamp_range_spin = DoubleSpinBox()
        self.mv_clamp_range_spin.setRange(1.0, 512.0)
        self.mv_clamp_range_spin.setValue(32.0)
        self.mv_clamp_range_spin.setSingleStep(1.0)
        self.mv_clamp_range_spin.setDecimals(1)
        self.mv_clamp_range_spin.setToolTip("Clamp range for motion vectors encoding formats")
        self.mv_clamp_range_spin.valueChanged.connect(self.on_setting_changed)
        flow_layout.addWidget(self.mv_clamp_range_spin, 7, 1)
        
        # Store the motion vectors clamp range label and spin for hiding/showing
        self.mv_clamp_range_label = flow_layout.itemAtPosition(7, 0).widget()
        self.update_mv_clamp_range_visibility()
        
        # Save flow format
        flow_layout.addWidget(BodyLabel("Save flow:"), 8, 0)
        self.save_flow_combo = ComboBox()
        self.save_flow_combo.addItems(['none', 'npz', 'flo', 'both'])
        self.save_flow_combo.setCurrentText('npz')
        self.save_flow_combo.setToolTip("Format for saving flow data: flo (Middlebury), npz (NumPy), both, none (don't save)")
        self.save_flow_combo.currentTextChanged.connect(self.on_setting_changed)
        flow_layout.addWidget(self.save_flow_combo, 8, 1)
        
        # Sequence length
        flow_layout.addWidget(BodyLabel("Sequence length:"), 9, 0)
        self.sequence_length_spin = SpinBox()
        self.sequence_length_spin.setRange(3, 20)
        self.sequence_length_spin.setValue(5)
        self.sequence_length_spin.setToolTip("Number of frames in processing sequence")
        self.sequence_length_spin.valueChanged.connect(self.on_setting_changed)
        flow_layout.addWidget(self.sequence_length_spin, 9, 1)
        
        layout.addWidget(flow_card)

        # Kalman filter card
        kalman_card = CardWidget()
        kalman_layout = QGridLayout(kalman_card)
        
        kalman_title = BodyLabel("Kalman Filter")
        kalman_layout.addWidget(kalman_title, 0, 0, 1, 2)
        
        kalman_layout.addWidget(BodyLabel("Smoothing:"), 1, 0)
        self.kalman_smooth_spin = DoubleSpinBox()
        self.kalman_smooth_spin.setRange(0.0, 1.0)
        self.kalman_smooth_spin.setValue(0.5)
        self.kalman_smooth_spin.setSingleStep(0.1)
        self.kalman_smooth_spin.setDecimals(2)
        self.kalman_smooth_spin.setToolTip("Kalman filter smoothing parameter (0.0-1.0)")
        self.kalman_smooth_spin.valueChanged.connect(self.on_setting_changed)
        kalman_layout.addWidget(self.kalman_smooth_spin, 1, 1)
        
        layout.addWidget(kalman_card)

        layout.addStretch()
        
        scroll.setWidget(container)
        return scroll

    def create_right_panel(self):
        """Create the right panel with video preview and file controls"""
        container = QWidget()
        layout = QVBoxLayout(container)

        # File I/O section
        io_card = CardWidget()
        io_layout = QGridLayout(io_card)
        
        io_title = SubtitleLabel("File I/O")
        io_layout.addWidget(io_title, 0, 0, 1, 3)
        
        # Input video
        io_layout.addWidget(BodyLabel("Input video:"), 1, 0)
        self.input_video_edit = LineEdit()
        self.input_video_edit.setToolTip("Path to input video file")
        self.input_video_edit.textChanged.connect(self.on_setting_changed)
        io_layout.addWidget(self.input_video_edit, 1, 1)
        
        browse_input_btn = PushButton("Browse")
        browse_input_btn.clicked.connect(self.browse_input_video)
        io_layout.addWidget(browse_input_btn, 1, 2)
        
        # Output path
        io_layout.addWidget(BodyLabel("Output path:"), 2, 0)
        self.output_path_edit = LineEdit()
        self.output_path_edit.setToolTip("Output directory path")
        self.output_path_edit.textChanged.connect(self.on_setting_changed)
        io_layout.addWidget(self.output_path_edit, 2, 1)
        
        browse_output_btn = PushButton("Browse")
        browse_output_btn.clicked.connect(self.browse_output_path)
        io_layout.addWidget(browse_output_btn, 2, 2)
        
        # Flow cache
        io_layout.addWidget(BodyLabel("Flow cache:"), 3, 0)
        self.flow_cache_edit = LineEdit()
        self.flow_cache_edit.setToolTip("Path to flow cache directory (optional)")
        self.flow_cache_edit.textChanged.connect(self.on_flow_cache_changed)
        io_layout.addWidget(self.flow_cache_edit, 3, 1)
        
        browse_cache_btn = PushButton("Browse")
        browse_cache_btn.clicked.connect(self.browse_flow_cache)
        io_layout.addWidget(browse_cache_btn, 3, 2)
        
        # Flow input video (for TAA comparison)
        io_layout.addWidget(BodyLabel("Flow input video:"), 4, 0)
        self.flow_input_edit = LineEdit()
        self.flow_input_edit.setToolTip("Path to video with encoded motion vectors in bottom half (for TAA comparison)")
        self.flow_input_edit.textChanged.connect(self.on_setting_changed)
        io_layout.addWidget(self.flow_input_edit, 4, 1)
        
        browse_flow_input_btn = PushButton("Browse")
        browse_flow_input_btn.clicked.connect(self.browse_flow_input)
        io_layout.addWidget(browse_flow_input_btn, 4, 2)
        
        # Cache status
        self.flow_cache_status_label = BodyLabel("")
        io_layout.addWidget(self.flow_cache_status_label, 5, 0, 1, 3)
        
        layout.addWidget(io_card)

        # Video preview section
        preview_card = CardWidget()
        preview_layout = QVBoxLayout(preview_card)
        
        preview_title = SubtitleLabel("Video Preview")
        preview_layout.addWidget(preview_title)
        
        # Video display container
        self.video_container = QWidget()
        self.video_container.setMinimumSize(400, 300)
        self.video_container.setStyleSheet("border: 1px solid gray; background-color: black;")
        
        # Video label inside container
        self.video_label = QLabel(self.video_container)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setText("No video loaded")
        self.video_label.setStyleSheet("border: none; background-color: black;")
        
        # Control buttons style
        button_style = """
            QPushButton {
                background-color: rgba(0, 0, 0, 150);
                border: 2px solid white;
                border-radius: 20px;
                color: white;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 200);
            }
        """
        
        # Pause/Play button in top-right corner
        self.pause_play_btn = PushButton(self.video_container)
        self.pause_play_btn.setIcon(qta.icon('fa5s.pause'))
        self.pause_play_btn.setFixedSize(40, 40)
        self.pause_play_btn.setStyleSheet(button_style)
        self.pause_play_btn.clicked.connect(self.toggle_video_playback)
        self.pause_play_btn.hide()  # Hidden until video is loaded
        
        # Restart button next to pause button
        self.restart_btn = PushButton(self.video_container)
        self.restart_btn.setIcon(qta.icon('fa5s.undo'))
        self.restart_btn.setFixedSize(40, 40)
        self.restart_btn.setStyleSheet(button_style)
        self.restart_btn.setToolTip("Restart video from beginning")
        self.restart_btn.clicked.connect(self.restart_video)
        self.restart_btn.hide()  # Hidden until video is loaded
        
        # Video statistics in bottom-right corner
        self.video_stats_label = QLabel(self.video_container)
        self.video_stats_label.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 150);
                border: 1px solid white;
                border-radius: 5px;
                color: white;
                padding: 5px;
                font-family: 'Consolas', monospace;
                font-size: 10px;
            }
        """)
        self.video_stats_label.setText("Frame: 0/0\nTime: 0.00s")
        self.video_stats_label.hide()  # Hidden until video is loaded
        
        preview_layout.addWidget(self.video_container)
        
        # Initialize video playback state
        self.is_video_playing = False
        self.current_frame_number = 0
        self.range_start = 0
        self.range_end = 100
        
        # Frame info
        self.frame_info_label = BodyLabel("No video loaded")
        preview_layout.addWidget(self.frame_info_label)
        
        layout.addWidget(preview_card)

        return container

    def create_bottom_panel(self, main_layout):
        """Create the bottom panel with command preview and execution controls"""
        bottom_widget = QWidget()
        bottom_widget.setMaximumHeight(200)
        bottom_layout = QVBoxLayout(bottom_widget)
        
        # Command preview
        cmd_title = BodyLabel("Generated Command")
        bottom_layout.addWidget(cmd_title)
        
        self.command_preview = TextEdit()
        self.command_preview.setMaximumHeight(60)
        self.command_preview.setReadOnly(True)
        self.command_preview.setStyleSheet("background-color: #2d2d2d; font-family: 'Consolas', monospace;")
        bottom_layout.addWidget(self.command_preview)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        self.interactive_btn = PushButton("Interactive Mode")
        self.interactive_btn.setIcon(qta.icon('fa5s.terminal'))
        self.interactive_btn.clicked.connect(self.run_interactive)
        buttons_layout.addWidget(self.interactive_btn)
        
        self.process_btn = PushButton("Start Processing")
        self.process_btn.setIcon(qta.icon('fa5s.play'))
        self.process_btn.clicked.connect(self.run_processing)
        buttons_layout.addWidget(self.process_btn)
        
        buttons_layout.addStretch()
        
        bottom_layout.addLayout(buttons_layout)
        
        # Add to main layout
        main_layout_widget = QWidget()
        main_layout_widget.setLayout(main_layout)
        
        final_layout = QVBoxLayout()
        final_layout.addWidget(main_layout_widget)
        final_layout.addWidget(bottom_widget)
        
        self.setLayout(final_layout)

    def load_settings(self):
        """Load saved settings"""
        # Temporarily block signals to prevent cascading updates during initialization
        self.blockSignals(True)
        
        try:
            # Load file paths (don't trigger signals)
            self.input_video_edit.blockSignals(True)
            self.output_path_edit.blockSignals(True)
            self.flow_cache_edit.blockSignals(True)
            self.flow_input_edit.blockSignals(True)
            
            self.input_video_edit.setText(self.settings.value('input_video', ''))
            self.output_path_edit.setText(self.settings.value('output_path', ''))
            self.flow_cache_edit.setText(self.settings.value('flow_cache', ''))
            self.flow_input_edit.setText(self.settings.value('flow_input', ''))
            
            self.input_video_edit.blockSignals(False)
            self.output_path_edit.blockSignals(False)
            self.flow_cache_edit.blockSignals(False)
            self.flow_input_edit.blockSignals(False)
            
            # Load processing settings
            for key, widget in self.flag_widgets.items():
                widget.blockSignals(True)
                value = self.settings.value(f'flag_{key}', widget.isChecked())
                widget.setChecked(value if isinstance(value, bool) else value == 'true')
                widget.blockSignals(False)
            
            # Load numeric settings
            numeric_widgets = [
                (self.start_frame_spin, 'start_frame', 0),
                (self.max_frames_spin, 'max_frames', 0),
                (self.start_time_spin, 'start_time', 0.0),
                (self.duration_spin, 'duration', 0.0),
                (self.sequence_length_spin, 'sequence_length', 5),
                (self.kalman_smooth_spin, 'kalman_smooth', 0.5),
                (self.mv_clamp_range_spin, 'mv_clamp_range', 32.0)
            ]
            
            for widget, key, default in numeric_widgets:
                widget.blockSignals(True)
                if isinstance(default, int):
                    widget.setValue(int(self.settings.value(key, default)))
                else:
                    widget.setValue(float(self.settings.value(key, default)))
                widget.blockSignals(False)
            
            # Load combo settings
            combo_settings = [
                (self.model_combo, 'model', 'videoflow', ['videoflow', 'memflow']),
                (self.dataset_combo, 'dataset', 'sintel', ['sintel', 'things', 'kitti']),
                (self.vf_architecture_combo, 'vf_architecture', 'mof', ['mof', 'bof']),
                (self.vf_variant_combo, 'vf_variant', 'standard', ['standard', 'noise']),
                (self.device_combo, 'device', 'cuda', ['cpu', 'cuda']),
                (self.flow_format_combo, 'flow_format', 'motion-vectors-rgb8', ['gamedev', 'hsv', 'torchvision', 'motion-vectors-rg8', 'motion-vectors-rgb8']),
                (self.save_flow_combo, 'save_flow', 'npz', ['none', 'npz', 'flo', 'both']),
                (self.time_control_combo, 'time_control', 'Control by frame', ['Control by frame', 'Control by time'])
            ]
            
            for combo, key, default, valid_values in combo_settings:
                combo.blockSignals(True)
                value = self.settings.value(key, default)
                if value in valid_values:
                    combo.setCurrentText(value)
                combo.blockSignals(False)
            
            # Apply time control visibility changes manually (without triggering save)
            self.update_time_control_visibility()
            
            # Apply model-specific UI state changes without triggering save
            current_model = self.model_combo.currentText()
            if current_model == 'videoflow':
                self.vf_architecture_combo.setEnabled(True)
                self.vf_variant_combo.setEnabled(True)
            else:  # memflow
                self.vf_architecture_combo.setEnabled(False)
                self.vf_variant_combo.setEnabled(False)
            
            # Apply motion vectors clamp range visibility
            self.update_mv_clamp_range_visibility()
            
        finally:
            self.blockSignals(False)
        
        # Defer video loading to prevent blocking UI during startup
        video_path = self.input_video_edit.text()
        if video_path and os.path.exists(video_path):
            # Use QTimer to load video after UI is fully initialized
            QTimer.singleShot(100, lambda: self.load_video(video_path))

    def save_settings(self):
        """Save current settings"""
        # Skip saving during initialization to prevent unnecessary I/O
        if not hasattr(self, '_initialization_complete') or not self._initialization_complete:
            return
            
        # Save file paths
        self.settings.setValue('input_video', self.input_video_edit.text())
        self.settings.setValue('output_path', self.output_path_edit.text())
        self.settings.setValue('flow_cache', self.flow_cache_edit.text())
        self.settings.setValue('flow_input', self.flow_input_edit.text())
        
        # Save processing settings
        for key, widget in self.flag_widgets.items():
            self.settings.setValue(f'flag_{key}', widget.isChecked())
        
        # Save numeric settings
        self.settings.setValue('start_frame', self.start_frame_spin.value())
        self.settings.setValue('max_frames', self.max_frames_spin.value())
        self.settings.setValue('start_time', self.start_time_spin.value())
        self.settings.setValue('duration', self.duration_spin.value())
        self.settings.setValue('sequence_length', self.sequence_length_spin.value())
        self.settings.setValue('kalman_smooth', self.kalman_smooth_spin.value())
        self.settings.setValue('mv_clamp_range', self.mv_clamp_range_spin.value())
        
        # Save combo settings
        self.settings.setValue('model', self.model_combo.currentText())
        self.settings.setValue('dataset', self.dataset_combo.currentText())
        self.settings.setValue('vf_architecture', self.vf_architecture_combo.currentText())
        self.settings.setValue('vf_variant', self.vf_variant_combo.currentText())
        self.settings.setValue('device', self.device_combo.currentText())
        self.settings.setValue('flow_format', self.flow_format_combo.currentText())
        self.settings.setValue('save_flow', self.save_flow_combo.currentText())
        self.settings.setValue('time_control', self.time_control_combo.currentText())

    def on_setting_changed(self):
        """Called when any setting changes"""
        self.save_settings()
        self.command_timer.start(100)  # Update command after 100ms delay
        
    def on_model_changed(self):
        """Handle model selection changes to show/hide model-specific options"""
        current_model = self.model_combo.currentText()
        
        # Show/hide VideoFlow-specific options
        if current_model == 'videoflow':
            self.vf_architecture_combo.setEnabled(True)
            self.vf_variant_combo.setEnabled(True)
        else:  # memflow
            self.vf_architecture_combo.setEnabled(False)
            self.vf_variant_combo.setEnabled(False)
        
        # Also trigger general setting change
        self.on_setting_changed()
    
    def on_flow_format_changed(self):
        """Handle flow format changes to show/hide motion vectors clamp range"""
        self.update_mv_clamp_range_visibility()
        self.on_setting_changed()
    
    def update_mv_clamp_range_visibility(self):
        """Show/hide motion vectors clamp range based on selected format"""
        flow_format = self.flow_format_combo.currentText()
        is_motion_vectors = flow_format.startswith('motion-vectors')
        
        self.mv_clamp_range_label.setVisible(is_motion_vectors)
        self.mv_clamp_range_spin.setVisible(is_motion_vectors)
        
    def update_time_control_visibility(self):
        """Update time control visibility without triggering signals"""
        control_type = self.time_control_combo.currentText()
        
        if control_type == 'Control by frame':
            # Show frame controls
            self.frame_start_label.show()
            self.start_frame_spin.show()
            self.frame_max_label.show()
            self.max_frames_spin.show()
            
            # Hide time controls
            self.time_start_label.hide()
            self.start_time_spin.hide()
            self.time_duration_label.hide()
            self.duration_spin.hide()
        else:  # Control by time
            # Hide frame controls
            self.frame_start_label.hide()
            self.start_frame_spin.hide()
            self.frame_max_label.hide()
            self.max_frames_spin.hide()
            
            # Show time controls
            self.time_start_label.show()
            self.start_time_spin.show()
            self.time_duration_label.show()
            self.duration_spin.show()

    def on_time_control_changed(self):
        """Called when time control type changes"""
        self.update_time_control_visibility()
        self.save_settings()
        self.command_timer.start(100)
        
    def on_flow_cache_changed(self):
        """Called when flow cache path changes"""
        self.flow_cache_path = self.flow_cache_edit.text()
        self.save_settings()
        self.check_flow_cache()
        self.command_timer.start(100)



    def browse_input_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Input Video", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm *.flv);;All Files (*)"
        )
        if file_path:
            self.input_video_edit.setText(file_path)
            self.load_video(file_path)

    def browse_output_path(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_path_edit.setText(dir_path)

    def browse_flow_cache(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Flow Cache Directory")
        if dir_path:
            self.flow_cache_edit.setText(dir_path)

    def browse_flow_input(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Flow Input Video", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm *.flv);;All Files (*)"
        )
        if file_path:
            self.flow_input_edit.setText(file_path)

    def load_video(self, video_path):
        """Load video for preview"""
        self.video_path = video_path
        
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.quit()
            self.video_thread.wait()
        
        self.video_thread = VideoThread(video_path)
        self.video_thread.frame_loaded.connect(self.on_video_loaded)
        self.video_thread.start()



    def check_flow_cache(self):
        if not self.flow_cache_path or self.total_frames == 0:
            self.flow_cache_status_label.setText("")
            return

        # Import VideoFlowProcessor only when needed
        if self.flow_processor_instance is None:
            from flow_processor import VideoFlowProcessor
            self.flow_processor_instance = VideoFlowProcessor(device='cpu')

        cache_exists, _, missing = self.flow_processor_instance.check_flow_cache_exists(
            self.flow_cache_path, self.total_frames
        )
        lods_exist = self.flow_processor_instance.check_flow_lods_exist(
            self.flow_cache_path, self.total_frames
        )
        
        status_text = []
        if cache_exists:
            status_text.append("<font color='green'>Cache is complete</font>")
        else:
            status_text.append(f"<font color='orange'>Cache is incomplete ({len(missing)} frames missing)</font>")

        if lods_exist:
            status_text.append("<font color='green'>LODs found</font>")
        else:
            status_text.append("<font color='orange'>LODs not found</font>")
            
        self.flow_cache_status_label.setText(", ".join(status_text))


    def on_video_loaded(self, frame, total_frames, fps):
        """Called when video is loaded"""
        self.current_frame = frame
        self.total_frames = total_frames
        self.fps = fps
        self.current_frame_number = 0
        
        # Update UI
        self.display_frame(frame)
        self.frame_info_label.setText(f"Total frames: {total_frames}, FPS: {fps:.2f}, Duration: {total_frames/fps:.2f}s")
        
        # Show video controls and stats
        self.pause_play_btn.show()
        self.restart_btn.show()
        self.video_stats_label.show()
        self.update_video_layout()
        
        # Auto-start video playback
        self.is_video_playing = True
        self.pause_play_btn.setIcon(qta.icon('fa5s.pause'))
        
        # Initialize and show stats
        self.update_video_stats()
        
        # Create and start timer for playback
        if not hasattr(self, 'video_timer'):
            self.video_timer = QTimer()
            self.video_timer.timeout.connect(self.update_video_frame)
        
        self.video_timer.start(int(1000 / self.fps))
        
        # Initialize range values for processing
        self.range_start = 0
        self.range_end = 100
        
        # Check flow cache
        self.check_flow_cache()

    def display_frame(self, frame):
        """Display frame in video label"""
        if frame is None:
            return
            
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.video_container.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)
        self.update_video_layout()
        
        # Update stats if video is loaded (for manual frame changes)
        if hasattr(self, 'total_frames') and self.total_frames > 0:
            self.update_video_stats()

    def update_video_layout(self):
        """Update video container layout and button position"""
        if not hasattr(self, 'video_container'):
            return
            
        # Resize video label to fill container
        self.video_label.resize(self.video_container.size())
        
        container_width = self.video_container.width()
        container_height = self.video_container.height()
        
        # Position restart button in top-right corner
        restart_x = container_width - self.restart_btn.width() - 10
        restart_y = 10
        self.restart_btn.move(restart_x, restart_y)
        
        # Position pause button next to restart button
        pause_x = restart_x - self.pause_play_btn.width() - 5
        pause_y = 10
        self.pause_play_btn.move(pause_x, pause_y)
        
        # Position video statistics in bottom-right corner
        self.video_stats_label.adjustSize()  # Resize to fit content
        stats_x = container_width - self.video_stats_label.width() - 10
        stats_y = container_height - self.video_stats_label.height() - 10
        self.video_stats_label.move(stats_x, stats_y)

    def resizeEvent(self, event):
        """Handle window resize"""
        super().resizeEvent(event)
        if hasattr(self, 'video_container'):
            QTimer.singleShot(1, self.update_video_layout)

    def toggle_video_playback(self):
        """Toggle video playback"""
        if not self.video_path or self.total_frames == 0:
            return
            
        if self.is_video_playing:
            # Pause playback
            if hasattr(self, 'video_timer'):
                self.video_timer.stop()
            self.is_video_playing = False
            self.pause_play_btn.setIcon(qta.icon('fa5s.play'))
        else:
            # Start playback
            self.is_video_playing = True
            self.pause_play_btn.setIcon(qta.icon('fa5s.pause'))
            
            # Create timer for playback
            if not hasattr(self, 'video_timer'):
                self.video_timer = QTimer()
                self.video_timer.timeout.connect(self.update_video_frame)
            
            self.video_timer.start(int(1000 / self.fps))  # Timer interval based on video FPS

    def restart_video(self):
        """Restart video from beginning"""
        if not self.video_path or self.total_frames == 0:
            return
            
        self.current_frame_number = 0
        self.load_and_display_frame(0)
        self.update_video_stats()
        
        # If video was playing, continue playing from beginning
        if self.is_video_playing and hasattr(self, 'video_timer'):
            self.video_timer.start(int(1000 / self.fps))

    def update_video_frame(self):
        """Update the current video frame during playback"""
        if not self.is_video_playing:
            return
            
        # Load and display current frame
        self.load_and_display_frame(self.current_frame_number)
        
        # Update video statistics
        self.update_video_stats()
        
        # Move to next frame
        self.current_frame_number += 1
        
        # Loop back to beginning when reaching the end
        if self.current_frame_number >= self.total_frames:
            self.current_frame_number = 0

    def update_video_stats(self):
        """Update video statistics display"""
        if not hasattr(self, 'video_stats_label') or self.total_frames == 0:
            return
            
        current_time = self.current_frame_number / self.fps
        total_time = self.total_frames / self.fps
        
        stats_text = f"Frame: {self.current_frame_number + 1}/{self.total_frames}\nTime: {current_time:.2f}s / {total_time:.2f}s"
        self.video_stats_label.setText(stats_text)

    def load_and_display_frame(self, frame_number):
        """Load and display a specific frame"""
        if not self.video_path or not os.path.exists(self.video_path):
            return
            
        try:
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                cap.release()
                return
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.display_frame(frame)
                
            cap.release()
        except Exception as e:
            print(f"Error loading frame {frame_number}: {e}")

    def generate_command(self, extra_flags=None):
        """Generate the flow_processor command based on current settings"""
        if not self.input_video_edit.text():
            return ""
        
        cmd_parts = ["python", "flow_processor.py"]
        
        # Input video (required)
        cmd_parts.extend(["--input", f'"{self.input_video_edit.text()}"'])
        
        # Output path
        if self.output_path_edit.text():
            cmd_parts.extend(["--output", f'"{self.output_path_edit.text()}"'])
        
        # Flow cache
        if self.flow_cache_edit.text():
            cmd_parts.extend(["--flow-cache", f'"{self.flow_cache_edit.text()}"'])
        
        # Flow input video
        if self.flow_input_edit.text():
            cmd_parts.extend(["--flow-input", f'"{self.flow_input_edit.text()}"'])
        
        # Boolean flags
        for key, widget in self.flag_widgets.items():
            if widget.isChecked():
                cmd_parts.append(f"--{key.replace('_', '-')}")
        
        # Time control parameters based on selected type
        control_type = self.time_control_combo.currentText()
        
        if control_type == 'Control by frame':
            # Use frame-based parameters
            if self.start_frame_spin.value() > 0:
                cmd_parts.extend(["--start-frame", str(int(self.start_frame_spin.value()))])
            
            if self.max_frames_spin.value() > 0:
                cmd_parts.extend(["--frames", str(int(self.max_frames_spin.value()))])
        else:  # Control by time
            # Use time-based parameters
            if self.start_time_spin.value() > 0:
                cmd_parts.extend(["--start-time", str(int(self.start_time_spin.value()))])
            
            if self.duration_spin.value() > 0:
                cmd_parts.extend(["--duration", str(int(self.duration_spin.value()))])
        
        # Model
        if self.model_combo.currentText() != 'videoflow':
            cmd_parts.extend(["--model", self.model_combo.currentText()])
        
        # Dataset selection (for both MemFlow and VideoFlow)
        if self.dataset_combo.currentText() != 'sintel':
            if self.model_combo.currentText() == 'memflow':
                cmd_parts.extend(["--stage", self.dataset_combo.currentText()])
            else:  # videoflow
                cmd_parts.extend(["--vf-dataset", self.dataset_combo.currentText()])
        
        # VideoFlow-specific options
        if self.model_combo.currentText() == 'videoflow':
            if self.vf_architecture_combo.currentText() != 'mof':
                cmd_parts.extend(["--vf-architecture", self.vf_architecture_combo.currentText()])
            
            if self.vf_variant_combo.currentText() != 'standard':
                cmd_parts.extend(["--vf-variant", self.vf_variant_combo.currentText()])
        
        # Device
        if self.device_combo.currentText() != 'cuda':
            cmd_parts.extend(["--device", self.device_combo.currentText()])
        
        # Flow format
        if self.flow_format_combo.currentText() != 'gamedev':
            cmd_parts.extend(["--flow-format", self.flow_format_combo.currentText()])
        
        # Motion vectors clamp range (only for motion-vectors formats)
        flow_format = self.flow_format_combo.currentText()
        if flow_format.startswith('motion-vectors') and self.mv_clamp_range_spin.value() != 32.0:
            cmd_parts.extend(["--motion-vectors-clamp-range", f"{self.mv_clamp_range_spin.value():.1f}"])
        
        # Save flow format
        if self.save_flow_combo.currentText() != 'none':
            cmd_parts.extend(["--save-flow", self.save_flow_combo.currentText()])
        
        # Sequence length
        if self.sequence_length_spin.value() != 5:
            cmd_parts.extend(["--sequence-length", str(self.sequence_length_spin.value())])
        
        # Kalman smoothing
        if self.kalman_smooth_spin.value() != 0.5:
            cmd_parts.extend(["--kalman-smooth", f"{self.kalman_smooth_spin.value():.2f}"])
        
        # Add extra flags if provided
        if extra_flags:
            cmd_parts.extend(extra_flags)
        
        return " ".join(cmd_parts)

    def update_command_preview(self):
        """Update the command preview"""
        # Skip during initialization to prevent unnecessary processing
        if not hasattr(self, '_initialization_complete') or not self._initialization_complete:
            return
            
        command = self.generate_command()
        self.command_preview.setPlainText(command)

    def run_interactive(self):
        """Run the command in interactive mode"""
        if not self.input_video_edit.text():
            InfoBar.warning(
                title="Warning",
                content="Please select an input video first",
                orient=Qt.Orientation.Horizontal,
                isClosable=True,
                duration=3000,
                position=InfoBarPosition.TOP,
                parent=self
            )
            return
        
        # Generate command with interactive flag
        command = self.generate_command(extra_flags=["--interactive"])
        
        InfoBar.info(
            title="Interactive Mode",
            content="Starting interactive flow visualizer in PowerShell",
            orient=Qt.Orientation.Horizontal,
            isClosable=True,
            duration=2000,
            position=InfoBarPosition.TOP,
            parent=self
        )
        
        # Run interactive command in separate PowerShell window
        ps_command = f'Write-Host "Starting Interactive Flow Visualizer..." -ForegroundColor Green; Write-Host "Command: {command}" -ForegroundColor Yellow; Write-Host ""; {command}; Write-Host ""; Write-Host "Interactive session completed. Press any key to close..." -ForegroundColor Green; $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")'
        subprocess.Popen([
            'powershell.exe', 
            '-Command', 
            ps_command
        ], creationflags=subprocess.CREATE_NEW_CONSOLE)

    def run_processing(self):
        """Run the processing command"""
        command = self.generate_command()
        if not command:
            InfoBar.warning(
                title="Warning", 
                content="Please select an input video first",
                orient=Qt.Orientation.Horizontal,
                isClosable=True,
                duration=3000,
                position=InfoBarPosition.TOP,
                parent=self
            )
            return
        
        InfoBar.info(
            title="Processing Started",
            content="Processing started in separate PowerShell window",
            orient=Qt.Orientation.Horizontal,
            isClosable=True,
            duration=3000,
            position=InfoBarPosition.TOP,
            parent=self
        )
        
        # Run processing in separate PowerShell window
        ps_command = f'Write-Host "Starting VideoFlow processing..." -ForegroundColor Green; Write-Host "Command: {command}" -ForegroundColor Yellow; Write-Host ""; {command}; Write-Host ""; Write-Host "Processing completed. Press any key to close..." -ForegroundColor Green; $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")'
        subprocess.Popen([
            'powershell.exe', 
            '-Command', 
            ps_command
        ], creationflags=subprocess.CREATE_NEW_CONSOLE)



    def closeEvent(self, event):
        """Handle application close"""
        self.save_settings()
        
        # Stop video playback
        if hasattr(self, 'video_timer') and self.video_timer.isActive():
            self.video_timer.stop()
        self.is_video_playing = False
        
        # Clean up threads
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.quit()
            self.video_thread.wait()
        
        if self.process_thread and self.process_thread.isRunning():
            self.process_thread.quit()
            self.process_thread.wait()
        
        event.accept()


def main():
    # Application already created at the top
    window = FlowRunnerApp()
    window.show()
    
    try:
        sys.exit(app.exec())
    except SystemExit:
        pass


if __name__ == "__main__":
    main() 