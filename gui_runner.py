import sys
import os

# Path fix for portable execution: Add the script's directory to sys.path
# This is necessary because the portable Python environment may not include the CWD.
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from PyQt6.QtWidgets import QApplication

# Create the application instance first to avoid initialization errors
app = QApplication(sys.argv)

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
from PyQt6.QtGui import QPixmap, QImage, QIcon, QFont, QPainter, QPen, QColor

from qfluentwidgets import (setTheme, Theme, TitleLabel, SubtitleLabel, LineEdit, PushButton,
                            InfoBar, InfoBarPosition, CheckBox, ComboBox, DoubleSpinBox, SpinBox,
                            BodyLabel, CardWidget, HyperlinkButton, ProgressBar, TextEdit, ToolButton)
import qtawesome as qta

# Do not import this at module level
# from flow_processor import VideoFlowProcessor


class CollapsibleCard(CardWidget):
    """A collapsible card widget"""
    def __init__(self, title, collapsed=False, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Header with toggle button
        header_layout = QHBoxLayout()
        self.toggle_btn = ToolButton()
        self.toggle_btn.clicked.connect(self.toggle_content)
        
        self.title_label = BodyLabel(title)
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.toggle_btn)
        
        header_widget = QWidget()
        header_widget.setLayout(header_layout)
        self.layout.addWidget(header_widget)
        
        # Content area
        self.content_widget = QWidget()
        self.content_layout = QGridLayout(self.content_widget)
        self.layout.addWidget(self.content_widget)
        
        # Set initial state
        self.is_expanded = not collapsed
        self.content_widget.setVisible(self.is_expanded)
        
        if self.is_expanded:
            self.toggle_btn.setIcon(qta.icon('fa5s.chevron-down'))
        else:
            self.toggle_btn.setIcon(qta.icon('fa5s.chevron-right'))
    
    def toggle_content(self):
        self.is_expanded = not self.is_expanded
        self.content_widget.setVisible(self.is_expanded)
        
        if self.is_expanded:
            self.toggle_btn.setIcon(qta.icon('fa5s.chevron-down'))
        else:
            self.toggle_btn.setIcon(qta.icon('fa5s.chevron-right'))
    
    def set_collapsed(self, collapsed=True):
        if self.is_expanded == (not collapsed):
            return  # Already in desired state
        self.toggle_content()


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
        
        # Import and create instance only when needed
        self.flow_processor_instance = None
        
        # Variables to store previous values when flow_only is enabled
        self._previous_taa_state = None
        self._previous_flow_input_text = None

        # --- UI Initialization ---
        self._initialization_complete = False
        self.init_ui()
        self.load_settings()
        
        # Mark initialization as complete
        self._initialization_complete = True
        
        # Update command preview and previews now that initialization is complete
        # Note: preview functions will check if video is loaded before showing previews
        self.update_command_preview()
        self.update_cache_preview()
        self.update_flow_input_preview()
        self.update_output_preview()  # Move to end to ensure all other previews are updated first
        # Update status badges
        self.update_output_status()
        self.update_cache_status()

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
            ('flow_only', False, 'Flow only', 'Output only optical flow without video composition'),
            ('taa', True, 'TAA', 'Apply Temporal Anti-Aliasing for smoother results'),
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
            
            # Add special handler for flow_only checkbox
            if key == 'flow_only':
                checkbox.stateChanged.connect(self.update_flow_only_dependent_controls)
            
            # Add special handler for tile checkbox to update video display
            if key == 'tile':
                checkbox.stateChanged.connect(self.update_tile_mode_display)
            
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
        self.time_control_combo.currentTextChanged.connect(self.update_output_preview)
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

        # Flow Encoding card (collapsible, collapsed by default)
        self.encoding_card = CollapsibleCard("Flow Encoding", collapsed=True)
        
        # Flow format
        self.encoding_card.content_layout.addWidget(BodyLabel("Flow format:"), 0, 0)
        self.flow_format_combo = ComboBox()
        self.flow_format_combo.addItems(['gamedev', 'hsv', 'torchvision', 'motion-vectors-rg8', 'motion-vectors-rgb8'])
        self.flow_format_combo.setToolTip("Output format for optical flow visualization")
        self.flow_format_combo.currentTextChanged.connect(self.on_flow_format_changed)
        self.encoding_card.content_layout.addWidget(self.flow_format_combo, 0, 1)
        
        # Motion vectors clamp range (initially hidden)
        self.encoding_card.content_layout.addWidget(BodyLabel("MV clamp range:"), 1, 0)
        self.mv_clamp_range_spin = DoubleSpinBox()
        self.mv_clamp_range_spin.setRange(1.0, 512.0)
        self.mv_clamp_range_spin.setValue(32.0)
        self.mv_clamp_range_spin.setSingleStep(1.0)
        self.mv_clamp_range_spin.setDecimals(1)
        self.mv_clamp_range_spin.setToolTip("Clamp range for motion vectors encoding formats")
        self.mv_clamp_range_spin.valueChanged.connect(self.on_setting_changed)
        self.encoding_card.content_layout.addWidget(self.mv_clamp_range_spin, 1, 1)
        
        # Store the motion vectors clamp range label and spin for hiding/showing
        self.mv_clamp_range_label = self.encoding_card.content_layout.itemAtPosition(1, 0).widget()
        self.update_mv_clamp_range_visibility()
        
        layout.addWidget(self.encoding_card)

        # Model Parameters card (collapsible, collapsed by default)
        self.model_card = CollapsibleCard("Model Parameters", collapsed=True)
        
        # Model selection
        self.model_card.content_layout.addWidget(BodyLabel("Model:"), 0, 0)
        self.model_combo = ComboBox()
        self.model_combo.addItems(['videoflow', 'memflow'])
        self.model_combo.setCurrentText('videoflow')
        self.model_combo.setToolTip("Optical flow model: VideoFlow (MOF) or MemFlow")
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        self.model_card.content_layout.addWidget(self.model_combo, 0, 1)
        
        # Dataset selection (for both models)
        self.model_card.content_layout.addWidget(BodyLabel("Dataset:"), 1, 0)
        self.dataset_combo = ComboBox()
        self.dataset_combo.addItems(['sintel', 'things', 'kitti'])
        self.dataset_combo.setCurrentText('things')  # Default to things
        self.dataset_combo.setToolTip("Training dataset for the model")
        self.dataset_combo.currentTextChanged.connect(self.on_setting_changed)
        self.model_card.content_layout.addWidget(self.dataset_combo, 1, 1)
        
        # VideoFlow architecture selection (only for VideoFlow)
        self.model_card.content_layout.addWidget(BodyLabel("VF Architecture:"), 2, 0)
        self.vf_architecture_combo = ComboBox()
        self.vf_architecture_combo.addItems(['mof', 'bof'])
        self.vf_architecture_combo.setCurrentText('mof')
        self.vf_architecture_combo.setToolTip("VideoFlow architecture: MOF (MOFNet) or BOF (BOFNet)")
        self.vf_architecture_combo.currentTextChanged.connect(self.on_setting_changed)
        self.model_card.content_layout.addWidget(self.vf_architecture_combo, 2, 1)
        
        # VideoFlow variant selection (only for VideoFlow)
        self.model_card.content_layout.addWidget(BodyLabel("VF Variant:"), 3, 0)
        self.vf_variant_combo = ComboBox()
        self.vf_variant_combo.addItems(['standard', 'noise'])
        self.vf_variant_combo.setCurrentText('noise')  # Default to noise
        self.vf_variant_combo.setToolTip("VideoFlow variant: standard or noise (things_288960noise)")
        self.vf_variant_combo.currentTextChanged.connect(self.on_setting_changed)
        self.model_card.content_layout.addWidget(self.vf_variant_combo, 3, 1)
        
        # Device
        self.model_card.content_layout.addWidget(BodyLabel("Device:"), 4, 0)
        self.device_combo = ComboBox()
        self.device_combo.addItems(['cuda', 'cpu'])
        self.device_combo.setCurrentText('cuda')
        self.device_combo.setToolTip("Processing device (CPU or CUDA GPU)")
        self.device_combo.currentTextChanged.connect(self.on_setting_changed)
        self.model_card.content_layout.addWidget(self.device_combo, 4, 1)
        
        # Sequence length
        self.model_card.content_layout.addWidget(BodyLabel("Sequence length:"), 5, 0)
        self.sequence_length_spin = SpinBox()
        self.sequence_length_spin.setRange(3, 20)
        self.sequence_length_spin.setValue(5)
        self.sequence_length_spin.setToolTip("Number of frames in processing sequence")
        self.sequence_length_spin.valueChanged.connect(self.on_setting_changed)
        self.model_card.content_layout.addWidget(self.sequence_length_spin, 5, 1)
        
        layout.addWidget(self.model_card)

        # General Parameters card (collapsible, collapsed by default)
        self.general_card = CollapsibleCard("General", collapsed=True)
        
        # Save flow format
        self.general_card.content_layout.addWidget(BodyLabel("Save flow:"), 0, 0)
        self.save_flow_combo = ComboBox()
        self.save_flow_combo.addItems(['none', 'npz', 'flo', 'both'])
        self.save_flow_combo.setCurrentText('npz')
        self.save_flow_combo.setToolTip("Format for saving flow data: flo (Middlebury), npz (NumPy), both, none (don't save)")
        self.save_flow_combo.currentTextChanged.connect(self.on_setting_changed)
        self.general_card.content_layout.addWidget(self.save_flow_combo, 0, 1)
        
        layout.addWidget(self.general_card)



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
        self.input_video_edit.textChanged.connect(self.update_output_preview)
        self.input_video_edit.textChanged.connect(self.update_cache_preview)
        self.input_video_edit.textChanged.connect(self.update_flow_input_preview)
        self.input_video_edit.textChanged.connect(self.update_output_status)
        self.input_video_edit.textChanged.connect(self.update_cache_status)
        io_layout.addWidget(self.input_video_edit, 1, 1)
        
        browse_input_btn = PushButton("...")
        browse_input_btn.clicked.connect(self.browse_input_video)
        io_layout.addWidget(browse_input_btn, 1, 2)
        
        # Output path
        io_layout.addWidget(BodyLabel("Output path:"), 2, 0)
        self.output_path_edit = LineEdit()
        self.output_path_edit.setPlaceholderText("Leave empty for automatic naming")
        self.output_path_edit.setToolTip("Output directory for processed video (leave empty for 'results' folder)")
        self.output_path_edit.textChanged.connect(self.on_setting_changed)
        self.output_path_edit.textChanged.connect(self.update_output_preview)
        self.output_path_edit.textChanged.connect(self.update_flow_input_preview)
        self.output_path_edit.textChanged.connect(self.update_output_status)
        io_layout.addWidget(self.output_path_edit, 2, 1)
        
        # Browse output button
        browse_output_btn = PushButton("...")
        browse_output_btn.setFixedWidth(40)
        browse_output_btn.clicked.connect(self.browse_output_path)
        io_layout.addWidget(browse_output_btn, 2, 2)

        # Output file status badge
        self.output_status_label = QLabel()
        self.output_status_label.setFixedSize(20, 20)
        self.output_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_status_label.setStyleSheet("QLabel { border-radius: 10px; }")
        self.update_output_status()
        io_layout.addWidget(self.output_status_label, 2, 3)

        # Flow cache
        io_layout.addWidget(BodyLabel("Flow cache:"), 3, 0)
        self.flow_cache_edit = LineEdit()
        self.flow_cache_edit.setToolTip("Path to flow cache directory (optional)")
        self.flow_cache_edit.textChanged.connect(self.on_flow_cache_changed)
        io_layout.addWidget(self.flow_cache_edit, 3, 1)
        
        # Browse cache button
        browse_cache_btn = PushButton("...")
        browse_cache_btn.setFixedWidth(40)
        browse_cache_btn.clicked.connect(self.browse_flow_cache)
        io_layout.addWidget(browse_cache_btn, 3, 2)

        # Flow cache status badge
        self.cache_status_label = QLabel()
        self.cache_status_label.setFixedSize(20, 20)
        self.cache_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cache_status_label.setStyleSheet("QLabel { border-radius: 10px; }")
        self.update_cache_status()
        io_layout.addWidget(self.cache_status_label, 3, 3)
        
        # Flow input video (for TAA comparison)
        self.flow_input_label = BodyLabel("Flow input video:")
        io_layout.addWidget(self.flow_input_label, 4, 0)
        self.flow_input_edit = LineEdit()
        self.flow_input_edit.setToolTip("Path to video with encoded motion vectors in bottom half (for TAA comparison)")
        self.flow_input_edit.textChanged.connect(self.on_setting_changed)
        io_layout.addWidget(self.flow_input_edit, 4, 1)
        
        self.browse_flow_input_btn = PushButton("...")
        self.browse_flow_input_btn.clicked.connect(self.browse_flow_input)
        io_layout.addWidget(self.browse_flow_input_btn, 4, 2)
        
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
            self.output_path_edit.setText(self.settings.value('output_path', 'results'))
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
                (self.dataset_combo, 'dataset', 'things', ['sintel', 'things', 'kitti']),
                (self.vf_architecture_combo, 'vf_architecture', 'mof', ['mof', 'bof']),
                (self.vf_variant_combo, 'vf_variant', 'noise', ['standard', 'noise']),
                (self.device_combo, 'device', 'cuda', ['cpu', 'cuda']),
                (self.flow_format_combo, 'flow_format', 'gamedev', ['gamedev', 'hsv', 'torchvision', 'motion-vectors-rg8', 'motion-vectors-rgb8']),
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
            
            # Apply flow_only dependent controls state
            self.update_flow_only_dependent_controls()
            
        finally:
            self.blockSignals(False)
        
        # Defer video loading to prevent blocking UI during startup
        video_path = self.input_video_edit.text()
        if video_path and os.path.exists(video_path):
            # Use QTimer to load video after UI is fully initialized
            QTimer.singleShot(100, lambda: self.load_video(video_path))
        
        # Update output filename preview
        self.update_output_preview()

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
        # Reset flow processor instance so it gets recreated with new settings
        self.flow_processor_instance = None
        self.save_settings()
        self.command_timer.start(100)  # Update command after 100ms delay
        self.update_output_preview()  # Update filename preview
        self.update_cache_preview()  # Update cache preview
        self.update_flow_input_preview()  # Update flow input preview
        self.update_output_status()  # Update output file status badge
        self.update_cache_status()  # Update cache status badge
        
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
        
        # Reset flow processor instance due to model change
        self.flow_processor_instance = None
        
        # Update tile display since different models have different tile behaviors
        self.update_tile_mode_display()
        
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
    
    def update_flow_only_dependent_controls(self):
        """Update controls that depend on the flow_only flag"""
        if not self._initialization_complete:
            return
        
        flow_only = self.flag_widgets['flow_only'].isChecked()
        
        if flow_only:
            # Store previous state before disabling
            self._previous_taa_state = self.flag_widgets['taa'].isChecked()
            self._previous_flow_input_text = self.flow_input_edit.text()
            
            # Disable TAA when flow_only is enabled
            self.flag_widgets['taa'].setChecked(False)
            self.flag_widgets['taa'].setEnabled(False)
            
            # Disable flow input when flow_only is enabled
            self.flow_input_edit.clear()
            self.flow_input_edit.setEnabled(False)
            self.flow_input_label.setEnabled(False)
            self.browse_flow_input_btn.setEnabled(False)
        else:
            # Re-enable TAA when flow_only is disabled
            self.flag_widgets['taa'].setEnabled(True)
            
            # Re-enable flow input when flow_only is disabled
            self.flow_input_edit.setEnabled(True)
            self.flow_input_label.setEnabled(True)
            self.browse_flow_input_btn.setEnabled(True)
            
            # Restore previous state if available
            if self._previous_taa_state is not None:
                self.flag_widgets['taa'].setChecked(self._previous_taa_state)
                self._previous_taa_state = None
            
            if self._previous_flow_input_text is not None:
                self.flow_input_edit.setText(self._previous_flow_input_text)
                self._previous_flow_input_text = None
        
        # Update related UI elements
        self.update_output_preview()
        self.update_cache_preview()
        self.update_flow_input_preview()
        self.update_output_status()
        self.update_cache_status()

    def update_tile_mode_display(self):
        """Update video display when tile mode is toggled"""
        if not self._initialization_complete:
            return
        
        # If there's a current frame loaded, redisplay it to show/hide tile grid
        # Also check that flag_widgets is initialized to avoid errors during startup
        if (hasattr(self, 'current_frame') and self.current_frame is not None and 
            hasattr(self, 'flag_widgets') and 'tile' in self.flag_widgets):
            self.display_frame(self.current_frame)
        
        # Also update command preview and other related displays
        self.update_command_preview()
        self.update_output_preview()
        self.update_cache_preview()
        self.update_output_status()
        self.update_cache_status()

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
        # Update previews when time control changes
        self.on_setting_changed()
        
    def on_flow_cache_changed(self):
        """Called when flow cache path changes"""
        self.flow_cache_path = self.flow_cache_edit.text()
        self.save_settings()
        self.check_flow_cache()
        self.command_timer.start(100)
        # Update cache preview when cache field changes
        self.update_cache_preview()
        self.update_cache_status()  # Update cache status badge



    def browse_input_video(self):
        # Try to use directory from current input or default to current directory
        start_dir = ""
        current_path = self.input_video_edit.text()
        if current_path and os.path.exists(current_path):
            start_dir = os.path.dirname(current_path)
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Input Video", start_dir, 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm *.flv);;All Files (*)"
        )
        if file_path:
            self.input_video_edit.setText(file_path)
            self.load_video(file_path)

    def browse_output_path(self):
        # Try to use path from placeholder if it exists
        start_dir = ""
        placeholder = self.output_path_edit.placeholderText()
        
        if placeholder and placeholder != "Load video to see filename preview":
            # Extract directory from the placeholder path
            placeholder_dir = os.path.dirname(placeholder)
            if os.path.exists(placeholder_dir):
                start_dir = placeholder_dir
            elif placeholder.startswith("results/") or "results" in placeholder:
                # Default to results directory
                results_dir = os.path.abspath("results")
                if os.path.exists(results_dir):
                    start_dir = results_dir
        
        # If no good default found, use current text or current directory
        if not start_dir:
            current_path = self.output_path_edit.text()
            if current_path and os.path.exists(current_path):
                start_dir = current_path
        
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory", start_dir)
        if dir_path:
            self.output_path_edit.setText(dir_path)
            self.update_output_status()  # Update status after path change

    def browse_flow_cache(self):
        # Try to use path from placeholder if it exists
        start_dir = ""
        placeholder = self.flow_cache_edit.placeholderText()
        
        if placeholder and placeholder != "Load video to see cache preview":
            # Check if placeholder path exists
            if os.path.exists(placeholder):
                if os.path.isdir(placeholder):
                    start_dir = placeholder
                else:
                    start_dir = os.path.dirname(placeholder)
            else:
                # Try parent directory of placeholder
                placeholder_parent = os.path.dirname(placeholder)
                if os.path.exists(placeholder_parent):
                    start_dir = placeholder_parent
        
        # If no good default found, use current text or input video directory
        if not start_dir:
            current_path = self.flow_cache_edit.text()
            if current_path and os.path.exists(current_path):
                start_dir = current_path
            else:
                # Default to input video directory
                input_path = self.input_video_edit.text()
                if input_path and os.path.exists(input_path):
                    start_dir = os.path.dirname(input_path)
        
        dir_path = QFileDialog.getExistingDirectory(self, "Select Flow Cache Directory", start_dir)
        if dir_path:
            self.flow_cache_edit.setText(dir_path)
            self.update_cache_status()  # Update status after path change

    def browse_flow_input(self):
        # Try to use path from placeholder if it exists
        start_dir = ""
        placeholder = self.flow_input_edit.placeholderText()
        
        if placeholder and placeholder not in ["Flow input disabled in flow-only mode", "Load video to see expected flow input preview"]:
            # Check if placeholder file exists
            if os.path.exists(placeholder):
                if os.path.isfile(placeholder):
                    # If exact file exists, pre-select it
                    file_path = placeholder
                    self.flow_input_edit.setText(file_path)
                    return
                else:
                    start_dir = placeholder
            else:
                # Try parent directory of placeholder
                placeholder_parent = os.path.dirname(placeholder)
                if os.path.exists(placeholder_parent):
                    start_dir = placeholder_parent
        
        # If no good default found, use current text, output directory, or input video directory
        if not start_dir:
            current_path = self.flow_input_edit.text()
            if current_path and os.path.exists(current_path):
                start_dir = os.path.dirname(current_path)
            else:
                # Try output directory
                output_dir = self.output_path_edit.text()
                if output_dir and os.path.exists(output_dir):
                    start_dir = output_dir
                else:
                    # Default to input video directory
                    input_path = self.input_video_edit.text()
                    if input_path and os.path.exists(input_path):
                        start_dir = os.path.dirname(input_path)
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Flow Input Video", start_dir, 
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
        if not self.video_path or self.total_frames == 0:
            self.flow_cache_status_label.setText("")
            return

        # Import VideoFlowProcessor only when needed
        if self.flow_processor_instance is None:
            from flow_processor import VideoFlowProcessor
            
            # Get current GUI parameters
            fast_mode = self.flag_widgets['fast'].isChecked()
            tile_mode = self.flag_widgets['tile'].isChecked()
            sequence_length = self.sequence_length_spin.value()
            flow_model = self.model_combo.currentText()
            
            # Get model-specific parameters
            if flow_model == 'videoflow':
                vf_dataset = self.dataset_combo.currentText()
                vf_architecture = self.vf_architecture_combo.currentText()
                vf_variant = self.vf_variant_combo.currentText()
                stage = 'sintel'  # Default for videoflow
            else:  # memflow
                vf_dataset = 'sintel'  # Default for memflow
                vf_architecture = 'mof'  # Default for memflow
                vf_variant = 'standard'  # Default for memflow
                stage = self.dataset_combo.currentText() # Use dataset combo for memflow
            
            # Initialize processor with current GUI settings
            self.flow_processor_instance = VideoFlowProcessor(
                device='cpu',
                fast_mode=fast_mode,
                tile_mode=tile_mode,
                sequence_length=sequence_length,
                flow_model=flow_model,
                stage=stage,
                vf_dataset=vf_dataset,
                vf_architecture=vf_architecture,
                vf_variant=vf_variant
            )

        # Get processing parameters from GUI
        start_frame = self.start_frame_spin.value()
        max_frames = self.max_frames_spin.value()
        
        # Generate the proper cache path based on current GUI parameters
        proper_cache_path = self.flow_processor_instance.generate_flow_cache_path(
            self.video_path,
            start_frame,
            max_frames,
            self.sequence_length_spin.value(),
            self.flag_widgets['fast'].isChecked(),
            self.flag_widgets['tile'].isChecked()
        )
        
        # Check if user has set a custom cache path
        custom_cache_path = self.flow_cache_edit.text().strip()
        
        # Use custom path if provided, otherwise use the generated path
        cache_path_to_check = custom_cache_path if custom_cache_path else proper_cache_path
        
        # Calculate actual frames to process based on video length and GUI settings
        actual_frames_to_process = min(max_frames, self.total_frames - start_frame) if max_frames > 0 else (self.total_frames - start_frame)
        
        cache_exists, _, missing = self.flow_processor_instance.check_flow_cache_exists(
            cache_path_to_check, actual_frames_to_process
        )
        lods_exist = self.flow_processor_instance.check_flow_lods_exist(
            cache_path_to_check, actual_frames_to_process
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
            
        # Add info about cache path if using custom path
        if custom_cache_path and custom_cache_path != proper_cache_path:
            status_text.append("<font color='blue'>Using custom cache path</font>")
        
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
        
        # Update output filename preview and cache preview
        self.update_output_preview()
        self.update_cache_preview()
        self.update_flow_input_preview()
        # Update status badges
        self.update_output_status()
        self.update_cache_status()

    def display_frame(self, frame):
        """Display frame in video label"""
        if frame is None:
            return
            
        # Store current frame for tile mode updates
        self.current_frame = frame
        
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
        
        # Apply tile grid overlay if tile mode is enabled
        if self.flag_widgets['tile'].isChecked():
            scaled_pixmap = self.draw_tile_grid_overlay(scaled_pixmap, width, height)
        
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

    def get_tile_grid_info(self, width, height):
        """
        Get tile grid information using the same logic as the selected model processor.
        This ensures consistency between GUI visualization and actual processing.
        
        Args:
            width, height: Original frame dimensions
            
        Returns:
            (tile_width, tile_height, cols, rows, tiles_info) or None if tile mode is disabled
        """
        if not self.flag_widgets['tile'].isChecked():
            return None
        
        # Get the selected model
        selected_model = self.model_combo.currentText()
        
        # Use the appropriate processor based on selected model
        try:
            if selected_model == 'memflow':
                from processing.memflow_processor import MemFlowProcessor
                return MemFlowProcessor.calculate_tile_grid(width, height)
            else:  # videoflow
                from processing.videoflow_processor import VideoFlowProcessor
                return VideoFlowProcessor.calculate_tile_grid(width, height)
        except Exception as e:
            print(f"Error calculating tile grid for {selected_model}: {e}")
            return None

    def draw_tile_grid_overlay(self, pixmap, original_width, original_height):
        """
        Draw tile grid overlay on the pixmap.
        
        Args:
            pixmap: QPixmap to draw on
            original_width, original_height: Original frame dimensions
            
        Returns:
            QPixmap with tile grid overlay
        """
        if not self.flag_widgets['tile'].isChecked():
            return pixmap
            
        # Get tile grid information
        tile_grid_info = self.get_tile_grid_info(original_width, original_height)
        if tile_grid_info is None:
            return pixmap
            
        tile_width, tile_height, cols, rows, tiles_info = tile_grid_info
        
        # Create a copy of the pixmap to draw on
        overlay_pixmap = pixmap.copy()
        
        # Initialize painter
        painter = QPainter(overlay_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Calculate scaling factors between original frame and displayed pixmap
        pixmap_width = overlay_pixmap.width()
        pixmap_height = overlay_pixmap.height()
        
        scale_x = pixmap_width / original_width
        scale_y = pixmap_height / original_height
        
        # Set up pen for drawing grid lines
        pen = QPen(Qt.GlobalColor.cyan, 2)
        pen.setStyle(Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        
        # Draw tile boundaries
        for tile_info in tiles_info:
            # Scale tile coordinates to pixmap coordinates
            x = int(tile_info['x'] * scale_x)
            y = int(tile_info['y'] * scale_y)
            width = int(tile_info['width'] * scale_x)
            height = int(tile_info['height'] * scale_y)
            
            # Draw tile rectangle
            painter.drawRect(x, y, width, height)
            

            
            # Reset pen for next tile rectangle
            painter.setPen(pen)
        

        
        painter.end()
        return overlay_pixmap

    def generate_output_filename_preview(self, input_path, output_dir, fps=30.0):
        """Generate preview of output filename based on current GUI settings"""
        if not input_path:
            return "no_video_selected.avi"
        
        from storage.filename_generator import generate_output_filename
        
        # Use results directory if output_dir is empty or not specified
        if not output_dir or output_dir.strip() == "":
            output_dir = "results"
        
        # Get parameters from GUI
        control_type = self.time_control_combo.currentText()
        
        if control_type == 'Control by frame':
            start_frame = self.start_frame_spin.value()
            max_frames = self.max_frames_spin.value()
            start_time = None
            duration = None
        else:  # Control by time
            start_time = self.start_time_spin.value() if self.start_time_spin.value() > 0 else None
            duration = self.duration_spin.value() if self.duration_spin.value() > 0 else None
            start_frame = 0
            max_frames = 1000
        
        return generate_output_filename(
            input_path=input_path,
            start_time=start_time,
            duration=duration,
            start_frame=start_frame,
            max_frames=max_frames,
            flow_only=self.flag_widgets['flow_only'].isChecked(),
            taa=self.flag_widgets['taa'].isChecked(),
            fast_mode=self.flag_widgets['fast'].isChecked(),
            tile_mode=self.flag_widgets['tile'].isChecked(),
            uncompressed=self.flag_widgets['uncompressed'].isChecked(),
            flow_format=self.flow_format_combo.currentText(),
            motion_vectors_clamp_range=self.mv_clamp_range_spin.value(),
            fps=fps  # Use provided fps instead of hardcoded 30.0
        )

    def generate_cache_directory_preview(self, input_path):
        """Generate preview of cache directory based on current GUI settings"""
        if not input_path:
            return "no_video_selected_flow_cache"
        
        from storage.filename_generator import generate_cache_directory
        
        # Get parameters from GUI
        control_type = self.time_control_combo.currentText()
        
        if control_type == 'Control by frame':
            start_frame = self.start_frame_spin.value()
            max_frames = self.max_frames_spin.value()
        else:  # Control by time
            start_frame = 0  # For cache purposes, time-based uses frame 0 start
            max_frames = 1000  # Default for time-based
        
        # GUI already contains correct values, no mapping needed
        model = self.model_combo.currentText()  # 'videoflow' or 'memflow'
        dataset = self.dataset_combo.currentText()  # 'things', 'sintel', 'kitti'
        architecture = self.vf_architecture_combo.currentText()  # 'mof' or 'bof'
        variant = self.vf_variant_combo.currentText()  # 'standard' or 'noise'
        
        return generate_cache_directory(
            input_path=input_path,
            start_frame=start_frame,
            max_frames=max_frames,
            sequence_length=self.sequence_length_spin.value(),
            fast_mode=self.flag_widgets['fast'].isChecked(),
            tile_mode=self.flag_widgets['tile'].isChecked(),
            model=model,
            dataset=dataset,
            architecture=architecture,
            variant=variant
        )

    def generate_flow_input_preview(self, input_path, output_dir, fps=30.0):
        """Generate preview of expected flow input filename based on current GUI settings with flow_only enabled"""
        if not input_path:
            return "no_video_selected_flow_only.avi"
        
        from storage.filename_generator import generate_output_filename
        
        # Use results directory if output_dir is empty or not specified
        if not output_dir or output_dir.strip() == "":
            output_dir = "results"
        
        # Get parameters from GUI (same as current settings but force flow_only=True)
        control_type = self.time_control_combo.currentText()
        
        if control_type == 'Control by frame':
            start_frame = self.start_frame_spin.value()
            max_frames = self.max_frames_spin.value()
            start_time = None
            duration = None
        else:  # Control by time
            start_time = self.start_time_spin.value() if self.start_time_spin.value() > 0 else None
            duration = self.duration_spin.value() if self.duration_spin.value() > 0 else None
            start_frame = 0
            max_frames = 1000
        
        return generate_output_filename(
            input_path=input_path,
            start_time=start_time,
            duration=duration,
            start_frame=start_frame,
            max_frames=max_frames,
            flow_only=True,  # Force flow_only for flow input preview
            taa=False,  # TAA is not applicable with flow_only
            fast_mode=self.flag_widgets['fast'].isChecked(),
            tile_mode=self.flag_widgets['tile'].isChecked(),
            uncompressed=self.flag_widgets['uncompressed'].isChecked(),
            flow_format=self.flow_format_combo.currentText(),
            motion_vectors_clamp_range=self.mv_clamp_range_spin.value(),
            fps=fps  # Use provided fps instead of hardcoded 30.0
        )

    def update_output_preview(self):
        """Update output path placeholder with preview filename"""
        if not hasattr(self, '_initialization_complete') or not self._initialization_complete:
            return
            
        input_path = self.input_video_edit.text()
        output_dir = self.output_path_edit.text()
        
        # Only show preview if video is loaded and has fps information
        if input_path and hasattr(self, 'fps') and self.fps and self.fps > 0:
            preview_filename = self.generate_output_filename_preview(input_path, output_dir, fps=self.fps)
            
            # Generate absolute path
            import os
            if not output_dir or output_dir.strip() == "":
                # Use default "results" directory relative to current working directory
                absolute_output_dir = os.path.abspath("results")
            else:
                # Use specified directory (convert to absolute if relative)
                absolute_output_dir = os.path.abspath(output_dir)
            
            absolute_filepath = os.path.join(absolute_output_dir, preview_filename)
            self.output_path_edit.setPlaceholderText(absolute_filepath)
        else:
            self.output_path_edit.setPlaceholderText("Load video to see filename preview")

    def update_cache_preview(self):
        """Update flow cache placeholder with preview directory name"""
        if not hasattr(self, '_initialization_complete') or not self._initialization_complete:
            return
            
        input_path = self.input_video_edit.text()
        
        # Only show preview if video is loaded and has fps information
        if input_path and hasattr(self, 'fps') and self.fps and self.fps > 0 and not self.flow_cache_edit.text():
            preview_cache = self.generate_cache_directory_preview(input_path)
            # Show absolute path to cache directory
            import os
            absolute_cache_path = os.path.abspath(preview_cache)
            self.flow_cache_edit.setPlaceholderText(absolute_cache_path)
        elif self.flow_cache_edit.text(): # If cache is manually set, clear placeholder
            self.flow_cache_edit.setPlaceholderText("")
        else:
            self.flow_cache_edit.setPlaceholderText("Load video to see cache preview")

    def update_flow_input_preview(self):
        """Update flow input placeholder with preview filename"""
        if not hasattr(self, '_initialization_complete') or not self._initialization_complete:
            return
            
        input_path = self.input_video_edit.text()
        output_dir = self.output_path_edit.text()
        
        # Only show preview if flow input is enabled and video is loaded with fps information
        if (input_path and hasattr(self, 'fps') and self.fps and self.fps > 0 and 
            self.flow_input_edit.isEnabled() and not self.flow_input_edit.text()):
            preview_filename = self.generate_flow_input_preview(input_path, output_dir, fps=self.fps)
            
            # Generate absolute path
            import os
            if not output_dir or output_dir.strip() == "":
                # Use default "results" directory relative to current working directory
                absolute_output_dir = os.path.abspath("results")
            else:
                # Use specified directory (convert to absolute if relative)
                absolute_output_dir = os.path.abspath(output_dir)
            
            absolute_filepath = os.path.join(absolute_output_dir, preview_filename)
            self.flow_input_edit.setPlaceholderText(absolute_filepath)
        elif not self.flow_input_edit.isEnabled():
            self.flow_input_edit.setPlaceholderText("Flow input disabled in flow-only mode")
        elif self.flow_input_edit.text(): # If flow input is manually set, clear placeholder
            self.flow_input_edit.setPlaceholderText("")
        else:
            self.flow_input_edit.setPlaceholderText("Load video to see expected flow input preview")

    def update_output_status(self):
        """Update output file status badge"""
        if not hasattr(self, 'output_status_label'):
            return
            
        input_path = self.input_video_edit.text()
        output_dir = self.output_path_edit.text()
        
        # Check if we can determine the output file path
        if input_path and hasattr(self, 'fps') and self.fps and self.fps > 0:
            preview_filename = self.generate_output_filename_preview(input_path, output_dir, fps=self.fps)
            
            import os
            if not output_dir or output_dir.strip() == "":
                absolute_output_dir = os.path.abspath("results")
            else:
                absolute_output_dir = os.path.abspath(output_dir)
            
            output_filepath = os.path.join(absolute_output_dir, preview_filename)
            
            if os.path.exists(output_filepath):
                # File exists - green checkmark
                self.output_status_label.setText("✓")
                self.output_status_label.setStyleSheet("""
                    QLabel { 
                        background-color: #28a745; 
                        color: white; 
                        border-radius: 10px; 
                        font-weight: bold;
                        font-size: 12px;
                    }
                """)
                self.output_status_label.setToolTip(f"Output file exists:\n{output_filepath}")
            else:
                # File doesn't exist - orange circle
                self.output_status_label.setText("○")
                self.output_status_label.setStyleSheet("""
                    QLabel { 
                        background-color: #ffc107; 
                        color: white; 
                        border-radius: 10px; 
                        font-weight: bold;
                        font-size: 12px;
                    }
                """)
                self.output_status_label.setToolTip(f"Output file will be created:\n{output_filepath}")
        else:
            # No video loaded or can't determine path - gray
            self.output_status_label.setText("?")
            self.output_status_label.setStyleSheet("""
                QLabel { 
                    background-color: #6c757d; 
                    color: white; 
                    border-radius: 10px; 
                    font-weight: bold;
                    font-size: 12px;
                }
            """)
            self.output_status_label.setToolTip("Load video to check output file status")

    def update_cache_status(self):
        """Update cache status badge"""
        if not hasattr(self, 'cache_status_label'):
            return
            
        input_path = self.input_video_edit.text()
        cache_path = self.flow_cache_edit.text()
        
        # If cache path is manually specified
        if cache_path and cache_path.strip():
            import os
            if os.path.exists(cache_path) and os.path.isdir(cache_path):
                # Check if cache contains any flow files
                flow_files = [f for f in os.listdir(cache_path) if f.startswith('flow_frame_') and (f.endswith('.npz') or f.endswith('.flo'))]
                if flow_files:
                    # Cache exists with files - green checkmark
                    self.cache_status_label.setText("✓")
                    self.cache_status_label.setStyleSheet("""
                        QLabel { 
                            background-color: #28a745; 
                            color: white; 
                            border-radius: 10px; 
                            font-weight: bold;
                            font-size: 12px;
                        }
                    """)
                    self.cache_status_label.setToolTip(f"Cache directory exists with {len(flow_files)} flow files:\n{cache_path}")
                else:
                    # Directory exists but empty - yellow
                    self.cache_status_label.setText("!")
                    self.cache_status_label.setStyleSheet("""
                        QLabel { 
                            background-color: #ffc107; 
                            color: white; 
                            border-radius: 10px; 
                            font-weight: bold;
                            font-size: 12px;
                        }
                    """)
                    self.cache_status_label.setToolTip(f"Cache directory exists but is empty:\n{cache_path}")
            else:
                # Specified path doesn't exist - red X
                self.cache_status_label.setText("✗")
                self.cache_status_label.setStyleSheet("""
                    QLabel { 
                        background-color: #dc3545; 
                        color: white; 
                        border-radius: 10px; 
                        font-weight: bold;
                        font-size: 12px;
                    }
                """)
                self.cache_status_label.setToolTip(f"Cache directory does not exist:\n{cache_path}")
        # If no cache path specified, check auto-generated cache path
        elif input_path and hasattr(self, 'fps') and self.fps and self.fps > 0:
            preview_cache = self.generate_cache_directory_preview(input_path)
            import os
            absolute_cache_path = os.path.abspath(preview_cache)
            
            if os.path.exists(absolute_cache_path) and os.path.isdir(absolute_cache_path):
                # Check if cache contains any flow files
                flow_files = [f for f in os.listdir(absolute_cache_path) if f.startswith('flow_frame_') and (f.endswith('.npz') or f.endswith('.flo'))]
                if flow_files:
                    # Auto cache exists with files - green checkmark
                    self.cache_status_label.setText("✓")
                    self.cache_status_label.setStyleSheet("""
                        QLabel { 
                            background-color: #28a745; 
                            color: white; 
                            border-radius: 10px; 
                            font-weight: bold;
                            font-size: 12px;
                        }
                    """)
                    self.cache_status_label.setToolTip(f"Auto cache exists with {len(flow_files)} flow files:\n{absolute_cache_path}")
                else:
                    # Directory exists but empty - yellow
                    self.cache_status_label.setText("!")
                    self.cache_status_label.setStyleSheet("""
                        QLabel { 
                            background-color: #ffc107; 
                            color: white; 
                            border-radius: 10px; 
                            font-weight: bold;
                            font-size: 12px;
                        }
                    """)
                    self.cache_status_label.setToolTip(f"Auto cache directory exists but is empty:\n{absolute_cache_path}")
            else:
                # Auto cache doesn't exist - orange circle
                self.cache_status_label.setText("○")
                self.cache_status_label.setStyleSheet("""
                    QLabel { 
                        background-color: #ffc107; 
                        color: white; 
                        border-radius: 10px; 
                        font-weight: bold;
                        font-size: 12px;
                    }
                """)
                self.cache_status_label.setToolTip(f"Auto cache will be created:\n{absolute_cache_path}")
        else:
            # No video loaded or can't determine cache path - gray
            self.cache_status_label.setText("?")
            self.cache_status_label.setStyleSheet("""
                QLabel { 
                    background-color: #6c757d; 
                    color: white; 
                    border-radius: 10px; 
                    font-weight: bold;
                    font-size: 12px;
                }
            """)
            self.cache_status_label.setToolTip("Load video to check cache status")

    def generate_command(self, extra_flags=None):
        """Generate the flow_processor command based on current settings"""
        if not self.input_video_edit.text():
            return ""
        
        cmd_parts = ["python", "flow_processor.py"]
        
        # Input video (required)
        cmd_parts.extend(["--input", f'"{self.input_video_edit.text()}"'])
        
        # Output path
        output_path = self.output_path_edit.text().strip()
        if not output_path:
            output_path = "results"  # Default directory
        cmd_parts.extend(["--output", f'"{output_path}"'])
        
        # Flow cache
        if self.flow_cache_edit.text():
            cmd_parts.extend(["--flow-cache", f'"{self.flow_cache_edit.text()}"'])
        
        # Flow input video (only if flow_only is not enabled and taa is enabled)
        flow_only_enabled = self.flag_widgets['flow_only'].isChecked()
        if not flow_only_enabled and self.flow_input_edit.text():
            cmd_parts.extend(["--flow-input", f'"{self.flow_input_edit.text()}"'])
        
        # Boolean flags
        for key, widget in self.flag_widgets.items():
            if widget.isChecked() and widget.isEnabled():
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
            content="Starting interactive flow visualizer in new console",
            orient=Qt.Orientation.Horizontal,
            isClosable=True,
            duration=2000,
            position=InfoBarPosition.TOP,
            parent=self
        )
        
        # Run interactive command in new console window, but use same Python environment
        current_python = sys.executable
        ps_command = f'Write-Host "Starting Interactive Flow Visualizer..." -ForegroundColor Green; Write-Host "Command: {command}" -ForegroundColor Yellow; Write-Host ""; & "{current_python}" {command.split("python", 1)[1]}; Write-Host ""; Write-Host "Interactive session completed. Press any key to close..." -ForegroundColor Green; $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")'
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
            content="Processing started in new console",
            orient=Qt.Orientation.Horizontal,
            isClosable=True,
            duration=3000,
            position=InfoBarPosition.TOP,
            parent=self
        )
        
        # Run processing in new console window, but use same Python environment
        current_python = sys.executable
        ps_command = f'Write-Host "Starting Flow processing..." -ForegroundColor Green; Write-Host "Command: {command}" -ForegroundColor Yellow; Write-Host ""; & "{current_python}" {command.split("python", 1)[1]}; Write-Host ""; Write-Host "Processing completed. Press any key to close..." -ForegroundColor Green; $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")'
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