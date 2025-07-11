#!/usr/bin/env python3
"""
Portable Version Builder for VideoFlow Processor

This script creates a portable version of the VideoFlow Processor that can be 
distributed to users without requiring Python installation or dependency management.
"""

import os
import sys
import shutil
import subprocess
import zipfile
import urllib.request
import tempfile
from pathlib import Path
from typing import Optional

class PortableBuilder:
    """Builder for creating portable VideoFlow Processor distributions"""
    
    def __init__(self, output_dir: str = "FlowRunner_Portable"):
        self.output_dir = Path(output_dir)
        self.python_version = "3.10.11"
        self.python_url = f"https://www.python.org/ftp/python/{self.python_version}/python-{self.python_version}-embed-amd64.zip"
        self.get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
        
        # Directories to exclude when copying app
        self.exclude_dirs = {'content', 'results', '.cursor', '__pycache__', '.git', '.venv', 'venv', 'FlowRunner_Portable'}
        self.exclude_files = {'.gitignore', '.gitattributes', 'create_portable.py', 'test_video_info.py'}
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with level"""
        print(f"[{level}] {message}")
    
    def cleanup_existing(self):
        """Remove existing portable directory if it exists"""
        if self.output_dir.exists():
            self.log(f"Removing existing directory: {self.output_dir}")
            shutil.rmtree(self.output_dir)
    
    def create_directories(self):
        """Create directory structure for portable version"""
        self.log("Creating directory structure...")
        
        # Create main directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "python").mkdir(exist_ok=True)
        (self.output_dir / "app").mkdir(exist_ok=True)
        
        self.log(f"Created directory structure at: {self.output_dir.absolute()}")
    
    def download_python(self):
        """Download and extract embedded Python"""
        self.log("Downloading embedded Python...")
        
        python_dir = self.output_dir / "python"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            python_zip = temp_path / "python.zip"
            
            # Download Python
            self.log(f"Downloading Python {self.python_version}...")
            urllib.request.urlretrieve(self.python_url, python_zip)
            
            # Extract Python
            self.log("Extracting Python...")
            with zipfile.ZipFile(python_zip, 'r') as zip_ref:
                zip_ref.extractall(python_dir)
            
            # Download get-pip.py
            self.log("Downloading get-pip.py...")
            get_pip_path = python_dir / "get-pip.py"
            urllib.request.urlretrieve(self.get_pip_url, get_pip_path)
            
            # Enable pip in embedded Python by modifying ._pth file
            pth_files = list(python_dir.glob("python*._pth"))
            if pth_files:
                pth_file = pth_files[0]
                content = pth_file.read_text()
                if "#import site" in content:
                    # Uncomment import site to enable pip
                    content = content.replace("#import site", "import site")
                    pth_file.write_text(content)
                    self.log("Enabled site-packages in embedded Python")
            
            self.log("Python installation completed")
    
    def install_dependencies(self):
        """Install Python dependencies"""
        self.log("Installing dependencies...")
        
        python_exe = self.output_dir / "python" / "python.exe"
        get_pip_script = self.output_dir / "python" / "get-pip.py"
        
        # Install pip
        self.log("Installing pip...")
        subprocess.run([str(python_exe), str(get_pip_script)], check=True)
        
        # Install PyTorch with CUDA support first
        self.log("Installing PyTorch with CUDA support...")
        try:
            subprocess.run([
                str(python_exe), "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu118",
                "--no-warn-script-location"
            ], check=True)
            self.log("PyTorch with CUDA 11.8 support installed successfully")
        except subprocess.CalledProcessError:
            self.log("Failed to install CUDA version, trying CPU version...", "WARNING")
            subprocess.run([
                str(python_exe), "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio",
                "--no-warn-script-location"
            ], check=True)
            self.log("PyTorch CPU version installed as fallback")
        
        # Install other dependencies from requirements.txt (excluding torch packages)
        self.log("Installing other packages from requirements.txt...")
        requirements_path = Path("requirements.txt")
        
        if requirements_path.exists():
            # Read requirements and filter out torch packages
            requirements_content = requirements_path.read_text()
            filtered_requirements = []
            
            for line in requirements_content.splitlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    # Skip torch packages as we installed them separately
                    if not any(pkg in line.lower() for pkg in ['torch', 'torchvision', 'torchaudio']):
                        filtered_requirements.append(line)
            
            # Create temporary requirements file without torch packages
            temp_requirements = self.output_dir / "temp_requirements.txt"
            temp_requirements.write_text('\n'.join(filtered_requirements))
            
            subprocess.run([
                str(python_exe), "-m", "pip", "install", 
                "-r", str(temp_requirements),
                "--no-warn-script-location"
            ], check=True)
            
            # Clean up temporary file
            temp_requirements.unlink()
            self.log("Other dependencies installed successfully")
        else:
            self.log("requirements.txt not found, skipping dependency installation", "WARNING")
        
        # Clean up get-pip.py
        get_pip_script.unlink(missing_ok=True)
    
    def copy_application(self):
        """Copy application files to portable version"""
        self.log("Copying application files...")
        
        source_dir = Path(".")
        app_dir = self.output_dir / "app"
        
        # Copy files and directories
        copied_files = 0
        skipped_items = []
        
        for item in source_dir.iterdir():
            if item.name in self.exclude_dirs or item.name in self.exclude_files:
                skipped_items.append(item.name)
                continue
            
            target_path = app_dir / item.name
            
            try:
                if item.is_dir():
                    shutil.copytree(item, target_path, 
                                  ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
                    copied_files += 1
                else:
                    shutil.copy2(item, target_path)
                    copied_files += 1
            except Exception as e:
                self.log(f"Error copying {item.name}: {e}", "ERROR")
        
        self.log(f"Copied {copied_files} items to app directory")
        if skipped_items:
            self.log(f"Skipped items: {', '.join(skipped_items)}")
        
        # Create empty results directory in app folder
        results_dir = app_dir / "results"
        results_dir.mkdir(exist_ok=True)
        self.log("Created empty results directory in app folder")
    
    def create_bat_files(self):
        """Create batch files for running the application"""
        self.log("Creating batch files...")
        
        # CLI batch file
        cli_bat_content = '''@echo off
rem This script runs the VideoFlow Processor CLI in a portable way.
rem It passes all command-line arguments to flow_processor.py.

rem Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"

rem Set paths for portable python and the application
set "PYTHON_DIR=%SCRIPT_DIR%python"
set "APP_DIR=%SCRIPT_DIR%app"

rem Add portable Python to the PATH for this session
echo Setting up environment for portable execution...
set "PATH=%PYTHON_DIR%;%PYTHON_DIR%\\Scripts;%PATH%"

rem Change to the application directory and run the CLI
echo Starting VideoFlow Processor CLI...
cd /d "%APP_DIR%"

rem Launch the application with all provided arguments
"%PYTHON_DIR%\\python.exe" flow_processor.py %*

echo.
echo CLI process has finished.
pause
'''
        
        # GUI batch file
        gui_bat_content = '''@echo off
rem This script runs the VideoFlow Processor GUI in a portable way.

rem Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"

rem Set paths for portable python and the application
set "PYTHON_DIR=%SCRIPT_DIR%python"
set "APP_DIR=%SCRIPT_DIR%app"

rem Add portable Python to the PATH for this session
echo Setting up environment for portable execution...
set "PATH=%PYTHON_DIR%;%PYTHON_DIR%\\Scripts;%PATH%"

rem Change to the application directory and run the GUI
echo Starting VideoFlow Processor GUI...
cd /d "%APP_DIR%"
echo Running from: %cd%

rem Launch the application
"%PYTHON_DIR%\\python.exe" gui_runner.py

echo.
echo Application has been closed.
pause
'''
        
        # CUDA check batch file
        cuda_check_bat_content = '''@echo off
rem This script checks CUDA availability using the portable Python environment.

rem Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"

rem Set paths for portable python and the application
set "PYTHON_DIR=%SCRIPT_DIR%python"
set "APP_DIR=%SCRIPT_DIR%app"

rem Add portable Python to the PATH for this session
echo Setting up environment for portable execution...
set "PATH=%PYTHON_DIR%;%PYTHON_DIR%\\Scripts;%PATH%"

rem Change to the application directory and run CUDA check
echo Checking CUDA availability...
cd /d "%APP_DIR%"

rem Launch the CUDA check script
"%PYTHON_DIR%\\python.exe" check_cuda.py

echo.
echo CUDA check completed.
pause
'''

        # Write batch files
        (self.output_dir / "run_cli.bat").write_text(cli_bat_content, encoding='utf-8')
        (self.output_dir / "run_gui.bat").write_text(gui_bat_content, encoding='utf-8')
        (self.output_dir / "check_cuda.bat").write_text(cuda_check_bat_content, encoding='utf-8')
        
        self.log("Created batch files: run_cli.bat, run_gui.bat, check_cuda.bat")
    
    def build(self):
        """Build the complete portable version"""
        self.log("Starting portable build process...")
        
        try:
            self.cleanup_existing()
            self.create_directories()
            self.download_python()
            self.install_dependencies()
            self.copy_application()
            self.create_bat_files()
            
            self.log("✓ Portable version created successfully!")
            self.log(f"Location: {self.output_dir.absolute()}")
            self.log("")
            self.log("You can now distribute the entire directory to users.")
            self.log("They can run the application using run_gui.bat or run_cli.bat")
            
        except Exception as e:
            self.log(f"Build failed: {e}", "ERROR")
            raise


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create portable VideoFlow Processor distribution")
    parser.add_argument("--output", "-o", default="FlowRunner_Portable",
                       help="Output directory name (default: FlowRunner_Portable)")
    parser.add_argument("--clean", "-c", action="store_true",
                       help="Clean existing output directory before building")
    
    args = parser.parse_args()
    
    builder = PortableBuilder(args.output)
    
    if args.clean and builder.output_dir.exists():
        builder.cleanup_existing()
    
    builder.build()


if __name__ == "__main__":
    main() 