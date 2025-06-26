"""
Device Manager - device management for processing
"""

import torch
from typing import Dict, Any, Optional


class DeviceManager:
    """Device manager for CUDA/CPU detection and management"""
    
    def __init__(self):
        self._device = None
        self._device_info = None
    
    def get_device(self, device_preference: str = 'auto') -> str:
        """
        Determines optimal device for processing
        
        Args:
            device_preference: 'auto', 'cuda', or 'cpu'
            
        Returns:
            Device name string ('cuda' or 'cpu')
        """
        if self._device is not None:
            return self._device
            
        if device_preference == 'cpu':
            self._device = 'cpu'
        elif device_preference == 'cuda':
            if torch.cuda.is_available():
                self._device = 'cuda'
            else:
                print("Warning: CUDA requested but not available, falling back to CPU")
                self._device = 'cpu'
        else:  # auto
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        return self._device
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Gets information about the used device
        
        Returns:
            Dictionary with device information
        """
        if self._device_info is not None:
            return self._device_info
            
        device = self.get_device()
        
        info = {
            'device': device,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if device == 'cuda' and torch.cuda.is_available():
            info.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                'gpu_memory_formatted': f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
            })
        
        self._device_info = info
        return info
    
    def print_device_info(self):
        """Prints device information to console"""
        info = self.get_device_info()
        
        print(f"CUDA available: {info['cuda_available']}")
        
        if info['device'] == 'cuda':
            print(f"GPU: {info['gpu_name']}")
            print(f"GPU Memory: {info['gpu_memory_formatted']}")
        else:
            print("Using CPU for processing")
    
    def reset(self):
        """Resets cached device information"""
        self._device = None
        self._device_info = None 