"""
Cache Manager - optical flow caching and storage management
"""

import os
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from tqdm import tqdm
import struct


class FlowFileHandler:
    """Handler for optical flow file operations (save/load)"""
    
    @staticmethod
    def save_flow_flo(flow: np.ndarray, filename: str):
        """Save optical flow in Middlebury .flo format (lossless)"""
        if torch.is_tensor(flow):
            flow = flow.cpu().numpy()
        
        height, width = flow.shape[:2]
        
        with open(filename, 'wb') as f:
            f.write(b'PIEH')
            f.write(struct.pack('<I', width))
            f.write(struct.pack('<I', height))
            flow_data = flow.astype(np.float32)
            f.write(flow_data.tobytes())
    
    @staticmethod
    def save_flow_npz(flow: np.ndarray, filename: str, frame_idx: Optional[int] = None, 
                     metadata: Optional[Dict[str, Any]] = None):
        """Save optical flow in NumPy .npz format (lossless, compressed)"""
        if torch.is_tensor(flow):
            flow = flow.cpu().numpy()
        
        save_data = {'flow': flow.astype(np.float32)}
        
        if frame_idx is not None:
            save_data['frame_idx'] = frame_idx
            
        if metadata is not None:
            save_data.update(metadata)
            
        np.savez_compressed(filename, **save_data)
    
    @staticmethod
    def load_flow_flo(filename: str) -> np.ndarray:
        """Load optical flow from Middlebury .flo format"""
        with open(filename, 'rb') as f:
            magic = f.read(4)
            if magic != b'PIEH':
                raise ValueError(f"Invalid .flo file magic number: {magic}")
            
            width = struct.unpack('<I', f.read(4))[0]
            height = struct.unpack('<I', f.read(4))[0]
            
            flow_data = f.read(width * height * 2 * 4)
            flow = np.frombuffer(flow_data, dtype=np.float32)
            flow = flow.reshape(height, width, 2)
            
        return flow
    
    @staticmethod
    def load_flow_npz(filename: str) -> Dict[str, Any]:
        """Load optical flow from NumPy .npz format"""
        data = np.load(filename)
        return dict(data)


class LODGenerator:
    """Level-of-Detail generator for optical flow data"""
    
    @staticmethod
    def generate_lods(flow: np.ndarray, num_lods: int = 5) -> List[np.ndarray]:
        """
        Generate Level-of-Detail (LOD) pyramid for flow data using arithmetic averaging with weighted padding
        
        Args:
            flow: Original flow data [H, W, 2]
            num_lods: Number of LOD levels to generate (default: 5)
            
        Returns:
            List of flow data at different LOD levels [original, lod1, lod2, ...]
        """
        if torch.is_tensor(flow):
            flow = flow.cpu().numpy()
        
        lods = [flow]  # LOD 0 is original
        
        current_flow = flow.copy()
        
        for lod_level in range(1, num_lods):
            h, w = current_flow.shape[:2]
            
            # Check if dimensions are odd (not divisible by 2) and need padding
            need_h_padding = (h % 2) != 0
            need_w_padding = (w % 2) != 0
            
            if need_h_padding or need_w_padding:
                # Calculate padding to make dimensions even (divisible by 2)
                pad_h = 1 if need_h_padding else 0
                pad_w = 1 if need_w_padding else 0
                
                # Add padding to bottom and right (simpler than centering)
                pad_top, pad_bottom = 0, pad_h
                pad_left, pad_right = 0, pad_w
                
                # Create weight mask (1 for original data, 0 for padding)
                weight_mask = np.ones((h, w), dtype=np.float32)
                padded_weight = np.pad(weight_mask, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                                     mode='constant', constant_values=0)
                
                # Pad flow data with zeros
                padded_flow = np.pad(current_flow, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                                   mode='constant', constant_values=0)
                
                current_flow = padded_flow
                h, w = current_flow.shape[:2]
            else:
                # No padding needed, create uniform weight mask for original data
                padded_weight = np.ones((h, w), dtype=np.float32)
            
            # Downsample by factor of 2 using weighted averaging
            new_h = h // 2
            new_w = w // 2
            
            downsampled_flow = np.zeros((new_h, new_w, 2), dtype=np.float32)
            
            for y in range(new_h):
                for x in range(new_w):
                    # Get 2x2 block
                    y_start, y_end = y * 2, min((y + 1) * 2, h)
                    x_start, x_end = x * 2, min((x + 1) * 2, w)
                    
                    flow_block = current_flow[y_start:y_end, x_start:x_end]
                    weight_block = padded_weight[y_start:y_end, x_start:x_end]
                    
                    # Calculate weighted average only using non-zero weights (original data)
                    total_weight = np.sum(weight_block)
                    if total_weight > 0:
                        # Weighted average for each channel
                        weighted_flow_u = np.sum(flow_block[:, :, 0] * weight_block) / total_weight
                        weighted_flow_v = np.sum(flow_block[:, :, 1] * weight_block) / total_weight
                        
                        # Scale flow vectors by 0.5 (since we're downsampling by 2)
                        downsampled_flow[y, x, 0] = weighted_flow_u * 0.5
                        downsampled_flow[y, x, 1] = weighted_flow_v * 0.5
                    else:
                        # All weights are zero (pure padding), keep zero flow
                        downsampled_flow[y, x] = 0
            
            lods.append(downsampled_flow)
            current_flow = downsampled_flow
            
            # Update weight mask for next iteration
            padded_weight = np.ones((new_h, new_w), dtype=np.float32)
        
        return lods


class FlowCacheManager:
    """Manager for optical flow caching and LOD operations"""
    
    def __init__(self):
        """Initialize cache manager"""
        self.file_handler = FlowFileHandler()
        self.lod_generator = LODGenerator()
    
    def generate_cache_path(self, input_path: str, start_frame: int, max_frames: int, 
                          sequence_length: int, fast_mode: bool, tile_mode: bool) -> str:
        """Generate cache directory path based on video processing parameters"""
        video_name = Path(input_path).stem
        cache_params = [
            f"seq{sequence_length}",
            f"start{start_frame}",
            f"frames{max_frames}"
        ]
        
        if fast_mode:
            cache_params.append("fast")
        if tile_mode:
            cache_params.append("tile")
            
        cache_id = "_".join(cache_params)
        cache_dir_name = f"{video_name}_flow_cache_{cache_id}"
        
        cache_path = Path(input_path).parent / cache_dir_name
        return str(cache_path)
    
    def check_cache_exists(self, cache_dir: str, max_frames: int) -> Tuple[bool, Optional[str]]:
        """Check if complete flow cache exists for the requested number of frames"""
        if not os.path.exists(cache_dir):
            return False, None
            
        flo_files = []
        npz_files = []
        
        for i in range(max_frames):
            flo_file = os.path.join(cache_dir, f"flow_frame_{i:06d}.flo")
            npz_file = os.path.join(cache_dir, f"flow_frame_{i:06d}.npz")
            
            if os.path.exists(flo_file):
                flo_files.append(flo_file)
            if os.path.exists(npz_file):
                npz_files.append(npz_file)
        
        if len(flo_files) == max_frames:
            return True, 'flo'
        elif len(npz_files) == max_frames:
            return True, 'npz'
        else:
            return False, None
    
    def load_cached_flow(self, cache_dir: str, frame_idx: int, format_type: str = 'auto') -> np.ndarray:
        """Load cached optical flow for specific frame"""
        if format_type == 'auto':
            npz_file = os.path.join(cache_dir, f"flow_frame_{frame_idx:06d}.npz")
            flo_file = os.path.join(cache_dir, f"flow_frame_{frame_idx:06d}.flo")
            
            if os.path.exists(npz_file):
                npz_data = self.file_handler.load_flow_npz(npz_file)
                return npz_data['flow']
            elif os.path.exists(flo_file):
                return self.file_handler.load_flow_flo(flo_file)
            else:
                raise FileNotFoundError(f"No cached flow found for frame {frame_idx}")
        
        elif format_type == 'npz':
            npz_file = os.path.join(cache_dir, f"flow_frame_{frame_idx:06d}.npz")
            npz_data = self.file_handler.load_flow_npz(npz_file)
            return npz_data['flow']
        
        elif format_type == 'flo':
            flo_file = os.path.join(cache_dir, f"flow_frame_{frame_idx:06d}.flo")
            return self.file_handler.load_flow_flo(flo_file)
        
        else:
            raise ValueError(f"Invalid format_type: {format_type}")
    
    def save_flow_to_cache(self, flow: np.ndarray, cache_dir: str, frame_idx: int, save_format: str = 'npz'):
        """Save flow data to cache"""
        os.makedirs(cache_dir, exist_ok=True)
        
        if save_format in ['flo', 'both']:
            flo_filename = os.path.join(cache_dir, f"flow_frame_{frame_idx:06d}.flo")
            self.file_handler.save_flow_flo(flow, flo_filename)
            
        if save_format in ['npz', 'both']:
            npz_filename = os.path.join(cache_dir, f"flow_frame_{frame_idx:06d}.npz")
            metadata = {
                'frame_idx': frame_idx,
                'shape': flow.shape,
                'dtype': str(flow.dtype)
            }
            self.file_handler.save_flow_npz(flow, npz_filename, frame_idx, metadata)
    
    def save_optical_flow_files(self, flow: np.ndarray, base_filename: str, frame_idx: int, save_format: str):
        """Save optical flow in specified format(s) with base filename"""
        if torch.is_tensor(flow):
            flow = flow.cpu().numpy()
        
        metadata = {
            'frame_idx': frame_idx,
            'shape': flow.shape,
            'dtype': str(flow.dtype),
            'min_flow': float(np.min(flow)),
            'max_flow': float(np.max(flow)),
            'mean_magnitude': float(np.mean(np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)))
        }
        
        if save_format in ['flo', 'both']:
            flo_filename = f"{base_filename}_frame_{frame_idx:06d}.flo"
            self.file_handler.save_flow_flo(flow, flo_filename)
            
        if save_format in ['npz', 'both']:
            npz_filename = f"{base_filename}_frame_{frame_idx:06d}.npz"
            self.file_handler.save_flow_npz(flow, npz_filename, frame_idx, metadata)
    
    def save_flow_lods(self, lods: List[np.ndarray], cache_dir: str, frame_idx: int):
        """Save LOD pyramid for a frame"""
        os.makedirs(cache_dir, exist_ok=True)
        
        for lod_level, lod_data in enumerate(lods):
            filename = os.path.join(cache_dir, f"flow_frame_{frame_idx:06d}_lod{lod_level}.npz")
            metadata = {
                'frame_idx': frame_idx,
                'lod_level': lod_level,
                'shape': lod_data.shape,
                'dtype': str(lod_data.dtype)
            }
            self.file_handler.save_flow_npz(lod_data, filename, frame_idx, metadata)
    
    def load_flow_lod(self, cache_dir: str, frame_idx: int, lod_level: int = 0) -> np.ndarray:
        """Load specific LOD level for a frame"""
        filename = os.path.join(cache_dir, f"flow_frame_{frame_idx:06d}_lod{lod_level}.npz")
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"LOD {lod_level} not found for frame {frame_idx}")
        
        npz_data = self.file_handler.load_flow_npz(filename)
        return npz_data['flow']
    
    def check_flow_lods_exist(self, cache_dir: str, max_frames: int, num_lods: int = 5) -> bool:
        """Check if LOD pyramids exist for all frames"""
        if not os.path.exists(cache_dir):
            return False
        
        for frame_idx in range(max_frames):
            for lod_level in range(num_lods):
                filename = os.path.join(cache_dir, f"flow_frame_{frame_idx:06d}_lod{lod_level}.npz")
                if not os.path.exists(filename):
                    return False
        
        return True
    
    def generate_lods_for_cache(self, cache_dir: str, max_frames: int, num_lods: int = 5):
        """Generate LOD pyramids for all cached flow data"""
        print(f"Generating LOD pyramids (levels 0-{num_lods-1}) for {max_frames} frames...")
        
        for frame_idx in tqdm(range(max_frames), desc="Generating LODs"):
            # Load original flow data
            flow_data = self.load_cached_flow(cache_dir, frame_idx)
            
            # Generate LOD pyramid
            lods = self.lod_generator.generate_lods(flow_data, num_lods)
            
            # Save LOD pyramid
            self.save_flow_lods(lods, cache_dir, frame_idx)
        
        print(f"LOD generation complete. Generated {num_lods} levels for {max_frames} frames.") 