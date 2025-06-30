import numpy as np
import cv2
from pathlib import Path
from VideoFlow.core.utils.frame_utils import writeFlow
import torch
import os
import time

def calculate_pixel_quality(src_color, target_color):
    """Calculate quality score for a single pixel pair"""
    src_f = src_color.astype(float)
    tgt_f = target_color.astype(float)
    rgb_distance = np.sqrt(np.sum((src_f - tgt_f) ** 2))
    rgb_max_distance = np.sqrt(3 * 255**2)
    rgb_similarity = 1.0 - (rgb_distance / rgb_max_distance)
    abs_diff = np.mean(np.abs(src_f - tgt_f)) / 255.0
    abs_similarity = 1.0 - abs_diff
    src_norm = np.linalg.norm(src_f)
    target_norm = np.linalg.norm(tgt_f)
    if src_norm > 1e-6 and target_norm > 1e-6:
        cosine_sim = np.dot(src_f, tgt_f) / (src_norm * target_norm)
        cosine_similarity = (cosine_sim + 1.0) / 2.0
    else:
        norm_diff = np.abs(src_norm - target_norm)
        cosine_similarity = 1.0 - (norm_diff / rgb_max_distance)
    overall_similarity = (rgb_similarity + abs_similarity + cosine_similarity) / 3.0
    return overall_similarity

def is_good_quality(similarity, threshold):
    """Check if a similarity score is above the defined quality threshold."""
    return similarity > threshold

def generate_spiral_path(width, height):
    """Generates coordinates in a spiral path outwards from the center."""
    x, y = 0, 0
    dx, dy = 0, -1
    for i in range(max(width, height)**2):
        if (-width/2 < x <= width/2) and (-height/2 < y <= height/2):
            yield (x, y)
        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
            dx, dy = -dy, dx
        x, y = x + dx, y + dy

def extract_region(image, center_x, center_y, radius):
    """Extract a square region around a center point"""
    h, w = image.shape[:2]
    x1 = max(0, int(center_x - radius))
    y1 = max(0, int(center_y - radius))
    x2 = min(w, int(center_x + radius))
    y2 = min(h, int(center_y + radius))
    region = image[y1:y2, x1:x2]
    target_size = int(2 * radius)
    if region.shape[0] < target_size or region.shape[1] < target_size:
        pad_h = max(0, target_size - region.shape[0])
        pad_w = max(0, target_size - region.shape[1])
        if len(image.shape) == 3:
            region = np.pad(region, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        else:
            region = np.pad(region, ((0, pad_h), (0, pad_w)), mode='constant')
    return region, (x1, y1, x2, y2)

def phase_correlation_with_rotation(img1, img2):
    """Compute phase correlation between two images to find translation."""
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    if img1.shape != img2.shape:
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        min_h, min_w = min(h1, h2), min(w1, w2)
        img1, img2 = img1[:min_h, :min_w], img2[:min_h, :min_w]
        if min_h < 2 or min_w < 2:
            return 0.0, 0.0, 0.0, 0.0
    (dx, dy), confidence = cv2.phaseCorrelate(img1, img2)
    return dx, dy, 0.0, confidence

def perform_coarse_correction(frame1, frame2, source_pixel, lod_flow_vector, detail_analysis_region_size):
    """Performs coarse correction using phase correlation."""
    orig_x, orig_y = source_pixel
    lod_flow_x, lod_flow_y = lod_flow_vector
    lod_target_x = orig_x - lod_flow_x
    lod_target_y = orig_y - lod_flow_y
    region1, _ = extract_region(frame1, orig_x, orig_y, detail_analysis_region_size)
    region2, _ = extract_region(frame2, lod_target_x, lod_target_y, detail_analysis_region_size)
    dx, dy, angle, confidence = phase_correlation_with_rotation(region1, region2)
    corrected_flow_x = lod_flow_x - dx
    corrected_flow_y = lod_flow_y - dy
    final_target_x = orig_x - corrected_flow_x
    final_target_y = orig_y - corrected_flow_y
    h, w = frame1.shape[:2]
    similarity = 0.0
    if (0 <= final_target_x < w and 0 <= final_target_y < h):
        similarity = calculate_pixel_quality(frame1[orig_y, orig_x], frame2[int(final_target_y), int(final_target_x)])
    return {'flow': (corrected_flow_x, corrected_flow_y), 'target': (final_target_x, final_target_y), 'similarity': similarity, 'phase_shift': (dx, dy), 'angle': angle, 'confidence': confidence}

def perform_fine_correction(frame1, frame2, source_pixel, coarse_target_pixel, template_radius, search_radius, good_quality_threshold):
    """Performs fine-tuning."""
    src_x, src_y = source_pixel
    source_color = frame1[src_y, src_x]
    template, _ = extract_region(frame1, src_x, src_y, template_radius)
    search_area, search_bounds = extract_region(frame2, coarse_target_pixel[0], coarse_target_pixel[1], search_radius)
    if template.shape[0] != int(2 * template_radius) or search_area.shape[0] != int(2 * search_radius):
        return None
    res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left_in_search = max_loc
    patch_center_x = search_bounds[0] + top_left_in_search[0] + template_radius
    patch_center_y = search_bounds[1] + top_left_in_search[1] + template_radius
    patch_target = (patch_center_x, patch_center_y)
    h, w = frame2.shape[:2]
    if not (0 <= patch_target[0] < w and 0 <= patch_target[1] < h):
        return None
    patch_center_color = frame2[int(patch_center_y), int(patch_center_x)]
    patch_center_similarity = calculate_pixel_quality(source_color, patch_center_color)
    final_target_coords = patch_target
    final_similarity = patch_center_similarity
    if not is_good_quality(patch_center_similarity, good_quality_threshold):
        search_dim = int(template_radius * 2)
        for dx, dy in generate_spiral_path(search_dim, search_dim):
            check_x = patch_target[0] + dx
            check_y = patch_target[1] + dy
            if 0 <= check_x < w and 0 <= check_y < h:
                target_color = frame2[int(check_y), int(check_x)]
                similarity = calculate_pixel_quality(source_color, target_color)
                if is_good_quality(similarity, good_quality_threshold):
                    final_target_coords = (check_x, check_y)
                    final_similarity = similarity
                    break
    final_target_x, final_target_y = final_target_coords
    final_flow_x = src_x - final_target_x
    final_flow_y = src_y - final_target_y
    res_vis = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    res_vis = cv2.cvtColor(res_vis, cv2.COLOR_GRAY2BGR)
    cv2.circle(res_vis, max_loc, 5, (0,255,0), 1)
    return {'flow': (final_flow_x, final_flow_y), 'target': (final_target_x, final_target_y), 'similarity': final_similarity, 'confidence': max_val, 'template': template, 'search_area': search_area, 'response_map': res_vis, 'match_location': max_loc}

def generate_quality_frame_fast(frame1, frame2, flow, good_quality_threshold):
    """Generate a quality visualization frame - optimized version"""
    if flow is None: return np.zeros_like(frame1)
    h, w = frame1.shape[:2]
    fh, fw = flow.shape[:2]
    quality_frame = np.zeros((h, w, 3), dtype=np.uint8)
    scale_x = w / fw if fw > 0 else 1.0
    scale_y = h / fh if fh > 0 else 1.0
    if (fh != h or fw != w):
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        flow[:, :, 0] *= scale_x
        flow[:, :, 1] *= scale_y
    downsample_factor = 1
    if h * w > 1000000: downsample_factor = 4
    elif h * w > 500000: downsample_factor = 2
    step = downsample_factor
    for y in range(0, h, step):
        for x in range(0, w, step):
            flow_x, flow_y = flow[y, x, 0], flow[y, x, 1]
            target_x, target_y = x - flow_x, y - flow_y
            if (0 <= target_x < w and 0 <= target_y < h):
                src_color, target_color = frame1[y, x], frame2[int(target_y), int(target_x)]
                similarity = calculate_pixel_quality(src_color, target_color)
                if is_good_quality(similarity, good_quality_threshold):
                    intensity = int(255 * (similarity - 0.5) * 2)
                    color = [0, np.clip(intensity, 0, 255), 0]
                else:
                    intensity = int(255 * (1.0 - similarity))
                    color = [intensity, 0, 0]
            else:
                color = [255, 0, 0]
            y_end, x_end = min(y + step, h), min(x + step, w)
            quality_frame[y:y_end, x:x_end] = color
    return quality_frame

def generate_quality_frame_gpu(frame1, frame2, flow, device, good_quality_threshold):
    """Generate a quality visualization frame using GPU (PyTorch) for acceleration."""
    h, w = frame1.shape[:2]
    frame1_t = torch.from_numpy(frame1).to(device).float() / 255.0
    frame2_t = torch.from_numpy(frame2).to(device).float() / 255.0
    flow_t = torch.from_numpy(flow).to(device).float()
    fh, fw = flow_t.shape[:2]
    if fh != h or fw != w:
        flow_t = torch.nn.functional.interpolate(flow_t.permute(2, 0, 1).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)
        flow_t[..., 0] *= w / fw
        flow_t[..., 1] *= h / fh
    y_coords, x_coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    target_x, target_y = x_coords - flow_t[..., 0], y_coords - flow_t[..., 1]
    out_of_bounds_mask = (target_x < 0) | (target_x >= w) | (target_y < 0) | (target_y >= h)
    target_x_int, target_y_int = target_x.long(), target_y.long()
    target_x_clamped, target_y_clamped = torch.clamp(target_x_int, 0, w - 1), torch.clamp(target_y_int, 0, h - 1)
    sampled_colors = frame2_t[target_y_clamped, target_x_clamped]
    src_colors = frame1_t
    rgb_distance = torch.sqrt(torch.sum((src_colors - sampled_colors) ** 2, dim=-1))
    rgb_similarity = 1.0 - (rgb_distance / 1.732)
    abs_diff = torch.mean(torch.abs(src_colors - sampled_colors), dim=-1)
    abs_similarity = 1.0 - abs_diff
    cosine_sim = torch.nn.functional.cosine_similarity(src_colors, sampled_colors, dim=-1)
    cosine_similarity = (cosine_sim + 1.0) / 2.0
    overall_similarity = (rgb_similarity + abs_similarity + cosine_similarity) / 3.0
    green_intensity = torch.clamp((overall_similarity - 0.5) * 2.0, 0, 1)
    red_intensity = torch.clamp(1.0 - overall_similarity, 0, 1)
    is_good_mask = overall_similarity > good_quality_threshold
    quality_frame_t = torch.zeros_like(src_colors)
    quality_frame_t[..., 1] = torch.where(is_good_mask, green_intensity, 0)
    quality_frame_t[..., 0] = torch.where(is_good_mask, 0, red_intensity)
    red_color = torch.tensor([1.0, 0.0, 0.0], device=device)
    quality_frame_t = torch.where(out_of_bounds_mask.unsqueeze(-1), red_color, quality_frame_t)
    return (quality_frame_t * 255).byte().cpu().numpy()

def get_highest_available_lod(frame_idx, flow_data, lod_data, max_lod_levels):
    """Get the highest available LOD level for a frame from provided data"""
    for lod_level in range(max_lod_levels - 1, -1, -1):
        if lod_level == 0:
            if flow_data is not None:
                return 0, flow_data
        lod = lod_data.get((frame_idx, lod_level))
        if lod is not None:
            return lod_level, lod
    return None, None

def worker_process(worker_id, frame_indices, frames, flow_data_cache, lod_data_cache, device_str, max_lod_levels, flow_files, constants):
    """The main function for a worker process."""
    pid = os.getpid()
    print(f"Worker {worker_id} (PID: {pid}) starting, processing {len(frame_indices)} frames: {frame_indices[0]} to {frame_indices[-1]}")
    
    worker_results = []
    
    for frame_idx in frame_indices:
        start_time = time.time()
        
        flow_data = flow_data_cache.get(frame_idx)
        if flow_data is None:
            print(f"Worker skipping frame {frame_idx}: No flow data.")
            worker_results.append({'initial': 0, 'final': 0, 'improved': 0, 'failed': 0, 'skipped': True})
            continue

        frame1 = frames[frame_idx]
        frame2 = frames[frame_idx + 1]
        flow = flow_data.copy()
        h, w = frame1.shape[:2]
        fh, fw = flow.shape[:2]

        if 'cuda' in device_str:
            quality_map = generate_quality_frame_gpu(frame1, frame2, flow, torch.device(device_str), constants['GOOD_QUALITY_THRESHOLD'])
        else:
            quality_map = generate_quality_frame_fast(frame1, frame2, flow, constants['GOOD_QUALITY_THRESHOLD'])

        bad_pixels_y, bad_pixels_x = np.where(quality_map[:, :, 0] > 0)
        bad_pixels_coords = list(zip(bad_pixels_x, bad_pixels_y))
        initial_error_count = len(bad_pixels_coords)

        if not bad_pixels_coords:
            worker_results.append({'initial': 0, 'final': 0, 'improved': 0, 'failed': 0, 'skipped': False})
            continue

        lod_level, lod_flow = get_highest_available_lod(frame_idx, flow_data, lod_data_cache, max_lod_levels)
        if lod_flow is None:
            print(f"Worker skipping frame {frame_idx}: No LOD data for correction.")
            worker_results.append({'initial': initial_error_count, 'final': initial_error_count, 'improved': 0, 'failed': initial_error_count, 'skipped': True})
            continue

        scale_x_frame_to_flow, scale_y_frame_to_flow = (fw / w if w > 0 else 1.0), (fh / h if h > 0 else 1.0)
        lod_h, lod_w = lod_flow.shape[:2]
        lod_scale_x_frame_to_lod, lod_scale_y_frame_to_lod = lod_w / w, lod_h / h
        
        improved_pixels_set = set()

        for (orig_x, orig_y) in bad_pixels_coords:
            # --- Calculate original similarity for comparison ---
            flow_y_coord = int(orig_y * scale_y_frame_to_flow)
            flow_x_coord = int(orig_x * scale_x_frame_to_flow)
            
            # Clamp flow coordinates to be safe
            flow_y_coord = max(0, min(flow_y_coord, fh - 1))
            flow_x_coord = max(0, min(flow_x_coord, fw - 1))

            original_flow_x = flow[flow_y_coord, flow_x_coord, 0] / scale_x_frame_to_flow
            original_flow_y = flow[flow_y_coord, flow_x_coord, 1] / scale_y_frame_to_flow
            
            original_target_x = orig_x - original_flow_x
            original_target_y = orig_y - original_flow_y
            
            original_similarity = 0.0
            # Clamp target coordinates to frame dimensions
            clamped_target_x = int(round(original_target_x))
            clamped_target_y = int(round(original_target_y))
            
            if (0 <= clamped_target_x < w and 0 <= clamped_target_y < h):
                original_similarity = calculate_pixel_quality(
                    frame1[orig_y, orig_x], 
                    frame2[clamped_target_y, clamped_target_x]
                )

            # --- Perform Correction ---
            lod_x, lod_y = max(0, min(int(orig_x * lod_scale_x_frame_to_lod), lod_w - 1)), max(0, min(int(orig_y * lod_scale_y_frame_to_lod), lod_h - 1))
            lod_flow_x, lod_flow_y = lod_flow[lod_y, lod_x, 0] / lod_scale_x_frame_to_lod, lod_flow[lod_y, lod_x, 1] / lod_scale_y_frame_to_lod
            
            coarse_result = perform_coarse_correction(frame1, frame2, (orig_x, orig_y), (lod_flow_x, lod_flow_y), constants['DETAIL_ANALYSIS_REGION_SIZE'])
            final_flow_vec, final_similarity = coarse_result['flow'], coarse_result['similarity']

            if coarse_result['similarity'] < constants['FINE_CORRECTION_THRESHOLD']:
                fine_result = perform_fine_correction(frame1, frame2, (orig_x, orig_y), coarse_result['target'], constants['TEMPLATE_RADIUS'], constants['SEARCH_RADIUS'], constants['GOOD_QUALITY_THRESHOLD'])
                if fine_result and fine_result['similarity'] > coarse_result['similarity']:
                    final_flow_vec, final_similarity = fine_result['flow'], fine_result['similarity']

            if is_good_quality(final_similarity, constants['GOOD_QUALITY_THRESHOLD']) or (final_similarity > original_similarity):
                final_flow_x_scaled, final_flow_y_scaled = final_flow_vec[0] * scale_x_frame_to_flow, final_flow_vec[1] * scale_y_frame_to_flow
                flow_y_coord, flow_x_coord = int(orig_y * scale_y_frame_to_flow), int(orig_x * scale_x_frame_to_flow)
                flow_y_coord, flow_x_coord = max(0, min(flow_y_coord, fh - 1)), max(0, min(flow_x_coord, fw - 1))
                flow[flow_y_coord, flow_x_coord] = [final_flow_x_scaled, final_flow_y_scaled]
                if not is_good_quality(final_similarity, constants['GOOD_QUALITY_THRESHOLD']):
                    improved_pixels_set.add((orig_x, orig_y))
        
        if 'cuda' in device_str:
            new_quality_map = generate_quality_frame_gpu(frame1, frame2, flow, torch.device(device_str), constants['GOOD_QUALITY_THRESHOLD'])
        else:
            new_quality_map = generate_quality_frame_fast(frame1, frame2, flow, constants['GOOD_QUALITY_THRESHOLD'])
            
        final_error_y, final_error_x = np.where(new_quality_map[:, :, 0] > 0)
        final_error_count = len(final_error_y)
        
        try:
            original_flow_path = Path(flow_files[frame_idx])
            corrected_flow_dir = original_flow_path.parent.with_name(original_flow_path.parent.name + "_corrected")
            corrected_flow_dir.mkdir(exist_ok=True)
            new_flow_path = corrected_flow_dir / original_flow_path.name
            if new_flow_path.suffix == '.flo':
                writeFlow(str(new_flow_path), flow)
            elif new_flow_path.suffix == '.npz':
                np.savez_compressed(str(new_flow_path), flow=flow)
        except Exception as e:
            print(f"Worker for frame {frame_idx} failed to save: {e}")

        duration = time.time() - start_time
        corrected = initial_error_count - final_error_count
        success_rate = (corrected / initial_error_count * 100) if initial_error_count > 0 else 0
        print(f"  [Worker {worker_id}] Frame {frame_idx:4d} | Errors: {initial_error_count:4d} -> {final_error_count:4d} | Success: {success_rate:5.1f}% | Time: {duration:.2f}s")
        
        worker_results.append({'initial': initial_error_count, 'final': final_error_count, 'improved': 0, 'failed': 0, 'skipped': False})
    
    print(f"Worker {worker_id} finished.")
    return worker_results 