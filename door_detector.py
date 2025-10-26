"""
Door Detection Script using YOLOv8
Detects doors in floor plan images or PDFs and saves cropped images of each detection.
"""

import os
import shutil
import argparse
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageEnhance, ImageOps
import torch
from pdf2image import convert_from_path
import numpy as np


def preprocess_image(image, enhance_for_detection=False):
    """
    Preprocess image to improve detection quality.
    Especially useful for PDF conversions that may be washed out.
    
    Args:
        image (PIL.Image): Input image
        enhance_for_detection (bool): Apply aggressive preprocessing for better detection
    
    Returns:
        PIL.Image: Preprocessed image
    """
    if not enhance_for_detection:
        return image
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Increase contrast to make lines more defined
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)  # Increase contrast by 50%
    
    # Increase sharpness to make edges crisper
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)  # Double the sharpness
    
    # Optionally adjust brightness to ensure white background
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.1)  # Slightly brighter
    
    return image
import numpy as np


from pdf2image import convert_from_path


def create_adaptive_grid(image, target_section_size=2000, min_grid_size=2, max_grid_size=8):
    """
    Create an adaptive grid based on image size for large floor plans.
    
    Args:
        image (PIL.Image): Input image
        target_section_size (int): Target size for each section in pixels
        min_grid_size (int): Minimum grid size (e.g., 2x2)
        max_grid_size (int): Maximum grid size (e.g., 8x8)
    
    Returns:
        list: List of section dictionaries with image, bounds, and grid position
    """
    width, height = image.size
    
    # Calculate optimal grid size
    grid_x = max(min_grid_size, min(max_grid_size, width // target_section_size))
    grid_y = max(min_grid_size, min(max_grid_size, height // target_section_size))
    
    # Calculate section dimensions
    section_width = width // grid_x
    section_height = height // grid_y
    
    # Calculate overlap (25% of section size)
    overlap_x = int(section_width * 0.25)
    overlap_y = int(section_height * 0.25)
    
    sections = []
    
    for row in range(grid_y):
        for col in range(grid_x):
            # Calculate section bounds with overlap
            x1 = max(0, col * section_width - (overlap_x if col > 0 else 0))
            y1 = max(0, row * section_height - (overlap_y if row > 0 else 0))
            x2 = min(width, (col + 1) * section_width + (overlap_x if col < grid_x - 1 else 0))
            y2 = min(height, (row + 1) * section_height + (overlap_y if row < grid_y - 1 else 0))
            
            # Crop the section
            section = image.crop((x1, y1, x2, y2))
            
            sections.append({
                'image': section,
                'bounds': (x1, y1, x2, y2),
                'grid_pos': (row, col),
                'section_id': f"section_{row}_{col}"
            })
    
    return sections


def merge_duplicate_detections(detections, overlap_threshold=0.3, size_ratio_threshold=0.5):
    """
    Merge detections that overlap significantly to handle duplicates from overlapping sections.
    Uses improved logic to avoid missing doors.
    
    Args:
        detections (list): List of detection dictionaries
        overlap_threshold (float): Minimum overlap ratio to consider duplicates
        size_ratio_threshold (float): Maximum size ratio to consider merging (0.5 = 50% size difference)
    
    Returns:
        list: Merged detections with duplicates removed
    """
    if not detections:
        return []
    
    merged = []
    used = set()
    
    for i, det1 in enumerate(detections):
        if i in used:
            continue
        
        # Find all detections that overlap with this one
        overlapping = [i]
        for j, det2 in enumerate(detections[i+1:], i+1):
            if j in used:
                continue
            
            # Calculate overlap between bounding boxes
            overlap_ratio = calculate_bbox_overlap(det1, det2)
            
            # Calculate size ratio to avoid merging very different sized doors
            size_ratio = calculate_size_ratio(det1, det2)
            
            # Only merge if they overlap significantly AND are similar in size
            if overlap_ratio > overlap_threshold and size_ratio > size_ratio_threshold:
                overlapping.append(j)
        
        # Improved selection logic
        if len(overlapping) == 1:
            # No overlaps, keep as is
            merged.append(detections[overlapping[0]])
        else:
            # Multiple overlapping detections - use smart selection
            best_detection = select_best_detection([detections[idx] for idx in overlapping])
            merged.append(best_detection)
        
        used.update(overlapping)
    
    return merged


def calculate_size_ratio(det1, det2):
    """
    Calculate size ratio between two detections to avoid merging very different sized doors.
    
    Args:
        det1, det2 (dict): Detection dictionaries with 'bbox' key
    
    Returns:
        float: Size ratio (0.0 to 1.0, where 1.0 means identical size)
    """
    bbox1 = det1.get('bbox', [0, 0, 0, 0])
    bbox2 = det2.get('bbox', [0, 0, 0, 0])
    
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    if area1 == 0 or area2 == 0:
        return 0.0
    
    # Return the ratio of the smaller area to the larger area
    return min(area1, area2) / max(area1, area2)


def select_best_detection(overlapping_detections):
    """
    Select the best detection from overlapping ones using multiple criteria.
    
    Args:
        overlapping_detections (list): List of overlapping detection dictionaries
    
    Returns:
        dict: Best detection based on multiple criteria
    """
    if len(overlapping_detections) == 1:
        return overlapping_detections[0]
    
    # Score each detection based on multiple criteria
    scored_detections = []
    
    for detection in overlapping_detections:
        score = 0
        
        # Confidence score (0-1)
        confidence = detection.get('confidence', 0)
        score += confidence * 0.4  # 40% weight for confidence
        
        # Size score (prefer medium-sized detections, not too small or too large)
        bbox = detection.get('bbox', [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        
        # Prefer doors between 1000 and 10000 pixels (typical door size range)
        if 1000 <= area <= 10000:
            size_score = 1.0
        elif area < 1000:
            size_score = area / 1000  # Smaller penalty for small doors
        else:
            size_score = max(0.1, 10000 / area)  # Penalty for very large doors
        
        score += size_score * 0.3  # 30% weight for size
        
        # Section coverage score (prefer detections from sections with fewer total detections)
        section_id = detection.get('section_id', '')
        # This would need to be calculated based on section detection counts
        # For now, give equal weight
        score += 0.3  # 30% weight for section coverage
        
        scored_detections.append((score, detection))
    
    # Return the detection with highest score
    scored_detections.sort(key=lambda x: x[0], reverse=True)
    return scored_detections[0][1]


def calculate_bbox_overlap(det1, det2):
    """
    Calculate overlap ratio between two bounding boxes.
    
    Args:
        det1, det2 (dict): Detection dictionaries with 'bbox' key
    
    Returns:
        float: Overlap ratio (0.0 to 1.0)
    """
    bbox1 = det1.get('bbox', [0, 0, 0, 0])
    bbox2 = det2.get('bbox', [0, 0, 0, 0])
    
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0  # No intersection
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0


def load_images_from_file(file_path, dpi=600, save_debug_images=False, debug_folder="debug_conversion"):
    """
    Load images from either a PDF or image file.
    
    Args:
        file_path (str): Path to the input file (PDF or image)
        dpi (int): DPI for PDF conversion (default: 600 for better quality)
        save_debug_images (bool): Save converted images for debugging (default: False)
        debug_folder (str): Folder to save debug images (default: "debug_conversion")
    
    Returns:
        list: List of tuples (page_number, PIL.Image) or [(1, image)] for single images
    """
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.pdf':
        print(f"Converting PDF to images at {dpi} DPI...")
        print("Note: Higher DPI = better detection but slower processing")
        
        # Try different DPI values if the image is too large
        dpi_values_to_try = [dpi, 800, 600, 400, 300]
        
        for attempt_dpi in dpi_values_to_try:
            try:
                print(f"  Attempting conversion at {attempt_dpi} DPI...")
                
                # Use higher quality settings for PDF conversion
                images = convert_from_path(
                    file_path, 
                    dpi=attempt_dpi,
                    fmt='png',  # Use PNG format for better quality
                    thread_count=4,  # Use multiple threads for faster conversion
                    use_pdftocairo=True,  # Use pdftocairo for better quality
                    grayscale=False,  # Keep color information
                    transparent=False  # Ensure solid background
                )
                
                # Check if any image is too large
                max_pixels = 150_000_000  # 150 million pixels limit
                too_large = False
                
                for i, img in enumerate(images):
                    pixel_count = img.size[0] * img.size[1]
                    if pixel_count > max_pixels:
                        print(f"  Page {i+1} too large: {pixel_count:,} pixels (limit: {max_pixels:,})")
                        too_large = True
                        break
                
                if too_large and attempt_dpi > 300:
                    print(f"  Image too large at {attempt_dpi} DPI, trying lower DPI...")
                    continue
                
                print(f"  Successfully converted at {attempt_dpi} DPI")
                print(f"Loaded {len(images)} page(s) from PDF")
                
                # Print image dimensions for debugging
                for i, img in enumerate(images):
                    pixel_count = img.size[0] * img.size[1]
                    print(f"  Page {i+1} size: {img.size[0]}x{img.size[1]} pixels ({pixel_count:,} total pixels)")
                    print(f"  Page {i+1} mode: {img.mode}")
                
                # Save debug images if requested
                if save_debug_images:
                    os.makedirs(debug_folder, exist_ok=True)
                    for i, img in enumerate(images):
                        debug_path = os.path.join(debug_folder, f"page_{i+1}_dpi_{attempt_dpi}.png")
                        img.save(debug_path, quality=95, optimize=False)
                        print(f"  Saved debug image: {debug_path}")
                
                # Return list of (page_number, image) tuples
                return [(i + 1, img) for i, img in enumerate(images)]
                
            except Exception as e:
                if "exceeds limit" in str(e) or "decompression bomb" in str(e):
                    print(f"  Image too large at {attempt_dpi} DPI: {e}")
                    if attempt_dpi > 300:
                        continue
                    else:
                        break
                else:
                    print(f"Error converting PDF at {attempt_dpi} DPI: {e}")
                    break
        
        print("❌ Could not convert PDF - all DPI values resulted in images that are too large")
        print("💡 Try using a smaller PDF or a different PDF file")
        print("\nMake sure poppler is installed:")
        print("  - Mac: brew install poppler")
        print("  - Ubuntu/Debian: sudo apt-get install poppler-utils")
        print("  - Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/")
        return []
    else:
        # Handle image files
        try:
            image = Image.open(file_path)
            print(f"Loaded image: {file_path}")
            print(f"Image size: {image.size[0]}x{image.size[1]} pixels")
            print(f"Image mode: {image.mode}")
            return [(1, image)]
        except Exception as e:
            print(f"Error loading image: {e}")
            return []


def detect_and_save_doors(image_path, output_folder, model_path='models/door_detection_model.pt', confidence=0.15, 
                         padding=20, min_size=None, dpi=600, save_converted=False,
                         enhance=True, save_debug_images=False, debug=False, use_adaptive_grid=True):
    """
    Detect doors in a floor plan image or PDF and save cropped images of each detection.
    
    Args:
        image_path (str): Path to the input floor plan image or PDF
        output_folder (str): Path to the folder where cropped door images will be saved
        model_path (str): Path to the YOLOv8 model file (default: 'models/door_detection_model.pt')
        confidence (float): Confidence threshold for detection (default: 0.25)
        padding (int): Pixels to add around each detection for context (default: 20)
        min_size (int): Minimum size to resize images to (default: None, no resizing)
        dpi (int): DPI for PDF conversion (default: 600, higher = better quality)
        save_converted (bool): Save the converted PDF pages for quality inspection (default: False)
        enhance (bool): Apply image enhancement for better detection (default: True, recommended for PDFs)
        debug (bool): Print all detections with coordinates and confidence (default: False)
    
    Returns:
        dict: Dictionary containing:
            - door_count (int): Number of doors detected and saved
            - door_images (list): List of paths to saved door images
            - output_folder (str): Path to the output folder
    """
    # Clean up existing output folder to ensure fresh run
    if os.path.exists(output_folder):
        print(f"🧹 Cleaning existing output folder: {output_folder}")
        shutil.rmtree(output_folder)
    
    # Create fresh output folder
    os.makedirs(output_folder, exist_ok=True)
    print(f"📁 Created fresh output folder: {output_folder}")
    
    # Load images from file (handles both PDF and image files)
    images = load_images_from_file(image_path, dpi=dpi, save_debug_images=save_debug_images, debug_folder=output_folder)
    
    if not images:
        print("Failed to load any images from the file.")
        return {
            "door_count": 0,
            "door_images": [],
            "output_folder": output_folder
        }
    
    # Fix PyTorch 2.6+ loading issue by temporarily disabling weights_only
    # This is safe since we trust the model file from ultralytics
    original_load = torch.load
    
    def safe_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    
    torch.load = safe_load
    
    # Load the YOLO model
    try:
        model = YOLO(model_path)
        print(f"Loaded model: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        torch.load = original_load
        return {
            "door_count": 0,
            "door_images": [],
            "output_folder": output_folder
        }
    
    torch.load = original_load  # Restore original torch.load
    
    # Process each page/image
    total_door_count = 0
    door_images = []  # List to track saved door image paths
    is_pdf = Path(image_path).suffix.lower() == '.pdf'
    
    # Safety check - ensure door_images is always defined
    if 'door_images' not in locals():
        door_images = []
    
    for page_num, image in images:
        page_prefix = f"page{page_num}_" if len(images) > 1 else ""
        print(f"\n{'='*50}")
        print(f"Processing {'page ' + str(page_num) if len(images) > 1 else 'image'}...")
        print(f"Image size: {image.size}")
        print(f"{'='*50}")
        
        # Optionally save the converted image BEFORE enhancement for comparison
        if save_converted and is_pdf:
            converted_filename = f"{page_prefix}converted_raw.png"
            converted_path = os.path.join(output_folder, converted_filename)
            image.save(converted_path, quality=95)
            print(f"Saved raw converted PDF page: {converted_filename}")
        
        # Apply enhancement if enabled (especially useful for PDF conversions)
        if enhance and is_pdf:
            print("Applying image enhancement for better detection...")
            image = preprocess_image(image, enhance_for_detection=True)
            
            # Save enhanced version if requested
            if save_converted:
                enhanced_filename = f"{page_prefix}converted_enhanced.png"
                enhanced_path = os.path.join(output_folder, enhanced_filename)
                image.save(enhanced_path, quality=95)
                print(f"Saved enhanced PDF page: {enhanced_filename}")
        
        # Choose processing method based on file type and size
        try:
            if is_pdf and use_adaptive_grid:
                # Use adaptive grid for large PDFs
                print(f"🔧 Using adaptive grid processing for large PDF...")
                page_door_count, page_door_images = process_image_with_adaptive_grid(
                    image, model, confidence, padding, min_size, output_folder, 
                    page_prefix, debug, page_num
                )
                door_images.extend(page_door_images)
            else:
                # Use standard processing for PNGs or when adaptive grid is disabled
                page_door_count, page_door_images = process_image_standard(
                    image, model, confidence, padding, min_size, output_folder, 
                    page_prefix, debug, page_num
                )
                door_images.extend(page_door_images)
        except Exception as e:
            print(f"❌ Error processing page {page_num}: {e}")
            print(f"   Continuing with next page...")
            page_door_count = 0
            page_door_images = []
        
        total_door_count += page_door_count
        print(f"Doors found on {'page ' + str(page_num) if len(images) > 1 else 'this image'}: {page_door_count}")

    print(f"\n{'='*50}")
    print(f"Total doors detected and saved: {total_door_count}")
    print(f"{'='*50}")
    
    # Final safety check
    if 'door_images' not in locals():
        door_images = []
    
    return {
        "door_count": total_door_count,
        "door_images": door_images,
        "output_folder": output_folder
    }


def process_image_standard(image, model, confidence, padding, min_size, output_folder, page_prefix, debug, page_num):
    """Process image using standard YOLO detection (for PNGs or small images)."""
    # Run prediction
    results = model.predict(image, conf=confidence, verbose=False)
    
    # Debug: Print all detections if debug mode is enabled
    if debug:
        print(f"\n🔍 DEBUG: All detections from YOLO model:")
        print(f"  Confidence threshold: {confidence}")
        print(f"  Total detections found: {sum(len(result.boxes) for result in results)}")
        print(f"  Image size: {image.size}")
        print("-" * 60)
    
    # Filter for only "Door" detections
    page_door_count = 0
    page_door_images = []
    all_detections = 0
    
    for result in results:
        boxes = result.boxes
        
        for i, box in enumerate(boxes):
            all_detections += 1
            
            # Get the class label and confidence
            class_id = int(box.cls[0])
            label = model.names[class_id]
            conf_score = float(box.conf[0])
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Debug: Print all detections
            if debug:
                print(f"  Detection {all_detections}: {label} (conf: {conf_score:.3f})")
                print(f"    Coordinates: ({x1}, {y1}) to ({x2}, {y2})")
                print(f"    Size: {x2-x1}x{y2-y1} pixels")
                print(f"    Above threshold: {'✓' if conf_score >= confidence else '✗'}")
            
            # Only process if it's a Door and above confidence threshold
            if label == 'Door' and conf_score >= confidence:
                    
                    # Add padding around the detection
                    img_width, img_height = image.size
                    x1_padded = max(0, x1 - padding)
                    y1_padded = max(0, y1 - padding)
                    x2_padded = min(img_width, x2 + padding)
                    y2_padded = min(img_height, y2 + padding)
                    
                    # Crop the image using the padded bounding box coordinates
                    cropped_image = image.crop((x1_padded, y1_padded, x2_padded, y2_padded))
                    
                    # Optionally resize to minimum size for better quality
                    if min_size:
                        crop_width, crop_height = cropped_image.size
                        if crop_width < min_size or crop_height < min_size:
                            # Calculate scaling factor to meet minimum size
                            scale = max(min_size / crop_width, min_size / crop_height)
                            new_width = int(crop_width * scale)
                            new_height = int(crop_height * scale)
                            # Use LANCZOS for high-quality upscaling
                            cropped_image = cropped_image.resize((new_width, new_height), Image.LANCZOS)
                    
                    # Get confidence score
                    conf_score = float(box.conf[0])
                    
                    # Create filename with page number (if multi-page), door number and confidence
                    filename = f"{page_prefix}door_{page_door_count + 1}_conf_{conf_score:.2f}.png"
                    output_path = os.path.join(output_folder, filename)
                    
                    # Save the cropped image with high quality
                    cropped_image.save(output_path, quality=95, optimize=False)
                    
                    # Add the saved image path to our list
                    page_door_count += 1
                    page_door_images.append(output_path)
                    print(f"Saved door {page_door_count}: {filename} (confidence: {conf_score:.2f})")
    
    # Debug: Print summary
    if debug:
        print(f"\n🔍 DEBUG SUMMARY:")
        print(f"  Total detections: {all_detections}")
        print(f"  Doors processed: {page_door_count}")
        print(f"  Confidence threshold: {confidence}")
        print("-" * 60)
    
    return page_door_count, page_door_images


def process_image_with_adaptive_grid(image, model, confidence, padding, min_size, output_folder, page_prefix, debug, page_num):
    """Process large PDF image using adaptive grid sections."""
    width, height = image.size
    
    # Create adaptive grid
    sections = create_adaptive_grid(image, target_section_size=2000, min_grid_size=2, max_grid_size=8)
    
    print(f"📐 Created {len(sections)} sections for image {width}x{height}")
    if debug:
        print(f"  Grid size: {max(s['grid_pos'][0] for s in sections) + 1}x{max(s['grid_pos'][1] for s in sections) + 1}")
        print(f"  Section size: ~{sections[0]['image'].size if sections else 'N/A'}")
    
    all_detections = []
    page_door_count = 0
    page_door_images = []
    
    # Process each section
    for section_idx, section in enumerate(sections):
        section_image = section['image']
        section_bounds = section['bounds']
        grid_pos = section['grid_pos']
        
        if debug:
            print(f"\n🔍 Processing section {section_idx + 1}/{len(sections)} (grid {grid_pos[0]},{grid_pos[1]})")
            print(f"  Section size: {section_image.size}")
            print(f"  Section bounds: {section_bounds}")
        
        # Run YOLO on this section
        section_results = model.predict(section_image, conf=confidence, verbose=False)
        
        # Process detections in this section
        section_detections = 0
        for result in section_results:
            boxes = result.boxes
            
            for box in boxes:
                # Get detection info
                class_id = int(box.cls[0])
                label = model.names[class_id]
                conf_score = float(box.conf[0])
                
                # Get coordinates relative to section
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Convert to full image coordinates
                full_x1 = section_bounds[0] + x1
                full_y1 = section_bounds[1] + y1
                full_x2 = section_bounds[0] + x2
                full_y2 = section_bounds[1] + y2
                
                if debug:
                    print(f"    Detection: {label} (conf: {conf_score:.3f})")
                    print(f"      Section coords: ({x1}, {y1}) to ({x2}, {y2})")
                    print(f"      Full coords: ({full_x1}, {full_y1}) to ({full_x2}, {full_y2})")
                
                # Only process doors
                if label == 'Door' and conf_score >= confidence:
                    section_detections += 1
                    
                    # Store detection for merging
                    all_detections.append({
                        'bbox': [full_x1, full_y1, full_x2, full_y2],
                        'confidence': conf_score,
                        'section_id': section['section_id'],
                        'grid_pos': grid_pos
                    })
        
        if debug:
            print(f"  Section detections: {section_detections}")
    
    # Merge duplicate detections from overlapping sections
    if all_detections:
        print(f"🔄 Merging {len(all_detections)} detections from overlapping sections...")
        if debug:
            print(f"  Pre-merge detections:")
            for i, det in enumerate(all_detections):
                bbox = det['bbox']
                print(f"    {i+1}: bbox={bbox}, conf={det['confidence']:.3f}, section={det['section_id']}")
        
        merged_detections = merge_duplicate_detections(all_detections, overlap_threshold=0.3, size_ratio_threshold=0.5)
        print(f"✅ Merged to {len(merged_detections)} unique doors")
        
        if debug:
            print(f"  Post-merge detections:")
            for i, det in enumerate(merged_detections):
                bbox = det['bbox']
                print(f"    {i+1}: bbox={bbox}, conf={det['confidence']:.3f}, section={det['section_id']}")
            print(f"  Merged {len(all_detections)} → {len(merged_detections)} (removed {len(all_detections) - len(merged_detections)} duplicates)")
        
        # Save cropped images for each unique door
        for door_idx, detection in enumerate(merged_detections):
            x1, y1, x2, y2 = detection['bbox']
            conf_score = detection['confidence']
            
            # Add padding
            img_width, img_height = image.size
            x1_padded = max(0, x1 - padding)
            y1_padded = max(0, y1 - padding)
            x2_padded = min(img_width, x2 + padding)
            y2_padded = min(img_height, y2 + padding)
            
            # Crop the door
            cropped_image = image.crop((x1_padded, y1_padded, x2_padded, y2_padded))
            
            # Resize if needed
            if min_size:
                crop_width, crop_height = cropped_image.size
                if crop_width < min_size or crop_height < min_size:
                    scale = max(min_size / crop_width, min_size / crop_height)
                    new_width = int(crop_width * scale)
                    new_height = int(crop_height * scale)
                    cropped_image = cropped_image.resize((new_width, new_height), Image.LANCZOS)
            
            # Save the cropped image
            filename = f"{page_prefix}door_{door_idx + 1}_conf_{conf_score:.2f}.png"
            output_path = os.path.join(output_folder, filename)
            cropped_image.save(output_path, quality=95, optimize=False)
            
            page_door_count += 1
            page_door_images.append(output_path)
            print(f"Saved door {page_door_count}: {filename} (confidence: {conf_score:.2f})")
    
    return page_door_count, page_door_images


def main():
    """
    Main function to run from command line.
    """
    parser = argparse.ArgumentParser(
        description='Detect doors in floor plan images or PDFs and save cropped detections.'
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to the input floor plan image or PDF'
    )
    parser.add_argument(
        'output_folder',
        type=str,
        help='Path to the folder where cropped door images will be saved'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/door_detection_model.pt',
        help='Path to the YOLOv8 model file (default: models/door_detection_model.pt)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.25,
        help='Confidence threshold for detection (default: 0.25)'
    )
    parser.add_argument(
        '--padding',
        type=int,
        default=20,
        help='Pixels to add around each detection for context (default: 20)'
    )
    parser.add_argument(
        '--min-size',
        type=int,
        default=None,
        help='Minimum size to upscale images to (e.g., 256 or 512)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=600,
        help='DPI for PDF conversion (default: 600). Try 800-1200 for very detailed plans.'
    )
    parser.add_argument(
        '--save-converted',
        action='store_true',
        help='Save the converted PDF page(s) to compare quality with original'
    )
    parser.add_argument(
        '--no-enhance',
        action='store_true',
        help='Disable image enhancement (enhancement is ON by default for PDFs)'
    )
    
    args = parser.parse_args()
    
    # Run detection
    result = detect_and_save_doors(
        image_path=args.image_path,
        output_folder=args.output_folder,
        model_path=args.model,
        confidence=args.confidence,
        padding=args.padding,
        min_size=args.min_size,
        dpi=args.dpi,
        save_converted=args.save_converted,
        enhance=not args.no_enhance  # Enhancement is ON by default
    )
    
    # Print structured result for command line usage
    print(f"\nDetection Results:")
    print(f"  Doors found: {result['door_count']}")
    print(f"  Output folder: {result['output_folder']}")
    print(f"  Door images saved: {len(result['door_images'])}")


if __name__ == "__main__":
    main()