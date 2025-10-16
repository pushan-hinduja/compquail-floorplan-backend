"""
Door Detection Script using YOLOv8
Detects doors in floor plan images or PDFs and saves cropped images of each detection.
"""

import os
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


def load_images_from_file(file_path, dpi=600):
    """
    Load images from either a PDF or image file.
    
    Args:
        file_path (str): Path to the input file (PDF or image)
        dpi (int): DPI for PDF conversion (default: 600 for better quality)
    
    Returns:
        list: List of tuples (page_number, PIL.Image) or [(1, image)] for single images
    """
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.pdf':
        print(f"Converting PDF to images at {dpi} DPI...")
        print("Note: Higher DPI = better detection but slower processing")
        try:
            # Use higher quality settings for PDF conversion
            images = convert_from_path(
                file_path, 
                dpi=dpi,
                fmt='png',  # Use PNG format for better quality
                thread_count=4  # Use multiple threads for faster conversion
            )
            print(f"Loaded {len(images)} page(s) from PDF")
            
            # Print image dimensions for debugging
            for i, img in enumerate(images):
                print(f"  Page {i+1} size: {img.size[0]}x{img.size[1]} pixels")
            
            # Return list of (page_number, image) tuples
            return [(i + 1, img) for i, img in enumerate(images)]
        except Exception as e:
            print(f"Error converting PDF: {e}")
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
            return [(1, image)]
        except Exception as e:
            print(f"Error loading image: {e}")
            return []


def detect_and_save_doors(image_path, output_folder, model_path='models/door_detection_model.pt', confidence=0.1, 
                         padding=20, min_size=None, dpi=600, save_converted=False,
                         enhance=True):
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
    
    Returns:
        dict: Dictionary containing:
            - door_count (int): Number of doors detected and saved
            - door_images (list): List of paths to saved door images
            - output_folder (str): Path to the output folder
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load images from file (handles both PDF and image files)
    images = load_images_from_file(image_path, dpi=dpi)
    
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
        
        # Run prediction
        results = model.predict(image, conf=confidence, verbose=False)
        
        # Filter for only "Door" detections
        page_door_count = 0
        for result in results:
            boxes = result.boxes
            
            for i, box in enumerate(boxes):
                # Get the class label
                class_id = int(box.cls[0])
                label = model.names[class_id]
                
                # Only process if it's a Door
                if label == 'Door':
                    # Get bounding box coordinates
                    # box.xyxy gives [x1, y1, x2, y2] format
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Convert to integers
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
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
                    door_images.append(output_path)
                    
                    page_door_count += 1
                    total_door_count += 1
                    print(f"Saved door {page_door_count}: {filename} (confidence: {conf_score:.2f})")
        
        print(f"Doors found on {'page ' + str(page_num) if len(images) > 1 else 'this image'}: {page_door_count}")
    
    print(f"\n{'='*50}")
    print(f"Total doors detected and saved: {total_door_count}")
    print(f"{'='*50}")
    
    return {
        "door_count": total_door_count,
        "door_images": door_images,
        "output_folder": output_folder
    }


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