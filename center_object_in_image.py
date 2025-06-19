#!/usr/bin/env python3
"""
Center Object in Image Script
Purpose: Automatically detect content in images with white backgrounds, 
         create bounding boxes, and center the object while preserving aspect ratio.

Perfect for centering FLUX-generated objects before background removal.

Usage:
python center_object_in_image.py --input image.png --output centered_image.png
python center_object_in_image.py --input image.png --output centered_image.png --padding 20
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
import argparse
import os
from pathlib import Path
from typing import Tuple, Optional

class ObjectCenterer:
    """Class to center objects in images with white backgrounds"""
    
    def __init__(self, white_threshold: int = 240, padding: int = 10):
        """
        Initialize the object centerer
        
        Args:
            white_threshold: Pixel values above this are considered white/background
            padding: Extra padding around the detected object (in pixels)
        """
        self.white_threshold = white_threshold
        self.padding = padding
    
    def detect_content_bbox(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the bounding box of non-white content in the image
        
        Args:
            image: Input image as numpy array (RGB or RGBA)
            
        Returns:
            Tuple of (x_min, y_min, x_max, y_max) or None if no content found
        """
        # Convert to grayscale for content detection
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                # Use alpha channel if available, otherwise convert RGB to grayscale
                alpha = image[:, :, 3]
                gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
                # Combine alpha and grayscale info
                gray = np.where(alpha > 0, gray, 255)
            else:  # RGB
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:  # Already grayscale
            gray = image
        
        # Create mask of non-white pixels
        content_mask = gray < self.white_threshold
        
        # Find contours of content
        contours, _ = cv2.findContours(
            content_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            print("⚠️ No content detected in image")
            return None
        
        # Find the bounding box that encompasses all content
        all_points = np.vstack(contours)
        x_min = np.min(all_points[:, :, 0])
        y_min = np.min(all_points[:, :, 1])
        x_max = np.max(all_points[:, :, 0])
        y_max = np.max(all_points[:, :, 1])
        
        # Add padding
        height, width = gray.shape
        x_min = max(0, x_min - self.padding)
        y_min = max(0, y_min - self.padding)
        x_max = min(width, x_max + self.padding)
        y_max = min(height, y_max + self.padding)
        
        return (x_min, y_min, x_max, y_max)
    
    def center_object(self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Center the detected object in the image
        
        Args:
            image: Input image as numpy array
            target_size: Optional target size (width, height). If None, uses original size
            
        Returns:
            Centered image as numpy array
        """
        original_height, original_width = image.shape[:2]
        
        if target_size is None:
            target_width, target_height = original_width, original_height
        else:
            target_width, target_height = target_size
        
        # Detect content bounding box
        bbox = self.detect_content_bbox(image)
        
        if bbox is None:
            print("⚠️ No content found, returning original image")
            return image
        
        x_min, y_min, x_max, y_max = bbox
        content_width = x_max - x_min
        content_height = y_max - y_min
        
        print(f"📦 Content bounding box: ({x_min}, {y_min}) to ({x_max}, {y_max})")
        print(f"📏 Content size: {content_width} x {content_height}")
        
        # Extract the content region
        content = image[y_min:y_max, x_min:x_max]
        
        # Create new image with white background
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                centered_image = np.full((target_height, target_width, 4), [255, 255, 255, 255], dtype=np.uint8)
            else:  # RGB
                centered_image = np.full((target_height, target_width, 3), [255, 255, 255], dtype=np.uint8)
        else:  # Grayscale
            centered_image = np.full((target_height, target_width), 255, dtype=np.uint8)
        
        # Calculate position to center the content
        center_x = target_width // 2
        center_y = target_height // 2
        
        # Calculate top-left corner for centering
        paste_x = center_x - content_width // 2
        paste_y = center_y - content_height // 2
        
        # Ensure the content fits within the target image
        paste_x = max(0, min(paste_x, target_width - content_width))
        paste_y = max(0, min(paste_y, target_height - content_height))
        
        # Paste the content into the centered position
        end_x = paste_x + content_width
        end_y = paste_y + content_height
        
        # Handle potential size mismatches
        if end_x > target_width:
            content = content[:, :target_width - paste_x]
            end_x = target_width
        if end_y > target_height:
            content = content[:target_height - paste_y]
            end_y = target_height
        
        centered_image[paste_y:end_y, paste_x:end_x] = content
        
        print(f"✅ Content centered at position ({paste_x}, {paste_y})")
        
        return centered_image
    
    def process_image_file(self, input_path: str, output_path: str, target_size: Optional[Tuple[int, int]] = None):
        """
        Process an image file and save the centered result
        
        Args:
            input_path: Path to input image
            output_path: Path to save output image
            target_size: Optional target size (width, height)
        """
        try:
            # Load image
            pil_image = Image.open(input_path)
            
            # Convert to RGB if necessary (handle RGBA, P, etc.)
            if pil_image.mode not in ['RGB', 'RGBA']:
                if pil_image.mode == 'P' and 'transparency' in pil_image.info:
                    pil_image = pil_image.convert('RGBA')
                else:
                    pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(pil_image)
            
            print(f"📷 Input image: {image_array.shape}")
            
            # Center the object
            centered_array = self.center_object(image_array, target_size)
            
            # Convert back to PIL and save
            centered_pil = Image.fromarray(centered_array)
            centered_pil.save(output_path)
            
            print(f"💾 Saved centered image to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            raise

def create_visualization(input_path: str, output_path: str, bbox_path: str, centerer: ObjectCenterer):
    """Create a visualization showing the detected bounding box"""
    try:
        # Load image
        pil_image = Image.open(input_path)
        if pil_image.mode not in ['RGB', 'RGBA']:
            pil_image = pil_image.convert('RGB')
        
        image_array = np.array(pil_image)
        
        # Detect bounding box
        bbox = centerer.detect_content_bbox(image_array)
        
        if bbox is None:
            print("⚠️ No content detected for visualization")
            return
        
        # Draw bounding box on original image
        viz_image = pil_image.copy()
        draw = ImageDraw.Draw(viz_image)
        
        x_min, y_min, x_max, y_max = bbox
        
        # Draw bounding box in red
        draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)
        
        # Add text label
        draw.text((x_min, y_min - 20), f"Content: {x_max-x_min}x{y_max-y_min}", fill='red')
        
        viz_image.save(bbox_path)
        print(f"📊 Saved bounding box visualization to: {bbox_path}")
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")

def main():
    parser = argparse.ArgumentParser(description="Center objects in images with white backgrounds")
    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument("--output", "-o", required=True, help="Output image path")
    parser.add_argument("--target-size", help="Target size as 'width,height' (e.g., '1024,1024')")
    parser.add_argument("--padding", type=int, default=20, help="Padding around detected object (default: 20)")
    parser.add_argument("--white-threshold", type=int, default=240, help="White threshold (0-255, default: 240)")
    parser.add_argument("--show-bbox", action="store_true", help="Create visualization showing detected bounding box")
    parser.add_argument("--batch", help="Process all images in a directory")
    
    args = parser.parse_args()
    
    # Parse target size
    target_size = None
    if args.target_size:
        try:
            width, height = map(int, args.target_size.split(','))
            target_size = (width, height)
            print(f"🎯 Target size: {width} x {height}")
        except ValueError:
            print("❌ Invalid target size format. Use 'width,height' (e.g., '1024,1024')")
            return
    
    # Create centerer
    centerer = ObjectCenterer(
        white_threshold=args.white_threshold,
        padding=args.padding
    )
    
    print(f"🔧 Settings: white_threshold={args.white_threshold}, padding={args.padding}")
    
    if args.batch:
        # Batch processing
        input_dir = Path(args.batch)
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in input_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"📁 Processing {len(image_files)} images from {input_dir}")
        
        for image_file in image_files:
            output_file = output_dir / f"centered_{image_file.name}"
            
            print(f"\n🔄 Processing: {image_file.name}")
            
            try:
                centerer.process_image_file(str(image_file), str(output_file), target_size)
                
                if args.show_bbox:
                    bbox_file = output_dir / f"bbox_{image_file.name}"
                    create_visualization(str(image_file), str(output_file), str(bbox_file), centerer)
                    
            except Exception as e:
                print(f"❌ Failed to process {image_file.name}: {e}")
                continue
        
        print(f"\n✅ Batch processing complete. Results saved to {output_dir}")
        
    else:
        # Single file processing
        if not os.path.exists(args.input):
            print(f"❌ Input file not found: {args.input}")
            return
        
        print(f"🔄 Processing: {args.input}")
        
        # Process the image
        centerer.process_image_file(args.input, args.output, target_size)
        
        # Create visualization if requested
        if args.show_bbox:
            bbox_path = args.output.replace('.', '_bbox.')
            create_visualization(args.input, args.output, bbox_path, centerer)
        
        print("✅ Processing complete!")

if __name__ == "__main__":
    main() 