#!/usr/bin/env python
# rgb_to_gray_png.py

import os
import argparse
import cv2
from tqdm import tqdm  # For progress bar

def convert_to_grayscale_png(input_path, output_path):
    """
    Convert a single RGB image to grayscale and save as PNG.
    
    Args:
        input_path (str): Path to the input RGB image
        output_path (str): Path to save the output grayscale PNG image
    
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    try:
        # Read the image
        img = cv2.imread(input_path)
        
        # Check if image was loaded successfully
        if img is None:
            print(f"Error: Could not read image {input_path}")
            return False
        
        # Convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Ensure the output file has .png extension
        output_path = os.path.splitext(output_path)[0] + '.png'
        
        # Save the grayscale image as PNG
        cv2.imwrite(output_path, gray_img)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def process_directory(input_dir, output_dir, extensions=None):
    """
    Convert all images in a directory from RGB to grayscale PNG.
    
    Args:
        input_dir (str): Directory containing RGB images
        output_dir (str): Directory to save grayscale PNG images
        extensions (list, optional): List of valid image extensions.
                                    Defaults to ['.jpg', '.jpeg', '.png', '.bmp']
    
    Returns:
        tuple: (number of successfully processed images, total number of images)
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Ensure directories exist
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(root, file))
    
    # Process each image
    success_count = 0
    total_count = len(image_files)
    
    print(f"Found {total_count} images to convert.")
    
    for input_path in tqdm(image_files):
        # Create output path with same relative structure but with .png extension
        rel_path = os.path.relpath(input_path, input_dir)
        output_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '.png')
        
        # Convert to grayscale and PNG
        if convert_to_grayscale_png(input_path, output_path):
            success_count += 1
    
    return success_count, total_count

def main():
    """Main entry point for the script."""
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description='Convert RGB images to grayscale PNG format.'
    )
    parser.add_argument(
        'input_dir', 
        help='Directory containing RGB images'
    )
    parser.add_argument(
        '--output_dir', '-o',
        default=None,
        help='Directory to save grayscale PNG images (default: input_dir + "_gray_png")'
    )
    parser.add_argument(
        '--extensions', '-e',
        nargs='+',
        default=['.jpg', '.jpeg', '.png', '.bmp'],
        help='List of image extensions to process (default: .jpg .jpeg .png .bmp)'
    )
    
    args = parser.parse_args()
    
    # Set output directory if not specified
    if args.output_dir is None:
        args.output_dir = args.input_dir + "_gray_png"
    
    # Process the directory
    try:
        success_count, total_count = process_directory(
            args.input_dir, args.output_dir, args.extensions
        )
        
        print(f"Conversion complete: {success_count}/{total_count} images processed successfully.")
        print(f"Grayscale PNG images saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())