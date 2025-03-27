#!/usr/bin/env python
# add_noise.py - Add noise to images for denoising model evaluation

import os
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

def add_noise(img_tensor, noise_level=25):
    """
    Add Gaussian noise to an image tensor
    
    Args:
        img_tensor (torch.Tensor): Clean image tensor [0,1]
        noise_level (float): Noise level (sigma)
        
    Returns:
        torch.Tensor: Noisy image tensor [0,1]
    """
    noise = torch.FloatTensor(img_tensor.size()).normal_(mean=0, std=noise_level/255.)
    noisy_img = img_tensor + noise
    return torch.clamp(noisy_img, 0., 1.)

def process_image(image_path, output_path, noise_level=25):
    """
    Process a single image: load, convert to tensor, add noise, save result
    
    Args:
        image_path (str): Path to input clean image
        output_path (str): Path to save noisy image
        noise_level (float): Noise level (sigma)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return False
            
        # Convert to float [0,1]
        img = img.astype(np.float32) / 255.0
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # Add channel dimension
        
        # Add noise
        noisy_tensor = add_noise(img_tensor, noise_level)
        
        # Convert back to numpy and scale to [0,255]
        noisy_img = noisy_tensor.squeeze().numpy() * 255.0
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the noisy image
        cv2.imwrite(output_path, noisy_img)
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False

def process_directory(input_dir, output_dir, noise_level=25, extensions=None):
    """
    Process all images in a directory
    
    Args:
        input_dir (str): Directory containing clean images
        output_dir (str): Directory to save noisy images
        noise_level (float): Noise level (sigma)
        extensions (list): Valid image extensions
        
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
    
    print(f"Found {total_count} images to process.")
    print(f"Adding Gaussian noise with sigma = {noise_level}")
    
    for input_path in tqdm(image_files):
        # Create output path with same relative structure
        rel_path = os.path.relpath(input_path, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        
        # Process the image
        if process_image(input_path, output_path, noise_level):
            success_count += 1
    
    return success_count, total_count

def main():
    """Main entry point for the script."""
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description='Add noise to clean images for denoising evaluation.'
    )
    parser.add_argument(
        'input_dir', 
        help='Directory containing clean images'
    )
    parser.add_argument(
        'output_dir',
        help='Directory to save noisy images'
    )
    parser.add_argument(
        '--noise', '-n',
        type=float,
        default=25.0,
        help='Noise level (sigma) to add (default: 25.0)'
    )
    parser.add_argument(
        '--extensions', '-e',
        nargs='+',
        default=['.jpg', '.jpeg', '.png', '.bmp'],
        help='List of image extensions to process (default: .jpg .jpeg .png .bmp)'
    )
    
    args = parser.parse_args()
    
    # Process the directory
    try:
        success_count, total_count = process_directory(
            args.input_dir, args.output_dir, args.noise, args.extensions
        )
        
        print(f"Processing complete: {success_count}/{total_count} images processed successfully.")
        print(f"Noisy images saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())