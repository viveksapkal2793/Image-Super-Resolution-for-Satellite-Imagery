#!/usr/bin/env python
# delete_jpg.py

import os
import argparse
import glob
from tqdm import tqdm  # For progress bar

def find_jpg_files(directory, recursive=False):
    """
    Find all JPG files in the specified directory.
    
    Args:
        directory (str): Directory to search for JPG files
        recursive (bool): Whether to search subdirectories
    
    Returns:
        list: List of found JPG file paths
    """
    # Define patterns for JPG files (case insensitive)
    patterns = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']
    
    jpg_files = []
    
    if recursive:
        for pattern in patterns:
            jpg_files.extend(glob.glob(os.path.join(directory, '**', pattern), recursive=True))
    else:
        for pattern in patterns:
            jpg_files.extend(glob.glob(os.path.join(directory, pattern)))
    
    return jpg_files

def delete_jpg_files(directory, recursive=False, dry_run=False):
    """
    Delete all JPG files in the specified directory.
    
    Args:
        directory (str): Directory containing JPG files
        recursive (bool): Whether to search subdirectories
        dry_run (bool): If True, only list files without deleting
    
    Returns:
        tuple: (number of successfully deleted files, total number of JPG files)
    """
    # Ensure directory exists
    if not os.path.exists(directory):
        raise ValueError(f"Directory does not exist: {directory}")
    
    # Find JPG files
    jpg_files = find_jpg_files(directory, recursive)
    total_count = len(jpg_files)
    
    if total_count == 0:
        print("No JPG files found.")
        return 0, 0
    
    print(f"Found {total_count} JPG files.")
    
    if dry_run:
        print("Dry run mode: Files will not be deleted.")
        for file_path in jpg_files:
            print(f"Would delete: {file_path}")
        return 0, total_count
    
    # Confirm deletion
    confirmation = input(f"Are you sure you want to delete {total_count} JPG files? (yes/no): ")
    if confirmation.lower() != "yes":
        print("Operation cancelled.")
        return 0, total_count
    
    # Delete files
    success_count = 0
    print("Deleting files...")
    
    for file_path in tqdm(jpg_files):
        try:
            os.remove(file_path)
            success_count += 1
        except Exception as e:
            print(f"Error deleting {file_path}: {str(e)}")
    
    return success_count, total_count

def main():
    """Main entry point for the script."""
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description='Delete all JPG images from a directory.'
    )
    parser.add_argument(
        'directory', 
        help='Directory containing JPG images to delete'
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Also search and delete JPG files in subdirectories'
    )
    parser.add_argument(
        '--dry-run', '-d',
        action='store_true',
        help='Only list files that would be deleted without actually deleting them'
    )
    
    args = parser.parse_args()
    
    try:
        success_count, total_count = delete_jpg_files(
            args.directory, args.recursive, args.dry_run
        )
        
        if not args.dry_run:
            print(f"Deletion complete: {success_count}/{total_count} JPG files deleted successfully.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())