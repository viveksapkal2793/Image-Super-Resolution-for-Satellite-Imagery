import scipy.io
import matplotlib.pyplot as plt
import numpy as np

def visualize_mat_file(mat_file_path, band_index=0):
    # Load the .mat file
    mat_contents = scipy.io.loadmat(mat_file_path)
    
    # Extract the image data
    image_data = mat_contents['pred']
    
    # Check if the band index is within the valid range
    if band_index < 0 or band_index >= image_data.shape[2]:
        raise ValueError(f"Invalid band index {band_index}. Must be between 0 and {image_data.shape[2] - 1}.")
    
    # Select the specified band
    band_data = image_data[:, :, band_index]
    
    # Display the image
    plt.imshow(np.clip(band_data, 0, 1), cmap='gray')
    plt.title(f'Super-Resolution Image - Band {band_index}')
    plt.axis('off')
    plt.show()

# Example usage
visualize_mat_file('demo_data/sr_output_imgs/result_sr_2D_it12000.mat', band_index=1)