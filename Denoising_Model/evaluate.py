import argparse
import os
import cv2
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from models import DMCN_prelu
from utils import weights_init_kaiming
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def load_model(model_path, device):
    model = DMCN_prelu()
    model.apply(weights_init_kaiming)
    model = model.to(device)
    
    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle the case where the model was wrapped in nn.DataParallel
    if 'module.' in list(state_dict.keys())[0]:
        # Create a new state dictionary without the 'module.' prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.eval()
    return model

def process_image(image_path, device):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize the image to be divisible by 16
    h, w = img.shape
    h = (h // 16) * 16
    w = (w // 16) * 16
    img_resized = cv2.resize(img, (w, h))
    img_tensor = np.expand_dims(img_resized, 0)
    img_tensor = np.expand_dims(img_tensor, 0)
    img_tensor = torch.FloatTensor(img_tensor / 255.0).to(device)
    return img_resized, img_tensor

def save_image(tensor, output_path):
    img = tensor.cpu().detach().numpy().squeeze()
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(output_path, img)
    return img

def evaluate(model, noisy_path, output_path, clean_path, device):
    # Process noisy image
    noisy_img, noisy_tensor = process_image(noisy_path, device)
    
    # Load clean image if provided
    if clean_path and os.path.exists(clean_path):
        clean_img = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
        # Resize to match noisy image dimensions
        h, w = noisy_img.shape
        clean_img = cv2.resize(clean_img, (w, h))
    else:
        clean_img = None
    
    # Denoise the image
    with torch.no_grad():
        noisy_tensor = Variable(noisy_tensor)
        output = model(noisy_tensor)
        output = torch.clamp(output, 0., 1.)
    
    # Save denoised image
    denoised_img = save_image(output, output_path)
    
    return clean_img, noisy_img, denoised_img

def calculate_psnr(original, processed):
    """Calculate PSNR between original and processed images"""
    # Convert to float in range [0,1]
    original = original.astype(np.float32) / 255.0
    processed = processed.astype(np.float32) / 255.0
    return compare_psnr(original, processed)

def calculate_ssim(original, processed, data_range=1.0):
    """Calculate SSIM between original and processed images"""
    # Convert to float in range [0,1]
    original = original.astype(np.float32) / 255.0
    processed = processed.astype(np.float32) / 255.0
    return compare_ssim(original, processed, data_range=data_range)

def display_images(clean, noisy, denoised, title="Original, Noisy, and Denoised", metrics=None):
    """Display original, noisy, and denoised images side by side with metrics"""
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(clean, cmap='gray')
    plt.title('Original Clean')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(noisy, cmap='gray')
    plt.title('Noisy')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(denoised, cmap='gray')
    plt.title('Denoised')
    plt.axis('off')
    
    if metrics:
        title = (f"{title}\n"
                f"Noisy vs Clean: PSNR: {metrics['noisy_psnr']:.2f} dB, SSIM: {metrics['noisy_ssim']:.4f}\n"
                f"Denoised vs Clean: PSNR: {metrics['denoised_psnr']:.2f} dB, SSIM: {metrics['denoised_ssim']:.4f}")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DnCNN Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--noisy_dir", type=str, required=True, help="Directory containing noisy images to evaluate")
    parser.add_argument("--clean_dir", type=str, help="Directory containing clean (original) images for comparison")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output images")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use for evaluation")
    parser.add_argument("--show_images", action="store_true", help="Display images after processing")
    opt = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    model = load_model(opt.model_path, device)
    
    # Lists to store metrics for all images
    noisy_psnr_values = []
    noisy_ssim_values = []
    denoised_psnr_values = []
    denoised_ssim_values = []
    
    print(f"Evaluating images from {opt.noisy_dir}...")
    if opt.clean_dir:
        print(f"Using clean images from {opt.clean_dir} for comparison")
        print("--------------------------------------------------------------------------------")
        print("Image Name              | Noisy PSNR | Noisy SSIM | Denoised PSNR | Denoised SSIM")
        print("--------------------------------------------------------------------------------")
    else:
        print("No clean images provided. Only processing denoising.")
        print("------------------------------------------------------")
        print("Image Name              | Processing Complete")
        print("------------------------------------------------------")

    for image_name in os.listdir(opt.noisy_dir):
        noisy_path = os.path.join(opt.noisy_dir, image_name)
        output_path = os.path.join(opt.output_dir, image_name)
        
        # If clean directory is provided, find the corresponding clean image
        if opt.clean_dir:
            clean_path = os.path.join(opt.clean_dir, image_name)
            if not os.path.exists(clean_path):
                print(f"Warning: Clean image for {image_name} not found. Skipping metrics.")
                clean_path = None
        else:
            clean_path = None
        
        # Process the image
        clean_img, noisy_img, denoised_img = evaluate(model, noisy_path, output_path, clean_path, device)
        
        # Calculate metrics if clean image is available
        if clean_img is not None:
            # Calculate noisy vs clean metrics
            noisy_psnr = calculate_psnr(clean_img, noisy_img)
            noisy_ssim = calculate_ssim(clean_img, noisy_img, data_range=1.0)
            noisy_psnr_values.append(noisy_psnr)
            noisy_ssim_values.append(noisy_ssim)
            
            # Calculate denoised vs clean metrics
            denoised_psnr = calculate_psnr(clean_img, denoised_img)
            denoised_ssim = calculate_ssim(clean_img, denoised_img, data_range=1.0)
            denoised_psnr_values.append(denoised_psnr)
            denoised_ssim_values.append(denoised_ssim)
            
            print(f"{image_name[:20]:<20} | {noisy_psnr:9.2f} | {noisy_ssim:.4f} | {denoised_psnr:12.2f} | {denoised_ssim:.4f}")
            
            # if opt.show_images:
            #     metrics = {
            #         'noisy_psnr': noisy_psnr,
            #         'noisy_ssim': noisy_ssim,
            #         'denoised_psnr': denoised_psnr,
            #         'denoised_ssim': denoised_ssim
            #     }
            #     display_images(clean_img, noisy_img, denoised_img, 
            #                   title=f"Clean, Noisy, and Denoised: {image_name}", 
            #                   metrics=metrics)
        else:
            print(f"{image_name[:20]:<20} | Processed (no metrics)")
    
    # Calculate and print average metrics
    if noisy_psnr_values:
        avg_noisy_psnr = sum(noisy_psnr_values) / len(noisy_psnr_values)
        avg_noisy_ssim = sum(noisy_ssim_values) / len(noisy_ssim_values)
        avg_denoised_psnr = sum(denoised_psnr_values) / len(denoised_psnr_values)
        avg_denoised_ssim = sum(denoised_ssim_values) / len(denoised_ssim_values)
        
        print("--------------------------------------------------------------------------------")
        print(f"AVERAGE                | {avg_noisy_psnr:9.2f} | {avg_noisy_ssim:.4f} | {avg_denoised_psnr:12.2f} | {avg_denoised_ssim:.4f}")
        print("--------------------------------------------------------------------------------")
        print(f"\nEvaluation completed. Processed {len(noisy_psnr_values)} images.")
        
        # Calculate improvement
        psnr_improvement = avg_denoised_psnr - avg_noisy_psnr
        ssim_improvement = avg_denoised_ssim - avg_noisy_ssim
        print(f"Average PSNR improvement: {psnr_improvement:.2f} dB")
        print(f"Average SSIM improvement: {ssim_improvement:.4f}")
        
    print(f"Results saved to: {opt.output_dir}")