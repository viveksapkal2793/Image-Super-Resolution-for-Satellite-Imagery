import argparse
import os
import cv2
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from models import DMCN_prelu
from utils import weights_init_kaiming
import numpy as np

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

def evaluate(model, image_path, output_path, device):
    original_img, img_tensor = process_image(image_path, device)
    with torch.no_grad():
        img_tensor = Variable(img_tensor)
        output = model(img_tensor)
        output = torch.clamp(output, 0., 1.)
    denoised_img = save_image(output, output_path)
    return original_img, denoised_img

def display_images(original, denoised, title="Original vs. Denoised"):
    """Display original and denoised images side by side"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(denoised, cmap='gray')
    plt.title('Denoised')
    plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DnCNN Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images to evaluate")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output images")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use for evaluation")
    parser.add_argument("--show_images", default=True, help="Display images after processing")
    opt = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # if not os.path.exists(opt.output_dir):
    #     os.makedirs(opt.output_dir)

    model = load_model(opt.model_path, device)

    for image_name in os.listdir(opt.image_dir):
        image_path = os.path.join(opt.image_dir, image_name)
        output_path = os.path.join(opt.output_dir, image_name)
        original, denoised = evaluate(model, image_path, output_path, device)
        print(f"Processed {image_name}")
        
        if opt.show_images:
            display_images(original, denoised, title=f"Original vs. Denoised: {image_name}")