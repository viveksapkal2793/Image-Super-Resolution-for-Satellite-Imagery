import os
import argparse
import cv2
import torch
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.io import savemat
from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from Denoising_Model.models import DMCN_prelu
from Denoising_Model.utils import weights_init_kaiming
from Super_Resolution_Model.models.downsampler import Downsampler
from Super_Resolution_Model.models.skip import skip
from Super_Resolution_Model.utils.sr_utils import get_noise, get_params, optimize


def load_model(model_path, device):
    model = DMCN_prelu()
    model.apply(weights_init_kaiming)
    model = model.to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    if 'module.' in list(state_dict.keys())[0]:
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.eval()
    return model

def process_image(image_path, device):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    h = (h // 16) * 16
    w = (w // 16) * 16
    img = cv2.resize(img, (w, h))
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 0)
    img = torch.FloatTensor(img / 255.0).to(device)
    print('processed image for denoisng')
    return img

def save_image(tensor, output_path):
    img = tensor.cpu().detach().numpy().squeeze()
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(output_path, img)

def denoise_image(model, image_path, output_path, device):
    img = process_image(image_path, device)
    with torch.no_grad():
        img = Variable(img)
        output = model(img)
        output = torch.clamp(output, 0., 1.)
    print('denoised image')
    save_image(output, output_path)

def prepare_sr_image(input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    img = img / 255.0
    HR = cv2.resize(img, (192, 192))
    LR = cv2.resize(HR, (96, 96))
    bicubic = cv2.resize(LR, (192, 192), interpolation=cv2.INTER_CUBIC)
    nearest = cv2.resize(LR, (192, 192), interpolation=cv2.INTER_NEAREST)
    print('preprocessed image for super-resolution model input')
    savemat(output_path, {'HR': HR, 'LR': LR, 'bicubic': bicubic, 'nearest': nearest})

def train_sr_model(sr_input_path, output_dir):
    print('training super-resolution model')
    imgs = scipy.io.loadmat(sr_input_path)
    factor = 2
    band = 24

    input_depth = imgs['HR'].shape[2]
    method = '2D'
    pad = 'reflection'
    OPT_OVER = 'net'
    KERNEL_TYPE = 'lanczos2'
    show_every = 500
    save_every = 2000
    LR = 0.01
    tv_weight = 0.0
    OPTIMIZER = 'adam'
    num_iter = 12001
    reg_noise_std = 0.01

    dtype = torch.cuda.FloatTensor
    net_input = get_noise(input_depth, method, (imgs['HR'].shape[0], imgs['HR'].shape[1])).type(dtype).detach()
    net = skip(input_depth, imgs['HR'].shape[2],
               num_channels_down=[128]*5,
               num_channels_up=[128]*5,
               num_channels_skip=[4]*5,
               filter_size_up=3, filter_size_down=3, filter_skip_size=1,
               upsample_mode='bilinear',
               need1x1_up=False,
               need_sigmoid=False, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

    mse = torch.nn.MSELoss().type(dtype)
    img_LR_var = imgs['LR'].transpose(2, 0, 1)
    img_LR_var = torch.from_numpy(img_LR_var).type(dtype)
    img_LR_var = img_LR_var[None, :].cuda()
    downsampler = Downsampler(n_planes=imgs['HR'].shape[2], factor=factor, kernel_type=KERNEL_TYPE, phase=0.5, preserve_size=True).type(dtype)

    psnr_history = []
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    def closure():
        nonlocal i, net_input
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out_HR = net(net_input)
        out_LR = downsampler(out_HR)
        total_loss = mse(out_LR, img_LR_var)

        if tv_weight > 0:
            total_loss += tv_weight * tv_loss(out_HR)

        total_loss.backward()

        psnr_LR = compare_psnr(imgs['LR'].astype(np.float32), out_LR.detach().cpu().squeeze().numpy().transpose(1, 2, 0))
        psnr_HR = compare_psnr(imgs['HR'].astype(np.float32), out_HR.detach().cpu().squeeze().numpy().transpose(1, 2, 0))
        print(f'Iteration {i:05d}    PSNR_LR {psnr_LR:.3f}   PSNR_HR {psnr_HR:.3f}', end='\r')

        psnr_history.append([psnr_LR, psnr_HR])

        if i % show_every == 0:
            out_HR_np = out_HR.detach().cpu().squeeze().numpy().transpose(1, 2, 0)
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 15))
            ax1.imshow(imgs['HR'][:, :, band], cmap='gray', vmin=0.0, vmax=0.5)
            ax2.imshow(imgs['bicubic'][:, :, band], cmap='gray', vmin=0.0, vmax=0.5)
            ax3.imshow(np.clip(out_HR_np, 0, 1)[:, :, band], cmap='gray', vmin=0.0, vmax=0.5)
            plt.show()

        if i % save_every == 0:
            out_HR_np = out_HR.detach().cpu().squeeze().numpy().transpose(1, 2, 0)
            scipy.io.savemat(os.path.join(output_dir, f"result_sr_2D_it{i:05d}.mat"), {'pred': np.clip(out_HR_np, 0, 1)})

        i += 1
        return total_loss

    i = 0
    p = get_params(OPT_OVER, net, net_input)
    optimize(OPTIMIZER, p, closure, LR, num_iter)
    print('training complete')
    print('super-resolution images saved in demo_data/sr_output_imgs')

def main():
    parser = argparse.ArgumentParser(description="Denoising and Super-Resolution Pipeline")
    parser.add_argument("--denoise_model_path", type=str, required=True, help="Path to the trained denoising model")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing noisy and low-resolution images")
    parser.add_argument("--denoised_dir", type=str, required=True, help="Directory to save the denoised images")
    parser.add_argument("--sr_input_dir", type=str, required=True, help="Directory to save the super-resolution input images")
    parser.add_argument("--sr_output_dir", type=str, required=True, help="Directory to save the super-resolution output images")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use for evaluation")
    opt = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(opt.denoised_dir):
        os.makedirs(opt.denoised_dir)
    if not os.path.exists(opt.sr_input_dir):
        os.makedirs(opt.sr_input_dir)
    if not os.path.exists(opt.sr_output_dir):
        os.makedirs(opt.sr_output_dir)

    model = load_model(opt.denoise_model_path, device)

    for image_name in os.listdir(opt.input_dir):
        image_path = os.path.join(opt.input_dir, image_name)
        denoised_path = os.path.join(opt.denoised_dir, image_name)
        sr_input_path = os.path.join(opt.sr_input_dir, image_name.replace('.png', '.mat'))

        denoise_image(model, image_path, denoised_path, device)
        prepare_sr_image(denoised_path, sr_input_path)
        print(f"Processed {image_name}")

        train_sr_model(sr_input_path, opt.sr_output_dir)

if __name__ == "__main__":
    main()