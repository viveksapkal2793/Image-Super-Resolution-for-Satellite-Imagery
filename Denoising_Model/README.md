# Denoising Model for Satellite Imagery

This repository contains source code necessary for denoising noisy satellite images using a deep neural network with encoder-decoder style architecture.

## Model Training

The denoising model is trained using the `main.py` script, which implements a CNN-based denoising network. The script handles dataset loading, model training, and evaluation on validation data throughout the training process.

### Training Parameters

| Parameter | Description |
|-----------|-------------|
| `--debug` | Use 'Y' to train with only 5 images (for debugging), 'N' for full training |
| `--resume` | Whether to resume training from a saved model (True/False) |
| `--model_name` | Name of the model file to load when resuming training |
| `--start_epoch` | Starting epoch number (useful when resuming training) |
| `--preprocess` | Whether to run the data preprocessing step |
| `--batchSize` | Training batch size |
| `--epochs` | Number of training epochs |
| `--milestone1` | First epoch to reduce learning rate |
| `--milestone2` | Second epoch to reduce learning rate |
| `--lr` | Initial learning rate |
| `--outf` | Output folder for logs and models |
| `--train_id` | Training ID (used as subfolder name) |
| `--noiseL` | Noise level for training (sigma value) |
| `--val_noiseL` | Noise level for validation |
| `--gpu_id` | GPU ID to use for training |

### Training Command Example

```bash
# Basic training command
python main.py --debug N --batchSize 128 --epochs 50 --lr 1e-3 --noiseL 25 --train_id 01 --gpu_id 0

# Resume training from a checkpoint
python main.py --debug N --resume True --model_name net_best.pth --start_epoch 25 --epochs 100 --train_id 01 --gpu_id 0

# Train with data preprocessing
python main.py --preprocess True --debug N --batchSize 128 --epochs 50 --noiseL 25 --train_id 01
```

During training, the script will:
1. Train the model for specified number of epochs
2. Reduce learning rate at milestone epochs
3. Save tensorboard logs for loss and PSNR
4. Save the best model based on validation PSNR
5. Display training statistics

## Model Evaluation

The `evaluate.py` script allows you to evaluate a trained model on test images. The script supports both denoising images and calculating metrics by comparing with ground truth clean images.

### Evaluation Parameters

| Parameter | Description |
|-----------|-------------|
| `--model_path` | Path to the trained model file |
| `--noisy_dir` | Directory containing noisy images to evaluate |
| `--clean_dir` | Directory containing clean (original) images for comparison (optional) |
| `--output_dir` | Directory to save the denoised images |
| `--gpu_id` | GPU ID to use for evaluation |
| `--show_images` | Flag to display images after processing |

### Evaluation Command Examples

```bash
# Basic evaluation with metrics calculation
python evaluate.py --model_path logs/01/net_best.pth --noisy_dir test_data/noisy --clean_dir test_data/clean --output_dir results/denoised --gpu_id 0

# Evaluation without ground truth images
python evaluate.py --model_path logs/01/net_best.pth --noisy_dir test_data/noisy --output_dir results/denoised --gpu_id 0

# Evaluation with image display
python evaluate.py --model_path logs/01/net_best.pth --noisy_dir test_data/noisy --clean_dir test_data/clean --output_dir results/denoised --gpu_id 0 --show_images
```

The evaluation script will:
1. Process all images in the noisy directory
2. If clean images are provided, calculate:
   - PSNR and SSIM between noisy and clean images
   - PSNR and SSIM between denoised and clean images
3. Display images side by side (if `--show_images` is specified)
4. Save denoised images to the output directory
5. Print average metrics and improvements

## Creating Test Data

To create noisy test images from clean images, you can use these additional scripts:

```bash
# Add Gaussian noise to clean images
python add_noise.py path/to/clean_images path/to/output_noisy_images --noise 25
```

## Requirements
* Python 3
* PyTorch

### Python Packages:
* matplotlib
* cv2
* h5py
* numpy
* skimage
* tensorboardX
* torchvision
* tqdm

## Dataset Structure

When using `--preprocess True`, the script expects data in this structure:
```
data/
  ├── Train/
  │    └── (training images)
  └── Test/
       └── (testing images)
```

## Additional Notes

- For best results, train with a noise level (`--noiseL`) that matches your expected test data noise level
- The model automatically saves checkpoints at the end of each epoch and keeps the best-performing model
- Training logs can be viewed using TensorBoard by running `tensorboard --logdir=logs/01`
- Use GPU for significantly faster training and evaluation