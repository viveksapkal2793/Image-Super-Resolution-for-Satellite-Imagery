# Deep Image Prior Technique for Single Image Super-Resolution

## Model Training and Evaluation

The super-resolution model is based on the Deep Image Prior (DIP) technique, which leverages the structure of a convolutional neural network as a handcrafted prior for image restoration tasks. Unlike traditional deep learning approaches, DIP does not require pre-training on large datasets - instead, it fits a randomly-initialized neural network to a single degraded image.

### Model Architecture

The model uses a "skip" architecture with encoder-decoder structure and skip connections:

- **Encoder path**: Progressive downsampling with convolutional layers
- **Decoder path**: Progressive upsampling with skip connections from encoder
- **Skip connections**: Preserve spatial details that might be lost during downsampling

### How to Run the Model

#### 1. Prepare the Data

You can either use the provided MAT files or convert your own images:

```python
# Convert a PNG image to the required MAT format
input_image = "path/to/your/image.png"
output_mat = "data/sr/your_image.mat"
convert_png_to_mat(input_image, output_mat)

# Load the MAT file for processing
imgs = scipy.io.loadmat(output_mat)
```

The MAT file should contain:
- 'HR': High-resolution original image
- 'LR': Low-resolution image
- 'bicubic': Bicubic upsampled version
- 'nearest': Nearest-neighbor upsampled version

#### 2. Set Parameters

Adjust the following parameters according to your needs:

```python
input_depth = imgs['HR'].shape[2]  # Number of input channels
method = '2D'                       # Image processing method
pad = 'reflection'                  # Padding type
OPT_OVER = 'net'                    # What to optimize
KERNEL_TYPE = 'lanczos2'            # Downsampling kernel
LR = 0.01                           # Learning rate
num_iter = 12001                    # Number of optimization iterations
reg_noise_std = 0.01                # Regularization noise standard deviation
```

#### 3. Run the Optimization

```python
# Initialize the network
net_input = get_noise(input_depth, method, (imgs['HR'].shape[0], imgs['HR'].shape[1])).type(dtype).detach()

net = skip(input_depth, imgs['HR'].shape[2],
           num_channels_down=[128]*5,
           num_channels_up=[128]*5,
           num_channels_skip=[4]*5,  
           filter_size_up=3, filter_size_down=3, filter_skip_size=1,
           upsample_mode='bilinear',
           need1x1_up=False,
           need_sigmoid=False, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

# Run the optimization
i = 0
p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `input_depth` | Number of input channels (equals number of bands in hyperspectral data) |
| `method` | Processing method (2D for standard images) |
| `num_channels_down` | Number of channels in each downsampling layer |
| `num_channels_up` | Number of channels in each upsampling layer |
| `num_channels_skip` | Number of channels in each skip connection |
| `upsample_mode` | Mode for upsampling ('nearest' or 'bilinear') |
| `LR` | Learning rate for optimization |
| `num_iter` | Number of iterations for optimization |
| `reg_noise_std` | Standard deviation of noise added for regularization |
| `tv_weight` | Weight for total variation regularization |
| `show_every` | How often to display results during optimization |
| `save_every` | How often to save intermediate results |

### Monitoring Progress

The optimization process displays metrics at each iteration:

```
Iteration 12000    PSNR_LR 35.421   PSNR_HR 27.384   SSIM_LR 0.9823   SSIM_HR 0.8756
```

- **PSNR_LR**: Peak Signal-to-Noise Ratio between the model's low-resolution output and the input low-resolution image
- **PSNR_HR**: PSNR between the model's super-resolution output and the ground-truth high-resolution image
- **SSIM_LR**: Structural Similarity Index between low-resolution images
- **SSIM_HR**: SSIM between high-resolution images

### Visualization

The notebook will display visual comparisons at intervals specified by `show_every`:
- Original high-resolution image (ground truth)
- Bicubic upsampled image (baseline)
- Current super-resolution result from the model

### Output

The final super-resolution result is saved as a MAT file:
```
results/result_sr_2D_it12000.mat
```

You can visualize the output using the provided visualization functions or external tools like MATLAB.

### Tips for Better Results

1. **Adjust `reg_noise_std`**: Try values between 0.01-0.1 to find the best regularization
2. **Increase iterations**: More iterations often lead to better results (try 15000-20000)
3. **Tune architecture**: Adjust channel counts in `num_channels_down/up/skip` for different complexity
4. **Change learning rate**: Try reducing learning rate for finer details
5. **Band selection**: For hyperspectral data, visualization of specific bands can provide insights

### Requirements
- python = 3.6
- pytorch = 0.4
- numpy
- scipy
- matplotlib
- scikit-image
- jupyter

### Some results

Super-Resolution:

![image sr](https://github.com/viveksapkal2793/Image-Super-Resolution-for-Satellite-Imagery/raw/main/Super_Resolution_Model/figs/dip.png)