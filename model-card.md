# Overview

This repository hosts a single model trained on defogging foggy images synthesized from the NYU Depth Dataset V2 at resolution 256x256.

# Dataset

This model was trained on a synthetic dataset generated from [NYU Depth Dataset V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).
Here, we describe characteristics of this synthetic dataset which impact model behavior:

**NYU Depth Dataset V2 (synthetic)**: The original dataset contains 1449 pairs of images and depth maps. The synthetic dataset was created by processing image-depth pairs and through the atmospheric scattering model <img src="https://render.githubusercontent.com/render/math?math=I=J(e^{-\beta d}) %2B A(1 - e^{-\beta d})"> where I is the synthetic foggy image, J is the original image, d is the depth from viewer, A is intensity, and beta is density. A was sampled from the range (0.7, 1.0) and beta was sampled from the range (0.2, 0.6). Images were resized to height 256 before being cropped randomly to 256x256. Each image-depth pair in the dataset was used to generate ten unique foggy samples.

 * The atmospheric scattering model assumes that fog is homogenous, which may not be realistic for real-life scenarios.
 * All images from the dataset are indoor images.

# Hyperparameters and Training

This model was finetuned off of a modified version of OpenAI's [64x64 -&gt; 256x256 upsampler checkpoint](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64_256_upsampler.pt) from their repo [openai/guided-diffusion](https://github.com/openai/guided-diffusion) where weights required by class conditional training were removed.

Training data was generated with [nyu_depth_dataset_v2.py](datasets/nyu_depth_dataset_v2.py) using the default settings.

Assuming the modified checkpoint is located at `models/64_256_upsampler.pt`, this model was trained with the following commands:

```
MODEL_FLAGS="--num_channels 192 --num_res_blocks 2 --learn_sigma True --image_size 256 --num_heads 2 --attention_resolutions 32,16,8 --resblock_updown True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --use_scale_shift_norm True"
TRAIN_FLAGS="--lr 3e-4 --batch_size 4 --lr_anneal_steps 700"
python scripts/defog_train.py --foggy_data_dir path/to/foggy/images --clear_data_dir path/to/clear/images --resume-checkpoint models/modified_64_256_upsampler.pt $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

# Performance

This models is intended to generate fog-free versions of foggy images.
Model performance has been measured in terms of PSNR, SSIM, FID, IS, and a modified version of PD that relies on Inception-v3 instead of Inception-v1.
The latter three metrics all rely on the representations of a [pre-trained Inception-v3 model](https://arxiv.org/abs/1512.00567),
which was trained on ImageNet, and so is likely to focus more on the ImageNet classes (such as animals) than on other visual features (such as human faces).

Qualitatively, the samples produced by this models often leave behind dark areas and and oversaturated colors.

# Intended Use

These models are intended to be used for research purposes only.
In particular, they can be used as a baseline for generative modeling research, or as a starting point to build off of for such research.

These models are not intended to be commercially deployed.
Additionally, they are not intended to be used to create propaganda or offensive imagery.

# Limitations

This model often creates dark areas in its samples and oversaturates colors. Furthermore, the atmospheric scattering model used to generate the training data assumes that fog is homogenous, which may not be realistic for real-life scenarios and cause the model to underperform on real-life foggy iamges. Also, the model was trained solely on indoor images, and has not been tested or trained on outdoor foggy images. Lastly, the ability of this model to create harmful targeted imagery was not probed.
