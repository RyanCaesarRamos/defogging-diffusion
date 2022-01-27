# defogging-diffusion

This is a codebase for preliminary explorations into leveraging diffusion models for image defogging.

This repository is based on the paper [Diffusion Models Beat GANS on Image Synthesis](http://arxiv.org/abs/2105.05233) and its corresponding repository [openai/guided-diffusion](https://github.com/openai/guided-diffusion).

# Download pre-trained model

Below is one checkpoint for a model trained on the defogging task. Before using it, please review the corresponding [model card](model-card.md) to understand its intended use and limitations.

 * 256x256 defogger: [256x256_defogger.pt](https://drive.google.com/file/d/19wXyyjfH0yDwQFgcyHT7AFhhzN4w7bR0/view)

# Sampling from pre-trained model

To sample from this model, you can use the `defog_sample.py` script.
We assume that you have downloaded the relevant model checkpoints into a folder called `models/`.

For these examples, we will generate 100 samples with batch size 4. Feel free to change these values.

```
SAMPLE_FLAGS="--batch_size 4 --num_samples 100 --timestep_respacing 250"
```

For these runs, we assume you have paired foggy-clear images in `foggy.npz` and `clear.npz`.
 
```
MODEL_FLAGS="--attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 192 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True"
python super_res_sample.py $MODEL_FLAGS --model_path models/256x256_defogger.pt --foggy_data_dir foggy.npz --clear_data_dir clear.npz $SAMPLE_FLAGS
```

# Results

This table summarizes our results on image defogging:

| Dataset                          | val size | PSNR  | SSIM | FID   | IS   | PD   |
|----------------------------------|----------|-------|------|-------|------|------|
| NYU Depth Dataset V2 (synthetic) | 100      | 16.67 | 0.80 | 10.79 | 4.87 | 0.56 |
| I-HAZE                           | 7        | 15.39 | 0.66 | 31.59 | 3.65 | 0.63 |

Below are sample qualitative results of our model:

| NYU Depth Dataset V2 (synthetic) | |
|-|-|
| foggy input     | ![](https://github.com/RyanCaesarRamos/defogging-diffusion/blob/main/sample_results/nyu/foggy_input.png?raw=true)    |
| defogged output | ![](https://github.com/RyanCaesarRamos/defogging-diffusion/blob/main/sample_results/nyu/defogged_output.png?raw=true) |
| ground truth    | ![](https://github.com/RyanCaesarRamos/defogging-diffusion/blob/main/sample_results/nyu/ground_truth.png?raw=true)    |

| I-HAZE | |
|-|-|
| foggy input     | ![](https://github.com/RyanCaesarRamos/defogging-diffusion/blob/main/sample_results/ihaze/foggy_input.png?raw=true)     |
| defogged output | ![](https://github.com/RyanCaesarRamos/defogging-diffusion/blob/main/sample_results/ihaze/defogged_output.png?raw=true) |
| ground truth    | ![](https://github.com/RyanCaesarRamos/defogging-diffusion/blob/main/sample_results/ihaze/ground_truth.png?raw=true)    |

# Training models

With the following as sample hyperparameters:

```
MODEL_FLAGS="--num_channels 192 --num_res_blocks 2 --learn_sigma True --image_size 256 --num_heads 2 --attention_resolutions 32,16,8 --resblock_updown True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --use_scale_shift_norm True"
TRAIN_FLAGS="--lr 3e-4 --batch_size 4 --lr_anneal_steps 700"
```

a model can be trained with the following command:

```
python scripts/defog_train.py --foggy_data_dir path/to/foggy/images --clear_data_dir path/to/clear/images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

More details on training can be found at the [repository](https://github.com/openai/guided-diffusion) this codebase was forked from. For the specific hyperparameters the hosted checkpoint was trained with, please see [model-card.md](model-card.md).
