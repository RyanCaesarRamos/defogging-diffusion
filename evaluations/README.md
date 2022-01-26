# Evaluations

To compare different defogging models, we use  PSNR, SSIM, FID, Inception Score, and a modified version of Perceptual Distance that relies on Inception-v3 instead of Inception-v1. These metrics can all be calculated using batches of samples, which we store in `.npz` (numpy) files.

# Download batches

We provide pre-computed sample batches for the reference datasets and our diffusion model. These are all stored in `.npz` format.

Each synthetic dataset batch contains 100 images for evaluation, while each I-HAZE batch contains seven images for evaluation.

Here are links to download all of the sample and reference batches:

 * NYU Depth Dataset V2 (synthetic): [reference batch](https://drive.google.com/file/d/1p9WWOpiXMtNjDSSvmXMUbi1kPdCSkBA-/view?usp=sharing)
   * [ADM](https://drive.google.com/file/d/1u0pO0rN3RsTSNRACCHFLXJtYURfo5kjr/view?usp=sharing)
* I-HAZE: [reference batch](https://drive.google.com/file/d/1XlW0dIdvrYOHq8LWZG5klaAb4PLiIg0C/view?usp=sharing)
   * [ADM](https://drive.google.com/file/d/1CXsh6jiG3bHY5lrxY03fYziS1AGl8Sn3/view?usp=sharing)

# Run evaluations

First, generate or download a batch of samples and download the corresponding reference batch for the given dataset. For this example, we'll use I-HAZE, so the refernce batch is `VIRTUAL_ihaze256.npz` and we can use the sample batch `admnet_ihaze256.npz`.

Next, run the `evaluator.py` script. The requirements of this script can be found in [requirements.txt](requirements.txt). Pass two arguments to the script: the reference batch and the sample batch. The script will download the Inception-v3 model used for evaluations into the current working directory (if it is not already present).

The output of the script will look something like this:

```
$ python evaluator.py VIRTUAL_ihaze256.npz admnet_ihaze256.npz
Computing evaluations...
PSNR: 15.387101643593114
SSIM: 0.6551572100261391
FID: 0.31589040831115855
Inception Score: 3.6453693204048396
Perceptual Distance: 0.6286566735597098
```
