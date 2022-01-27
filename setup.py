from setuptools import setup

setup(
    name="defogging-diffusion",
    packages=["defogging_diffusion"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm", "einops", "torchvision", "h5py"],
)
