"""
Convert an LSUN lmdb database into a directory of images.
"""

import argparse
import os

from PIL import Image
import h5py
import torch
from torch import nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from einops import rearrange
from tqdm import tqdm


def read_data(mat_path):
    data = h5py.File(mat_path)
    for img, depth in zip(data['images'], data['depths']):
        img = rearrange(torch.from_numpy(img / 255), 'c w h -> c h w')
        depth = rearrange(torch.from_numpy(depth), 'w h -> h w')
        yield img, depth


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def generate_random_values(val_min, val_max, num_values):
    assert val_max > val_min, "val_max must be greater than val_min"
    val_range = val_max - val_min
    return (torch.rand(num_values) * val_range + val_min).tolist()
        
        
def synthesize_foggy_image(img, depth, a, beta):
    t = rearrange(torch.exp(-beta * depth), 'h w -> () h w')
    foggy_img = img * t + a * (1 - t)
    return foggy_img


def group_resize_random_crop(imgs, size):
    imgs = torch.cat(imgs, dim=0)
    imgs = T.Resize(size)(imgs)
    imgs = T.RandomCrop(size)(imgs)
    imgs = imgs.chunk(imgs.size(0) // 3, dim=0)
    imgs = [TF.to_pil_image(img) for img in imgs]
    return imgs


def generate_foggy_image_batch(img, depth, a_range, beta_range, num_images):
    a_list = generate_random_values(*a_range, num_images)
    beta_list = generate_random_values(*beta_range, num_images)
    for a, beta in zip(a_list, beta_list):
        foggy_img = synthesize_foggy_image(img, depth, a, beta)
        yield foggy_img
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", help="new image size", type=int, default=256)
    parser.add_argument("--a-range", help="range of A", type=str, default='0.7,1.0')
    parser.add_argument("--beta-range", help="range of beta", type=str, default='0.2,0.6')
    parser.add_argument("--test-reserve", help="number of images to reserve for test set", type=int, default=100)
    parser.add_argument("--num-reuse", help="number of times to synthesize from single image", type=int, default=10)
    parser.add_argument("--seed", help="random seed for dataset generation", type=int, default=1986)
    parser.add_argument("data_path", help="path to NYU Depth Dataset V2 mat file")
    parser.add_argument("out_dir", help="path to output directory")
    args = parser.parse_args()
    
    data = read_data(args.data_path)

    create_dir(args.out_dir)
    for l1 in ('train', 'test'):
        create_dir(os.path.join(args.out_dir, l1))
        for l2 in ('clear', 'foggy'):
            create_dir(os.path.join(args.out_dir, l1, l2))
        
    a_range = eval(f'{args.a_range}')
    beta_range = eval(f'{args.beta_range}')
    
    torch.manual_seed(args.seed)
    
    for i, (img, depth) in tqdm(enumerate(data), total=len(h5py.File(args.data_path)['images'])):
        if i < args.test_reserve:
            split = 'test'
        else:
            i = (i - args.test_reserve)
            split = 'train'
            
        for j, foggy_img in enumerate(generate_foggy_image_batch(img, depth, a_range, beta_range, args.num_reuse)):
            img_no = i * args.num_reuse + j
            resized_img, resized_foggy_img = group_resize_random_crop((img, foggy_img), size=args.image_size)
            resized_img.save(os.path.join(args.out_dir, split, 'clear', f'{img_no:06}.png'))
            resized_foggy_img.save(os.path.join(args.out_dir, split, 'foggy', f'{img_no:06}.png'))


if __name__ == "__main__":
    main()
