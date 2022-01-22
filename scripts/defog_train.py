"""
Train a defogging model.
"""

import argparse

import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_paired_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    df_model_and_diffusion_defaults,
    df_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    model, diffusion = df_create_model_and_diffusion(
        **args_to_dict(args, df_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_defog_data(
        args.clear_data_dir,
        args.foggy_data_dir,
        args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        save_dir=args.save_dir,
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        train_steps=args.train_steps
    ).run_loop()


def load_defog_data(clear_data_dir, foggy_data_dir, batch_size, image_size, class_cond=False):
    data = load_paired_data(
        tgt_data_dir=clear_data_dir,
        src_data_dir=foggy_data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=class_cond,
    )
    for clear_large_batch, foggy_large_batch, model_kwargs in data:
        model_kwargs["foggy"] = foggy_large_batch
        yield clear_large_batch, model_kwargs


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(df_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
