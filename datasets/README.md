# Downloading datasets

This directory includes instructions and a script for synthesizing data from the NYU Depth Dataset V2 for use in this codebase.

## NYU Depth Dataset V2

To download the NYU Depth Dataset V2 dataset, download that relevant mat file [here](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat). You can pass this to our [nyu_depth_dataset_v2.py](nyu_depth_dataset_v2.py) script like so:

```
python nyu_depth_dataset_v2.py /path/to/nyu_depth_v2_labeled.mat nyu_train_output_dir
```

This creates the following directories:

```
nyu_train_output_dir/train/foggy
nyu_train_output_dir/train/clear
nyu_train_output_dir/test/foggy
nyu_train_output_dir/test/clear
```

The train directories can be passed to the training scripts via the `--foggy_data_dir` and `--clear_data_dir` arguments.
