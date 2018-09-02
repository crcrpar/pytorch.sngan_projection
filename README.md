# SNGAN and cGAN with projection discriminator
**This is unofficial PyTorch implementation of sngan_projection.**  
The original is available at https://github.com/pfnet-research/sngan_projection.

## Requirements
- Python 3.6.4
- PyTorch 0.4.1
- torchvision 0.2.1
- NumPy: Used in FID score calculation and data loader
- SciPy: Used in FID score calculation
- tensorflow (optional)
- tensorboardX (optional)
- tqdm: Progressbar and Log

If you want to use **tensorboard** for beautiful training update visualization, please install tensorflow and tensorboardX.  
When using only tensorboard, tensorflow cpu is enough.

## Dataset
- tiny ImageNet[^1].

> Tiny Imagenet has 200 classes. Each class has 500 training images, 50 validation images, and 50 test images.

[^1]: https://tiny-imagenet.herokuapp.com/

## Training configuration
Default parameters are the same as the original Chainer implementation.

- to train cGAN with projection discriminator: run `train_64.py` with `--cGAN` option.
- to train cGAN with concat discriminator: run `train_64.py` with both `--cGAN` and `--dis_arch_concat`.
- to run without `tensorboard`, please add `--no_tensorboard`.
- to calculate FID, add `--calc_FID` (not tested)
- to use make discriminator relativistic, add `--relativistic_loss` or `-relloss` (not tested)

To see all the available arguments, run `python train_64.py --help`.

## TODO
- [ ] check relativistic loss
- [ ] implement super-resolution (cGAN)

# Acknowledgement
1. https://github.com/pfnet-research/sngan_projection
2. https://github.com/mseitzer/pytorch-fid: FID score
3. https://github.com/naoto0804/pytorch-AdaIN: Infinite Sampler of DataLoader
