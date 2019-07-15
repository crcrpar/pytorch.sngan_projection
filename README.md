**UPDATE 2019/07/14: C-GAN has a bug in network definition!**  
Thank you @UdonDa for reporting and pointing out ([issue#28](https://github.com/crcrpar/pytorch.sngan_projection/issues/28))

**UPDATE 2019/05/24: Current implementation of FID score is incorrect!**  
Thank you @youngjung for the report & fix suggestions ([issue#25](https://github.com/crcrpar/pytorch.sngan_projection/issues/25))!

---

The original is available at https://github.com/pfnet-research/sngan_projection.

# SNGAN and cGANs with projection discriminator
_**This is unofficial PyTorch implementation of sngan_projection.**_  
_**This does not reproduce the experiments and results reported in the paper due to the lack of GPUs.**_  
**This repository does some experiments on images of size 64x64.**

Some results are on issues with _results_ label.

## SNGAN
> Spectral Normalization for Generative Adversarial Networks  
> Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida  
> OpenReview: https://openreview.net/forum?id=B1QRgziT-  
> arXiv: https://arxiv.org/abs/1802.05957

## cGANs with projection discriminator
> cGANs with Projection Discriminator  
> Takeru Miyato, Masanori Koyama  
> OpenReview: https://openreview.net/forum?id=ByS1VpgRZ  
> arXiv: https://arxiv.org/abs/1802.05637  

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

### Docker environment
Dockerfiles for pytorch 1.0 environment and tensorboard are added. PyTorch 1.0 Dockerfile requires an nvidia driver that supports CUDA 9.2.
Also, this dockerized environment needs some environment variables:
- `DATA`: Path to dataset
- `RESULTS`: Path to save results
- `PORT`: Port number for jupyter notebook.

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
- [ ] implement super-resolution (cGAN)

# Acknowledgement
1. https://github.com/pfnet-research/sngan_projection
2. https://github.com/mseitzer/pytorch-fid: FID score
3. https://github.com/naoto0804/pytorch-AdaIN: Infinite Sampler of DataLoader
