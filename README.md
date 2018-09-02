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
- to calculate FID, add `--calc_FID`

### Available options
```python
parser.add_argument('--cGAN', default=False, action='store_true',
                    help='to train cGAN, set this ``True``. default: False')
parser.add_argument('--data_root', type=str, default='tiny-imagenet-200',
                    help='path to dataset root directory. default: tiny-imagenet-200')
parser.add_argument('--batch_size', '-B', type=int, default=64,
                    help='mini-batch size of training data. default: 64')
parser.add_argument('--eval_batch_size', '-eB', default=None,
                    help='mini-batch size of evaluation data. default: None')
parser.add_argument('--num_workers', type=int, default=8,
                    help='Number of workers for training data loader. default: 8')
# Generator configuration
parser.add_argument('--gen_num_features', '-gnf', type=int, default=64,
                    help='Number of features of generator (a.k.a. nplanes or ngf). default: 64')
parser.add_argument('--gen_dim_z', '-gdz', type=int, default=128,
                    help='Dimension of generator input noise. default: 128')
parser.add_argument('--gen_bottom_width', '-gbw', type=int, default=4,
                    help='Initial size of hidden variable of generator. default: 4')
parser.add_argument('--gen_distribution', '-gd', type=str, default='normal',
                    help='Input noise distribution: normal (default) or uniform.')
# Discriminator (Critic) configuration
parser.add_argument('--dis_arch_concat', '-concat', default=False, action='store_true',
                    help='If use concat discriminator, set this true. default: False')
parser.add_argument('--dis_emb', type=int, default=128,
                    help='Parameter for concat discriminator. default: 128')
parser.add_argument('--dis_num_features', '-dnf', type=int, default=64,
                    help='Number of features of discriminator (a.k.a nplanes or ndf). default: 64')
# Optimizer settings
parser.add_argument('--lr', type=float, default=0.0002,
                    help='Initial learning rate of Adam. default: 0.0002')
parser.add_argument('--beta1', type=float, default=0.0,
                    help='beta1 (betas[0]) value of Adam. default: 0.0')
parser.add_argument('--beta2', type=float, default=0.9,
                    help='beta2 (betas[1]) value of Adam. default: 0.9')
parser.add_argument('--lr_decay_start', '-lds', type=int, default=50000,
                    help='Start point of learning rate decay. default: 50000')
# Training setting
parser.add_argument('--seed', type=int, default=46,
                    help='Random seed. default: 46 (derived from Nogizaka46)')
parser.add_argument('--max_iteration', '-N', type=int, default=100000,
                    help='Max iteration number of training. default: 100000')
parser.add_argument('--n_dis', type=int, default=5,
                    help='Number of discriminator updater per generator updater. default: 5')
parser.add_argument('--num_classes', '-nc', type=int, default=0,
                    help='Number of classes in training data. No need to set. default: 0')
parser.add_argument('--loss_type', type=str, default='hinge',
                    help='loss function name. hinge (default) or dcgan.')
parser.add_argument('--calc_FID', default=False, action='store_true',
                    help='If calculate FID score, set this ``True``. default: False')
# Log and Save interval configuration
parser.add_argument('--results_root', type=str, default='results',
                    help='Path to results directory. default: results')
parser.add_argument('--no_tensorboard', action='store_true', default=False,
                    help='If you dislike tensorboard, set this ``False``. default: True')
parser.add_argument('--no_image', action='store_true', default=False,
                    help='If you dislike saving images on tensorboard, set this ``True``. default: False')
parser.add_argument('--checkpoint_interval', '-ci', type=int, default=1000,
                    help='Interval of saving checkpoints (model and optimizer). default: 1000')
parser.add_argument('--log_interval', '-li', type=int, default=100,
                    help='Interval of showing losses. default: 100')
parser.add_argument('--eval_interval', '-ei', type=int, default=1000,
                    help='Interval for evaluation (save images and FID calculation). default: 1000')
parser.add_argument('--n_eval_batches', '-neb', type=int, default=100,
                    help='Number of mini-batches used in evaluation. default: 100')
parser.add_argument('--n_fid_images', '-nfi', type=int, default=5000,
                    help='Number of images to calculate FID. default: 5000')
parser.add_argument('--test', default=False, action='store_true',
                    help='If test this python program, set this ``True``. default: False')
```

## TODO
- [ ] check SNGAN
- [ ] check relativistic loss
- [ ] implement super-resolution (cGAN)

# Acknowledgement
1. https://github.com/pfnet-research/sngan_projection
2. https://github.com/mseitzer/pytorch-fid: FID score
3. https://github.com/naoto0804/pytorch-AdaIN: Infinite Sampler of DataLoader
