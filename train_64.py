# Training script for tiny-imagenet.
# Again, this script has a lot of bugs everywhere.
import argparse
import datetime
import json
import os
import shutil
import subprocess

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm

import evaluation
import losses as L
from models.discriminators.snresnet64 import SNResNetConcatDiscriminator
from models.discriminators.snresnet64 import SNResNetProjectionDiscriminator
from models.generators.resnet64 import ResNetGenerator
from models import inception
import utils


# Copied from https://github.com/naoto0804/pytorch-AdaIN/blob/master/sampler.py#L5-L15
def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


# Copied from https://github.com/naoto0804/pytorch-AdaIN/blob/master/sampler.py#L18-L26
class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


def prepare_results_dir(args):
    """Makedir, init tensorboard if required, save args."""
    if args.test:
        import tempfile
        args.results_root = tempfile.mkdtemp()
        args.max_iteration = 10
        args.log_interval = 2
        args.eval_interval = 10
        args.n_fid_images = 100
        args.n_eval_batches = 10
    root = os.path.join(args.results_root, "cGAN" if args.cGAN else "SNGAN",
                        datetime.datetime.now().strftime('%y%m%d_%H%M'))
    os.makedirs(root, exist_ok=True)
    if not args.no_tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(root)
    else:
        writer = None

    train_image_root = os.path.join(root, "preview", "train")
    eval_image_root = os.path.join(root, "preview", "eval")
    os.makedirs(train_image_root, exist_ok=True)
    os.makedirs(eval_image_root, exist_ok=True)

    args.results_root = root
    args.train_image_root = train_image_root
    args.eval_image_root = eval_image_root

    try:
        commit_hash = subprocess.check_output('git rev-parse HEAD'.split(' '))
    except Exception:
        commit_hash = ''
    else:
        commit_hash = commit_hash.decode('utf-8').strip()
    args.commit_hash = commit_hash

    if args.cGAN:
        if args.num_classes > args.n_eval_batches:
            args.n_eval_batches = args.num_classes
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size // 4

    if args.calc_FID:
        args.n_fid_batches = args.n_fid_images // args.batch_size

    with open(os.path.join(root, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print(json.dumps(args.__dict__, indent=2))
    return args, writer


def decay_lr(opt, max_iter, start_iter, initial_lr):
    """Decay learning rate linearly till 0."""
    coeff = -initial_lr / (max_iter - start_iter)
    for pg in opt.param_groups:
        pg['lr'] += coeff


def get_args():
    parser = argparse.ArgumentParser()
    # Dataset configuration
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
    # Common model setting
    parser.add_argument('--use_reflection_pad', '-urp', default=False, action='store_true',
                        help='If you use ReflectionPad2d, set this. default: False')
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
    parser.add_argument('--relativistic_loss', '-relloss', default=False, action='store_true',
                        help='Apply relativistic loss or not. default: False')
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
    # Resume training
    parser.add_argument('--args_path', default=None, help='Checkpoint args json path. default: None')
    parser.add_argument('--gen_ckpt_path', '-gcp', default=None,
                        help='Generator and optimizer checkpoint path. default: None')
    parser.add_argument('--dis_ckpt_path', '-dcp', default=None,
                        help='Discriminator and optimizer checkpoint path. default: None')
    args = parser.parse_args()

    if args.use_reflection_pad:
        args.padding = 'reflection'
    else:
        args.padding = 'zero'
    return args


def sample_from_data(args, device, data_loader):
    """Sample real images and labels from data_loader.

    Args:
        args (argparse object)
        device (torch.device)
        data_loader (DataLoader)

    Returns:
        real, y

    """

    real, y = next(data_loader)
    real, y = real.to(device), y.to(device)
    if not args.cGAN:
        y = None
    return real, y


def sample_from_gen(args, device, num_classes, gen):
    """Sample fake images and labels from generator.

    Args:
        args (argparse object)
        device (torch.device)
        num_classes (int): for pseudo_y
        gen (nn.Module)

    Returns:
        fake, pseudo_y, z

    """

    z = utils.sample_z(
        args.batch_size, args.gen_dim_z, device, args.gen_distribution
    )
    if args.cGAN:
        pseudo_y = utils.sample_pseudo_labels(
            num_classes, args.batch_size, device
        )
    else:
        pseudo_y = None

    fake = gen(z, pseudo_y)
    return fake, pseudo_y, z


def main():
    args = get_args()
    # CUDA setting
    if not torch.cuda.is_available():
        raise ValueError("Should buy GPU!")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True

    def _rescale(img):
        return img * 2.0 - 1.0

    def _noise_adder(img):
        return torch.empty_like(img, dtype=img.dtype).uniform_(0.0, 1/128.0) + img

    # dataset
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_root, 'train'),
        transforms.Compose([
            transforms.ToTensor(), _rescale, _noise_adder,
        ])
    )
    train_loader = iter(data.DataLoader(
        train_dataset, args.batch_size,
        sampler=InfiniteSamplerWrapper(train_dataset),
        num_workers=args.num_workers, pin_memory=True)
    )
    if args.calc_FID:
        eval_dataset = datasets.ImageFolder(
            os.path.join(args.data_root, 'val'),
            transforms.Compose([
                transforms.ToTensor(), _rescale,
            ])
        )
        eval_loader = iter(data.DataLoader(
            eval_dataset, args.batch_size,
            sampler=InfiniteSamplerWrapper(eval_dataset),
            num_workers=args.num_workers, pin_memory=True)
        )
    else:
        eval_loader = None
    num_classes = len(train_dataset.classes)
    print(' prepared datasets...')
    print(' Number of training images: {}'.format(len(train_dataset)))
    # Prepare directories.
    args.num_classes = num_classes
    args, writer = prepare_results_dir(args)
    # initialize models.
    _n_cls = num_classes if args.cGAN else 0
    gen = ResNetGenerator(
        args.gen_num_features, args.gen_dim_z, args.padding,
        args.gen_bottom_width, activation=F.relu, num_classes=_n_cls,
        distribution=args.gen_distribution
    ).to(device)
    if args.dis_arch_concat:
        dis = SNResNetConcatDiscriminator(
            args.dis_num_features, args.padding, _n_cls, F.relu, args.dis_emb
        ).to(device)
    else:
        dis = SNResNetProjectionDiscriminator(
            args.dis_num_features, args.padding, _n_cls, F.relu
        ).to(device)
    if args.calc_FID:
        inception_model = inception.InceptionV3().to(device)
    else:
        inception_model = None

    opt_gen = optim.Adam(gen.parameters(), args.lr, (args.beta1, args.beta2))
    opt_dis = optim.Adam(dis.parameters(), args.lr, (args.beta1, args.beta2))

    # gen_criterion = getattr(L, 'gen_{}'.format(args.loss_type))
    # dis_criterion = getattr(L, 'dis_{}'.format(args.loss_type))
    gen_criterion = L.GenLoss(args.loss_type, args.relativistic_loss)
    dis_criterion = L.DisLoss(args.loss_type, args.relativistic_loss)
    print(' Initialized models...\n')

    if args.args_path is not None:
        print(' Load weights...\n')
        prev_args, gen, opt_gen, dis, opt_dis = utils.resume_from_args(
            args.args_path, args.gen_ckpt_path, args.dis_ckpt_path
        )

    # Training loop
    for n_iter in tqdm.tqdm(range(1, args.max_iteration + 1)):

        if n_iter >= args.lr_decay_start:
            decay_lr(opt_gen, args.max_iteration, args.lr_decay_start, args.lr)
            decay_lr(opt_dis, args.max_iteration, args.lr_decay_start, args.lr)

        # ==================== Beginning of 1 iteration. ====================
        _l_g = .0
        cumulative_loss_dis = .0
        for i in range(args.n_dis):
            if i == 0:
                fake, pseudo_y, _ = sample_from_gen(args, device, num_classes, gen)
                dis_fake = dis(fake, pseudo_y)
                if args.relativistic_loss:
                    real, y = sample_from_data(args, device, train_loader)
                    dis_real = dis(real, y)
                else:
                    dis_real = None

                loss_gen = gen_criterion(dis_fake, dis_real)
                gen.zero_grad()
                loss_gen.backward()
                opt_gen.step()
                _l_g += loss_gen.item()
                if n_iter % 10 == 0 and writer is not None:
                    writer.add_scalar('gen', _l_g, n_iter)

            fake, pseudo_y, _ = sample_from_gen(args, device, num_classes, gen)
            real, y = sample_from_data(args, device, train_loader)

            dis_fake, dis_real = dis(fake, pseudo_y), dis(real, y)
            loss_dis = dis_criterion(dis_fake, dis_real)

            dis.zero_grad()
            loss_dis.backward()
            opt_dis.step()

            cumulative_loss_dis += loss_dis.item()
            if n_iter % 10 == 0 and i == args.n_dis - 1 and writer is not None:
                cumulative_loss_dis /= args.n_dis
                writer.add_scalar('dis', cumulative_loss_dis / args.n_dis, n_iter)
        # ==================== End of 1 iteration. ====================

        if n_iter % args.log_interval == 0:
            tqdm.tqdm.write(
                'iteration: {:07d}/{:07d}, loss gen: {:05f}, loss dis {:05f}'.format(
                    n_iter, args.max_iteration, _l_g, cumulative_loss_dis))
            if not args.no_image:
                writer.add_image(
                    'fake', torchvision.utils.make_grid(
                        fake, nrow=4, normalize=True, scale_each=True))
                writer.add_image(
                    'real', torchvision.utils.make_grid(
                        real, nrow=4, normalize=True, scale_each=True))
            # Save previews
            utils.save_images(
                n_iter, n_iter // args.checkpoint_interval, args.results_root,
                args.train_image_root, fake, real
            )
        if n_iter % args.checkpoint_interval == 0:
            # Save checkpoints!
            utils.save_checkpoints(
                args, n_iter, n_iter // args.checkpoint_interval,
                gen, opt_gen, dis, opt_dis
            )
        if n_iter % args.eval_interval == 0:
            # TODO (crcrpar): implement Ineption score, FID, and Geometry score
            # Once these criterion are prepared, val_loader will be used.
            fid_score = evaluation.evaluate(
                args, n_iter, gen, device, inception_model, eval_loader
            )
            tqdm.tqdm.write(
                '[Eval] iteration: {:07d}/{:07d}, FID: {:07f}'.format(
                    n_iter, args.max_iteration, fid_score))
            if writer is not None:
                writer.add_scalar("FID", fid_score, n_iter)
                # Project embedding weights if exists.
                embedding_layer = getattr(dis, 'l_y', None)
                if embedding_layer is not None:
                    writer.add_embedding(
                        embedding_layer.weight.data,
                        list(range(args.num_classes)),
                        global_step=n_iter
                    )
    if args.test:
        shutil.rmtree(args.results_root)


if __name__ == '__main__':
    main()
