import os

import numpy as np
import torchvision

import metrics.fid
import utils


def evaluate(args, current_iter, gen, device,
             inception_model=None, eval_iter=None):
    """Evaluate model using 100 mini-batches."""
    calc_fid = (inception_model is not None) and (eval_iter is not None)
    num_batches = args.n_eval_batches
    gen.eval()
    fake_list, real_list = [], []
    conditional = args.cGAN
    for i in range(1, num_batches + 1):
        if conditional:
            class_id = i % args.num_classes
        else:
            class_id = None
        fake = utils.generate_images(
            gen, device, args.batch_size, args.gen_dim_z,
            args.gen_distribution, class_id=class_id
        )
        if calc_fid and i <= args.n_fid_batches:
            fake_list.append((fake.cpu().numpy() + 1.0) / 2.0)
            real_list.append((next(eval_iter)[0].cpu().numpy() + 1.0) / 2.0)
        # Save generated images.
        root = args.eval_image_root
        if conditional:
            root = os.path.join(root, "class_id_{:04d}".format(i))
        if not os.path.isdir(root):
            os.makedirs(root)
        fn = "image_iter_{:07d}_batch_{:04d}.png".format(current_iter, i)
        torchvision.utils.save_image(
            fake, os.path.join(root, fn), nrow=4, normalize=True, scale_each=True
        )
    # Calculate FID scores
    if calc_fid:
        fake_images = np.concatenate(fake_list)
        real_images = np.concatenate(real_list)
        mu_fake, sigma_fake = metrics.fid.calculate_activation_statistics(
            fake_images, inception_model, args.batch_size, device=device
        )
        mu_real, sigma_real = metrics.fid.calculate_activation_statistics(
            real_images, inception_model, args.batch_size, device=device
        )
        fid_score = metrics.fid.calculate_frechet_distance(
            mu_fake, sigma_fake, mu_real, sigma_real
        )
    else:
        fid_score = -1000
    gen.train()
    return fid_score
