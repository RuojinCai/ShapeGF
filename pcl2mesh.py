import os
import yaml
import time
import tqdm
import torch
import trimesh
import torch.nn as nn
import argparse
import importlib
from torch.backends import cudnn
from shutil import copy2
from pprint import pprint
from utils.libmise import MISE
from skimage import measure
import numpy as np
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")


def get_args():
    # command line args
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("config", type=str, help="The configuration file.")

    # distributed training
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Use multi-processing distributed training to "
        "launch N processes per node, which has N GPUs. "
        "This is the fastest way to use PyTorch for "
        "either single node or multi node data parallel "
        "training",
    )

    # Resume:
    parser.add_argument(
        "--pretrained", default=None, type=str, help="Pretrained cehckpoint"
    )

    # Evaluation split
    parser.add_argument(
        "--eval_split", default="val", type=str, help="The split to be evaluated."
    )

    # Parameters
    parser.add_argument(
        "--sigma", default=0.01, type=float,
        help="Sigma for the gradient field.")
    parser.add_argument(
        "--threshold", default=1e-3, type=float,
        help="Threshold for the gradient field.")
    args = parser.parse_args()

    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    # parse config file
    with open(args.config, "r") as f:
        config = yaml.load(f)
    config = dict2namespace(config)

    # #  Create log_name
    cfg_file_name = os.path.splitext(os.path.basename(args.config))[0]
    run_time = time.strftime("%Y-%b-%d-%H-%M-%S")
    # Currently save dir and log_dir are the same
    config.log_name = "logs/%s_val_%s" % (cfg_file_name, run_time)
    config.save_dir = "logs/%s_val_%s" % (cfg_file_name, run_time)
    config.log_dir = "logs/%s_val_%s" % (cfg_file_name, run_time)
    os.makedirs(config.log_dir + "/config")
    copy2(args.config, config.log_dir + "/config")
    return args, config


def eval_points(score_net, x, z, sigma):
    """
    Params:
        [sigma] float
        [z]  (bs, latent_dim)
        [x]  (bs, npoints, 3)
    """
    with torch.no_grad():
        score_net.eval()
        bs = z.size(0)
        z_sigma = torch.cat((
            z, torch.ones((bs, 1)).to(z) * sigma), dim=1)
        x = x.view(1, -1, 3) * 2
        grad = score_net(x, z_sigma)  # (bs, npoints, 3)
        return grad.norm(dim=-1, keepdim=False)


def generate_from_latent(score_net, z, sigma, threshold=1e-5,
                         padding=0.1, upsampling_steps=3, resolution0=32):
    ''' Generates mesh from latent.

    Args:
        z (tensor): latent code z
        c (tensor): latent conditioned code c
        stats_dict (dict): stats dictionary
    '''
    box_size = 1 + padding

    # Shortcut
    mesh_extractor = MISE(
        resolution0, upsampling_steps, threshold)
    points = mesh_extractor.query()
    while points.shape[0] != 0:
        # Query points
        pointsf = torch.FloatTensor(points).to(DEVICE)
        # Normalize to bounding box
        pointsf = pointsf / mesh_extractor.resolution
        pointsf = box_size * (pointsf - 0.5)
        # Evaluate model and update
        values = eval_points(score_net, pointsf, z, sigma).cpu().numpy()
        values = values.astype(np.float64).reshape(-1)
        mesh_extractor.update(points, values)
        points = mesh_extractor.query()
    value_grid = mesh_extractor.to_dense()

    verts, faces, _, _ = measure.marching_cubes(value_grid, threshold)

    # Create mesh
    mesh = trimesh.Trimesh(verts, faces, process=False)
    return mesh


def main_worker(cfg, args):
    # basic setup
    cudnn.benchmark = True
    data_lib = importlib.import_module(cfg.data.type)
    loaders = data_lib.get_data_loaders(cfg.data, args)
    test_loader = loaders["test_loader"]
    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg, args)

    if args.distributed:  # Multiple processes, single GPU per process
        def wrapper(m):
            return nn.DataParallel(m)

        trainer.multi_gpu_wrapper(wrapper)
    trainer.resume(args.pretrained)
    print(cfg.save_dir)
    # TODO
    trainer.score_net.to(DEVICE)
    trainer.encoder.to(DEVICE)
    for data in tqdm.tqdm(test_loader):
        inp = data["tr_points"].to(DEVICE)
        with torch.no_grad():
            trainer.encoder.eval()
            z, _ = trainer.encoder(inp)
            z = z[0, ...].view(1, -1)
            mesh = generate_from_latent(
                trainer.score_net, z, args.sigma, threshold=args.threshold,
                padding=0.1, upsampling_steps=3, resolution0=32)
            mesh.export("mesh_recon.obj")
            break


if __name__ == "__main__":
    # command line args
    args, cfg = get_args()

    print("Arguments:")
    print(args)

    print("Configuration:")
    print(cfg)

    main_worker(cfg, args)
