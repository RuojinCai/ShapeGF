import os
import torch
import trimesh
import numpy as np
from torch.utils.data import Dataset
from torch.utils import data
import random


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        assert len(scene_or_mesh.geometry) > 0
        # we lose texture information here
        mesh = trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                  for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


class SingleShape(Dataset):

    def __init__(self, cfg, cfgdata):
        self.mesh = as_mesh(trimesh.load_mesh(cfg.path))
        vert_center = 0.5 * (self.mesh.vertices.max(axis=0) + self.mesh.vertices.min(axis=0))
        # vert_scale = np.linalg.norm(self.mesh.vertices - vert_center, axis=-1).max()
        vert_scale = (self.mesh.vertices.max(axis=0) - self.mesh.vertices.min(axis=0)).max() * 0.5
        norm_vert = (self.mesh.vertices - vert_center) / vert_scale
        self.mesh = trimesh.Trimesh(vertices=norm_vert, faces=self.mesh.faces)

        self.tr_max_sample_points = cfg.tr_max_sample_points
        self.te_max_sample_points = cfg.te_max_sample_points
        self.length = int(cfgdata.length)

        # Default display axis order
        self.display_axis_order = [0, 1, 2]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # TODO!
        tr_out = torch.from_numpy(self.mesh.sample(self.tr_max_sample_points)).float()
        te_out = torch.from_numpy(self.mesh.sample(self.te_max_sample_points)).float()
        m = torch.zeros(1, 3).float()
        s = torch.ones(1, 3).float()
        return {
            'idx': idx,
            'tr_points': tr_out,
            'te_points': te_out,
            'mean': m, 'std': s,
            'display_axis_order': self.display_axis_order
        }


def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)


def get_datasets(cfg, args):
    tr_dataset = SingleShape(cfg, cfg.train)
    te_dataset = SingleShape(cfg, cfg.val)
    return tr_dataset, te_dataset


def get_data_loaders(cfg, args):
    tr_dataset, te_dataset = get_datasets(cfg, args)
    train_loader = data.DataLoader(
        dataset=tr_dataset, batch_size=cfg.train.batch_size,
        shuffle=True, num_workers=cfg.num_workers, drop_last=True,
        worker_init_fn=init_np_seed)
    test_loader = data.DataLoader(
        dataset=te_dataset, batch_size=cfg.val.batch_size,
        shuffle=False, num_workers=cfg.num_workers, drop_last=False,
        worker_init_fn=init_np_seed)

    loaders = {
        "test_loader": test_loader,
        'train_loader': train_loader,
    }
    return loaders
