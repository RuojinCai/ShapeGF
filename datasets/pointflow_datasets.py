import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils import data
import random
import tqdm


# taken from https://github.com/optas/latent_3d_points/blob/
# 8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


class Uniform15KPC(Dataset):
    def __init__(self, root_dir, subdirs, tr_sample_size=10000,
                 te_sample_size=10000, split='train', scale=1.,
                 normalize_per_shape=False, random_subsample=False,
                 normalize_std_per_axis=False, recenter_per_shape=False,
                 all_points_mean=None, all_points_std=None,
                 input_dim=3):
        self.root_dir = root_dir
        self.split = split
        self.in_tr_sample_size = tr_sample_size
        self.in_te_sample_size = te_sample_size
        self.subdirs = subdirs
        self.scale = scale
        self.random_subsample = random_subsample
        self.input_dim = input_dim

        self.all_cate_mids = []
        self.cate_idx_lst = []
        self.all_points = []
        for cate_idx, subd in tqdm.tqdm(enumerate(self.subdirs), total=len(self.subdirs)):
            # NOTE: [subd] here is synset id
            sub_path = os.path.join(root_dir, subd, self.split)
            if not os.path.isdir(sub_path):
                print("Directory missing : %s" % sub_path)
                continue

            all_mids = []
            for x in os.listdir(sub_path):
                if not x.endswith('.npy'):
                    continue
                all_mids.append(os.path.join(self.split, x[:-len('.npy')]))

            # NOTE: [mid] contains the split: i.e. "train/<mid>"
            # or "val/<mid>" or "test/<mid>"
            for mid in all_mids:
                # obj_fname = os.path.join(sub_path, x)
                obj_fname = os.path.join(root_dir, subd, mid + ".npy")
                try:
                    point_cloud = np.load(obj_fname)  # (15k, 3)
                except:  # nofa: E722
                    continue

                assert point_cloud.shape[0] == 15000
                self.all_points.append(point_cloud[np.newaxis, ...])
                self.cate_idx_lst.append(cate_idx)
                self.all_cate_mids.append((subd, mid))

        # Shuffle the index deterministically (based on the number of examples)
        self.shuffle_idx = list(range(len(self.all_points)))
        random.Random(38383).shuffle(self.shuffle_idx)
        self.cate_idx_lst = [self.cate_idx_lst[i] for i in self.shuffle_idx]
        self.all_points = [self.all_points[i] for i in self.shuffle_idx]
        self.all_cate_mids = [self.all_cate_mids[i] for i in self.shuffle_idx]

        # Normalization
        self.all_points = np.concatenate(self.all_points)  # (N, 15000, 3)
        self.normalize_per_shape = normalize_per_shape
        self.normalize_std_per_axis = normalize_std_per_axis
        self.recenter_per_shape = recenter_per_shape
        if all_points_mean is not None and all_points_std is not None and not self.recenter_per_shape:
            # using loaded dataset stats
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std
        elif self.recenter_per_shape:  # per shape center
            # TODO: bounding box scale at the large dim and center
            B, N = self.all_points.shape[:2]
            self.all_points_mean = (
                (np.amax(self.all_points, axis=1)).reshape(B, 1, input_dim) +
                (np.amin(self.all_points, axis=1)).reshape(B, 1, input_dim)
            ) / 2
            self.all_points_std = np.amax((
                (np.amax(self.all_points, axis=1)).reshape(B, 1, input_dim) -
                (np.amin(self.all_points, axis=1)).reshape(B, 1, input_dim)
            ), axis=-1).reshape(B, 1, 1) / 2
        elif self.normalize_per_shape:  # per shape normalization
            B, N = self.all_points.shape[:2]
            self.all_points_mean = self.all_points.mean(
                axis=1).reshape(B, 1, input_dim)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(
                    B, N, -1).std(axis=1).reshape(B, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(
                    B, -1).std(axis=1).reshape(B, 1, 1)
        else:  # normalize across the dataset
            self.all_points_mean = self.all_points.reshape(
                -1, input_dim).mean(axis=0).reshape(1, 1, input_dim)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(
                    -1, input_dim).std(axis=0).reshape(1, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(
                    -1).std(axis=0).reshape(1, 1, 1)

        self.all_points = (self.all_points - self.all_points_mean) / \
            self.all_points_std
        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

        self.tr_sample_size = min(10000, tr_sample_size)
        self.te_sample_size = min(5000, te_sample_size)
        print("Total number of data:%d" % len(self.train_points))
        print("Min number of points: (train)%d (test)%d"
              % (self.tr_sample_size, self.te_sample_size))
        assert self.scale == 1, "Scale (!= 1) is deprecated"

        # Default display axis order
        self.display_axis_order = [0, 1, 2]

    def get_pc_stats(self, idx):
        if self.recenter_per_shape:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s

        if self.normalize_per_shape:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s

        return self.all_points_mean.reshape(1, -1), \
            self.all_points_std.reshape(1, -1)

    def renormalize(self, mean, std):
        self.all_points = self.all_points * self.all_points_std + \
                          self.all_points_mean
        self.all_points_mean = mean
        self.all_points_std = std
        self.all_points = (self.all_points - self.all_points_mean) / \
            self.all_points_std
        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

    def __len__(self):
        return len(self.train_points)

    def __getitem__(self, idx):
        tr_out = self.train_points[idx]
        if self.random_subsample:
            tr_idxs = np.random.choice(tr_out.shape[0], self.tr_sample_size)
        else:
            tr_idxs = np.arange(self.tr_sample_size)
        tr_out = torch.from_numpy(tr_out[tr_idxs, :]).float()

        te_out = self.test_points[idx]
        if self.random_subsample:
            te_idxs = np.random.choice(te_out.shape[0], self.te_sample_size)
        else:
            te_idxs = np.arange(self.te_sample_size)
        te_out = torch.from_numpy(te_out[te_idxs, :]).float()

        m, s = self.get_pc_stats(idx)
        cate_idx = self.cate_idx_lst[idx]
        sid, mid = self.all_cate_mids[idx]

        return {
            'idx': idx,
            'tr_points': tr_out,
            'te_points': te_out,
            'mean': m, 'std': s, 'cate_idx': cate_idx,
            'sid': sid, 'mid': mid,
            'display_axis_order': self.display_axis_order
        }


class ShapeNet15kPointClouds(Uniform15KPC):
    def __init__(self, root_dir="data/ShapeNetCore.v2.PC15k",
                 categories=['airplane'], tr_sample_size=10000,
                 te_sample_size=2048,
                 split='train', scale=1., normalize_per_shape=False,
                 normalize_std_per_axis=False, recenter_per_shape=False,
                 random_subsample=False,
                 all_points_mean=None, all_points_std=None):
        self.root_dir = root_dir
        self.split = split
        assert self.split in ['train', 'test', 'val']
        self.tr_sample_size = tr_sample_size
        self.te_sample_size = te_sample_size
        self.cates = categories
        if 'all' in categories:
            self.synset_ids = list(cate_to_synsetid.values())
        else:
            self.synset_ids = [cate_to_synsetid[c] for c in self.cates]

        # assert 'v2' in root_dir, "Only supporting v2 right now."
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        super(ShapeNet15kPointClouds, self).__init__(
            root_dir, self.synset_ids,
            tr_sample_size=tr_sample_size,
            te_sample_size=te_sample_size,
            split=split, scale=scale,
            normalize_per_shape=normalize_per_shape,
            normalize_std_per_axis=normalize_std_per_axis,
            recenter_per_shape=recenter_per_shape,
            random_subsample=random_subsample,
            all_points_mean=all_points_mean, all_points_std=all_points_std,
            input_dim=3)


def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)


def get_datasets(cfg, args):
    tr_dataset = ShapeNet15kPointClouds(
        categories=cfg.cates, split='train',
        tr_sample_size=cfg.tr_max_sample_points,
        te_sample_size=cfg.te_max_sample_points,
        scale=cfg.dataset_scale, root_dir=cfg.data_dir,
        normalize_per_shape=cfg.normalize_per_shape,
        normalize_std_per_axis=cfg.normalize_std_per_axis,
        recenter_per_shape=cfg.recenter_per_shape,
        random_subsample=True)

    eval_split = getattr(args, "eval_split", "val")
    te_dataset = ShapeNet15kPointClouds(
        categories=cfg.cates, split=eval_split,
        tr_sample_size=cfg.tr_max_sample_points,
        te_sample_size=cfg.te_max_sample_points,
        scale=cfg.dataset_scale, root_dir=cfg.data_dir,
        normalize_per_shape=cfg.normalize_per_shape,
        normalize_std_per_axis=cfg.normalize_std_per_axis,
        recenter_per_shape=cfg.recenter_per_shape,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
    )
    return tr_dataset, te_dataset


def get_data_loaders(cfg, args):
    tr_dataset, te_dataset = get_datasets(cfg, args)
    train_loader = data.DataLoader(
        dataset=tr_dataset, batch_size=cfg.batch_size,
        shuffle=True, num_workers=cfg.num_workers, drop_last=True,
        worker_init_fn=init_np_seed, pin_memory=True
    )
    test_loader = data.DataLoader(
        dataset=te_dataset, batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers, drop_last=False,
        worker_init_fn=init_np_seed
    )

    loaders = {
        "test_loader": test_loader,
        'train_loader': train_loader,
    }
    return loaders
