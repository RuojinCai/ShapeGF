from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # nofa: #401


# Visualization
def visualize_point_clouds_3d(pcl_lst, title_lst=None):
    # pts, gtr, inp):
    pcl_lst = [pcl.cpu().detach().numpy() for pcl in pcl_lst]
    if title_lst is None:
        title_lst = [""] * len(pcl_lst)

    fig = plt.figure(figsize=(3 * len(pcl_lst), 3))
    for idx, (pts, title) in enumerate(zip(pcl_lst, title_lst)):
        ax1 = fig.add_subplot(1, len(pcl_lst), 1 + idx, projection='3d')
        ax1.set_title(title)
        ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5)
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        ax1.set_zlim(-1, 1)
    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    res = np.array(fig.canvas.renderer._renderer)
    res = np.transpose(res, (2, 0, 1))

    plt.close()
    return res


# Visualization
def visualize_point_clouds_3d_scan(pcl_lst, title_lst=None):
    # pts, gtr, inp):
    pcl_lst = [pcl.cpu().detach().numpy() for pcl in pcl_lst]
    if title_lst is None:
        title_lst = [""] * len(pcl_lst)

    fig = plt.figure(figsize=(3 * len(pcl_lst), 3))
    for idx, (pts, title) in enumerate(zip(pcl_lst, title_lst)):
        ax1 = fig.add_subplot(1, len(pcl_lst), 1 + idx, projection='3d')
        ax1.set_title(title)
        ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 1], s=0.5)
        # print(min(pts[:, 0]), max(pts[:, 0]), min(pts[:, 1]), max(pts[:, 1]), min(pts[:, 2]), max(pts[:, 2]))
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        ax1.set_zlim(-1, 1)
    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    res = np.array(fig.canvas.renderer._renderer)
    res = np.transpose(res, (2, 0, 1))

    plt.close()
    return res


# Visualization moving field and likelihood
def get_grid(x, k=10):
    # TODO: set the range of field
    # x = self.get_prior(
    #        1, num_points, self.cfg.models.scorenet.dim).cuda()
    # ind_x = np.arange(x[:,:,0].min(),x[:,:,0].max(),3/k)
    # ind_y = np.arange(x[:,:,1].min(),x[:,:,0].max(),3/k)
    ind_x = np.arange(-1.5, 1.5, 3 / k)
    ind_y = np.arange(-1.5, 1.5, 3 / k)
    X, Y = np.meshgrid(ind_x, ind_y)
    X = torch.tensor(X).view(k * k).to(x)
    Y = torch.tensor(Y).view(k * k).to(x)

    point_grid = torch.ones((1, k * k, 2), dtype=torch.double).to(x)
    point_grid[0, :, 1] = point_grid[0, :, 1] * X
    point_grid[0, :, 0] = point_grid[0, :, 0] * Y
    point_grid = point_grid.float()
    point_grid = point_grid.expand(x.size(0), -1, -1)
    return point_grid


def visualize_point_clouds_2d_overlay(pcl_lst, title_lst=None, path=None):
    # pts, gtr, inp):
    pcl_lst = [pcl.cpu().detach().numpy() for pcl in pcl_lst]
    if title_lst is None:
        title_lst = [""] * len(pcl_lst)

    fig = plt.figure(figsize=(3, 3))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title(title_lst[0])
    for idx, pts in enumerate(pcl_lst):
        ax1.scatter(pts[:, 0], pts[:, 1], s=5)
    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    res = np.array(fig.canvas.renderer._renderer)
    res = np.transpose(res, (2, 0, 1))
    if path:
        plt.savefig(path)
    plt.close()
    return res


def visualize_field(gtr, grid, field, k, label='field'):
    grid_ = np.reshape(grid.cpu().detach().numpy(), (1, k * k, 2))
    if field.size(-1) == 2:
        field = np.reshape(field.cpu().detach().numpy(), (1, k * k, 2))
        fig = plt.figure(figsize=(int(k / 100) * 2, int(k / 100)))
        plt.title(label)
        cs = fig.add_subplot(1, 2, 1)
        field_val = np.sqrt(np.reshape((field ** 2).sum(axis=-1), (1, k * k, 1)))
    else:
        fig = plt.figure(figsize=(int(k / 100), int(k / 100)))
        plt.title(label)
        cs = fig.add_subplot(1, 1, 1)
        field_val = np.reshape(field.cpu().detach().numpy(), (1, k * k, 1))

    gt = gtr.cpu().detach().numpy()

    for i in range(np.shape(field_val)[0]):
        # cs = fig.add_subplot(1, 2, 1)
        X = np.reshape(grid_[i, :, 0], (k, k))
        Y = np.reshape(grid_[i, :, 1], (k, k))
        cs.contourf(X, Y, np.reshape(field_val[i, :], (k, k)), 20,
                    vmin=min(field_val[i, :]), vmax=max(field_val[i, :]),
                    cmap=cm.coolwarm)
        print(min(field_val[i, :]), max(field_val[i, :]))
        m = plt.cm.ScalarMappable(cmap=cm.coolwarm)
        m.set_array(np.reshape(field_val[i, :], (k, k)))
        m.set_clim(min(field_val[i, :]), max(field_val[i, :]))

    if np.shape(field)[-1] == 2:
        for i in range(np.shape(field_val)[0]):
            ax = fig.add_subplot(1, 2, 2)
            scale = 20
            indx = np.array([np.arange(0, k, scale) + t * k for t in range(0, k, scale)])
            X = np.reshape(grid_[i, indx, 0], int(k * k / scale / scale))
            Y = np.reshape(grid_[i, indx, 1], int(k * k / scale / scale))
            u = np.reshape(field[i, indx, 0], int(k * k / scale / scale))
            v = np.reshape(field[i, indx, 1], int(k * k / scale / scale))

            color = np.sqrt(v ** 2 + u ** 2)
            field_norm = field / field_val
            u = np.reshape(field_norm[i, indx, 0], int(k * k / scale / scale))
            v = np.reshape(field_norm[i, indx, 1], int(k * k / scale / scale))
            ax.quiver(X, Y, u, v, color, alpha=0.8, cmap=cm.coolwarm)
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.set_aspect('equal')
            ax.scatter(gt[:, 0], gt[:, 1], s=1, color='r')

    fig.canvas.draw()
    # grab the pixel buffer and dump it into a numpy array
    res = np.array(fig.canvas.renderer._renderer)
    res = np.transpose(res, (2, 0, 1))

    plt.close()
    return res


def visualize_procedure(sigmas, fig_list, gtr, num_vis, cfg, name="Rec_gt"):
    all_imgs = []
    sigmas = np.append([0], sigmas)
    for idx in range(num_vis):
        img = visualize_point_clouds_3d(
            [fig_list[i][idx] for i in
             range(0, len(fig_list), 1)] + [gtr[idx]],
            [(name + " step" +
              str(i * int(getattr(cfg.inference, "num_steps", 5))) +
              " sigma%.3f" % sigmas[i])
             for i in range(0, len(fig_list), 1)] + ["gt shape"])
        all_imgs.append(img)
    img = np.concatenate(all_imgs, axis=1)
    return img

