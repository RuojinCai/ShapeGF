import torch
import torch.nn as nn
import numpy as np
import pdb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResnetBlockConv1d(nn.Module):
    """ 1D-Convolutional ResNet block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(
        self,
        c_dim,
        size_in,
        size_h=None,
        size_out=None,
        norm_method="batch_norm",
        legacy=False,
    ):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        if norm_method == "batch_norm":
            norm = nn.BatchNorm1d
        elif norm_method == "sync_batch_norm":
            norm = nn.SyncBatchNorm
        else:
            raise Exception("Invalid norm method: %s" % norm_method)

        self.bn_0 = norm(size_in)
        self.bn_1 = norm(size_h)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.fc_c = nn.Conv1d(c_dim, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)

        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        net = self.fc_0(self.actvn(self.bn_0(x)))
        dx = self.fc_1(self.actvn(self.bn_1(net)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        out = x_s + dx + self.fc_c(c)

        return out


class Decoder(nn.Module):
    """ Decoder conditioned by adding.

    Example configuration:
        z_dim: 128
        hidden_size: 256
        n_blocks: 5
        out_dim: 3  # we are outputting the gradient
        sigma_condition: True
        xyz_condition: True
    """

    def __init__(self, _, cfg):
        super().__init__()
        self.cfg = cfg
        self.z_dim = z_dim = cfg.z_dim
        self.dim = dim = cfg.dim
        self.out_dim = out_dim = cfg.out_dim
        self.hidden_size = hidden_size = cfg.hidden_size
        self.n_blocks = n_blocks = cfg.n_blocks
        self.m_dim = m_dim = getattr(cfg, "m_dim", 256)
        self.sigma = sigma = getattr(cfg, "sigma", 12)

        self.bvals = torch.randn(1, m_dim, dim) * sigma  # (1, 256, 3)
        self.bvals.requires_grad = False
        self.avals = torch.ones(self.bvals[:, :, 0].shape)
        self.avals.requires_grad = False

        # Input = Conditional = zdim (shape) + dim (xyz) + 1 (sigma)
        c_dim = z_dim + 2 * self.m_dim + 1 + dim
        self.conv_p = nn.Conv1d(c_dim, hidden_size, 1)
        self.blocks = nn.ModuleList(
            [ResnetBlockConv1d(c_dim, hidden_size) for _ in range(n_blocks)]
        )
        self.bn_out = nn.BatchNorm1d(hidden_size)
        self.conv_out = nn.Conv1d(hidden_size, out_dim, 1)
        self.actvn_out = nn.ReLU()

    # This should have the same signature as the sig condition one
    def forward(self, x, cee):
        """
        :param x: (bs, npoints, self.dim) Input coordinate (xyz)
        :param c: (bs, self.zdim + 1) Shape latent code + sigma
        :return: (bs, npoints, self.dim) Gradient (self.dim dimension)
        """
        pvar = x.transpose(1, 2)  # (bs, dim, n_points)
        batch_size, Dim, num_points = pvar.size()
        encoding = self.encode(x)
        pvar = torch.cat((pvar, encoding), dim=1)  # (bs, 2m+dim, n_pts)
        c_expand = cee.unsqueeze(2).expand(-1, -1, num_points)  # (bs, zdim, n_pts)
        c_xyz = torch.cat([pvar, c_expand], dim=1)  # (bs, 2m+dim+zdim, n_pts)
        net = self.conv_p(c_xyz)
        for block in self.blocks:
            net = block(net, c_xyz)
        out = self.conv_out(self.actvn_out(self.bn_out(net))).transpose(1, 2)
        return out

    def encode(self, input):
        # implementing guass only for now
        # Bmm: batch matrix-matrix-multiply]
        bvals = self.bvals.expand(input.size(0), -1, -1).to(DEVICE)  # (bs, m, dim)
        avals = self.avals.to(DEVICE)  # (m, 1)
        input = input.permute(0, 2, 1).to
        vals1 = self.avals.T * torch.sin(
            2 * np.pi * torch.bmm(bvals, input)
        )  # (bs, m, npoints)
        vals2 = self.avals.T * torch.cos(
            2 * np.pi * torch.bmm(bvals, input)
        )  # (bs, m, npoints)
        encoded_input = torch.cat((vals1, vals2), dim=1)
        encoded_input /= torch.norm(self.avals)  # (bs, 2m, npoints)
        del vals1, vals2
        return encoded_input
