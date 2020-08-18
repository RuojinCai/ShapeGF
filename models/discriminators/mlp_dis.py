import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):

    def __init__(self, cfg, cfgmodel):
        super().__init__()
        self.cfg = cfg
        self.cfgmodel = cfgmodel

        self.inp_dim = cfgmodel.inp_dim
        self.use_bn = getattr(cfgmodel, "use_bn", False)
        self.use_ln = getattr(cfgmodel, "use_ln", False)
        self.use_sigmoid = getattr(cfgmodel, "use_sigmoid", False)
        self.dims = cfgmodel.dims

        curr_dim = self.inp_dim
        self.layers = []
        self.bns = []
        self.lns = []
        for hid in self.dims:
            self.layers.append(nn.Linear(curr_dim, hid))
            if self.use_bn:
                self.bns.append(nn.BatchNorm1d(hid))
            else:
                self.bns.append(None)
            if self.use_ln:
                self.lns.append(nn.LayerNorm(hid))
            else:
                self.lns.append(None)
            curr_dim = hid
        self.layers = nn.ModuleList(self.layers)
        self.bns = nn.ModuleList(self.bns)
        self.lns = nn.ModuleList(self.lns)
        self.out = nn.Linear(curr_dim, 1)

    def forward(self, z=None, bs=None, return_all=False):
        if z is None:
            assert bs is not None
            z = torch.randn(bs, self.inp_dim).cuda()

        y = z
        for layer, bn, ln in zip(self.layers, self.bns, self.lns):
            y = layer(y)
            if self.use_bn:
                y = bn(y)
            if self.use_ln:
                y = ln(y)
            y = F.leaky_relu(y, 0.2)
        y = self.out(y)

        if self.use_sigmoid:
            y = torch.sigmoid(y)
        if return_all:
            return {
                'x': y
            }
        else:
            return y
