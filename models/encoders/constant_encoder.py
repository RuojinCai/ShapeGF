import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, cfgmodel):
        super(Encoder, self).__init__()
        self.zdim = cfgmodel.zdim
        self.out = nn.Parameter(torch.randn(1, self.zdim), requires_grad=True)

    def forward(self, x):
        bs = x.size(0)
        m = v = self.out.expand(bs, -1)
        return m, v
