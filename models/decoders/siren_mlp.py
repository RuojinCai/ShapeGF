import torch
import torch.nn as nn
import numpy as np
import pdb


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    torch.tensor(-np.sqrt((6 / self.in_features) / self.omega_0)),
                    torch.tensor(np.sqrt((6 / self.in_features) / self.omega_0)),
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Decoder(nn.Module):
    def __init__(self, _, cfg):
        super().__init__()
        self.cfg = cfg
        self.z_dim = z_dim = cfg.z_dim
        self.dim = dim = cfg.dim
        self.out_dim = out_dim = cfg.out_dim  # * 2048
        self.hidden_size = hidden_size = cfg.hidden_size
        self.n_blocks = n_blocks = cfg.n_blocks

        # TODO: use configs
        out_features = self.out_dim
        hidden_features = self.hidden_size
        hidden_layers = self.n_blocks
        c_dim = z_dim + dim + 1
        batch_size = 50
        in_features = c_dim  # 270336
        outermost_linear = True
        first_omega_0 = 30
        hidden_omega_0 = 30.0

        self.net = []
        self.net.append(
            SineLayer(
                in_features, hidden_features, is_first=True, omega_0=first_omega_0
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    torch.tensor((-np.sqrt(6 / hidden_features) / hidden_omega_0)),
                    torch.tensor((np.sqrt(6 / hidden_features) / hidden_omega_0)),
                )

            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, x, c):
        # coords = (
        #     coords.clone().detach().requires_grad_(True)
        # )  # allows to take derivative w.r.t. input
        p = x.transpose(1, 2)  # (bs, dim, n_points)
        batch_size, D, num_points = p.size()

        # ToDo: Not sure this viewing is right.
        c_expand = c.unsqueeze(2).expand(-1, -1, num_points)
        c_xyz = torch.cat([p, c_expand], dim=1)
        # c_xyz = c_xyz.view(batch_size, -1)
        c_xyz = c_xyz.view(batch_size, num_points, -1)
        output = self.net(c_xyz)
        # output = output.view(batch_size, -1, 3)
        return output  # , coords

    def forward_with_activations(self, coords, retain_grad=False):
        """Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!"""
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations["input"] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations[
                    "_".join((str(layer.__class__), "%d" % activation_count))
                ] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations["_".join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations
