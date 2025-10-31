import torch
import torch.nn as nn
import torch.nn.functional as F

class NeRFMLP(nn.Module):
    def __init__(self, D=8, W=256, input_ch=63, input_ch_dir=27, skips=[4]):
        """
        D: number of layers
        W: number of hidden units per layer
        input_ch: input dimension of encoded position
        input_ch_dir: input dimension of encoded view direction
        skips: list of layers to apply skip connection
        """
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_dir = input_ch_dir
        self.skips = skips

        # Build position-only layers
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] +
            [nn.Linear(W + input_ch if i in skips else W, W) for i in range(D-1)]
        )

        # Output layers
        self.sigma_linear = nn.Linear(W, 1)      # density
        self.feature_linear = nn.Linear(W, W)    # features for color
        self.dir_linear = nn.Linear(W + input_ch_dir, W//2)
        self.rgb_linear = nn.Linear(W//2, 3)     # RGB output

    def forward(self, x, d):
        """
        x: [N, input_ch] encoded 3D position
        d: [N, input_ch_dir] encoded view direction
        returns: sigma [N,1], rgb [N,3]
        """
        h = x
        for i, l in enumerate(self.pts_linears):
            h = F.relu(l(h))
            if i in self.skips:
                h = torch.cat([x, h], dim=-1)

        sigma = F.softplus(self.sigma_linear(h))  # density >= 0

        features = self.feature_linear(h)
        h_dir = torch.cat([features, d], dim=-1)
        h_dir = F.relu(self.dir_linear(h_dir))
        rgb = torch.sigmoid(self.rgb_linear(h_dir))  # color in [0,1]

        return sigma, rgb
