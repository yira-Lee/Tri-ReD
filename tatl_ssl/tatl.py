from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .base_method import BaseMethod
from .backbones import ResNetBackbone


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class TATL(BaseMethod):
    """
    """

    def __init__(
        self,
        backbone: ResNetBackbone,
        lambd: float,
        batch_size: int,
        h_dim: Optional[int] = None,
        num_proj_layers: int = 3,
        ):
        super(TATL, self).__init__()

        self.backbone = backbone

        z_dim = self.backbone.out_dim
        if h_dim is None:
            h_dim = 4 * z_dim

        sizes = [z_dim] + [h_dim for _ in range(num_proj_layers)]
        layers = []
        for i in range(len(sizes) - 2):
            layers += [
                nn.Linear(sizes[i], sizes[i + 1], bias=False),
                nn.BatchNorm1d(sizes[i + 1]),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projection_head = nn.Sequential(*layers)

        self.lambd = lambd
        self.batch_size = batch_size
        self.bn = nn.BatchNorm1d(h_dim, affine=False)
       
    def forward(self, x1: Tensor, x2: Tensor, x3: Tensor) -> Tensor:
        z1 = self.projection_head(self.backbone(x1))
        z2 = self.projection_head(self.backbone(x2))
        z3 = self.projection_head(self.backbone(x3))
        # Compute the cross correlation matrix C
        c1 = self.bn(z1).T @ self.bn(z2)  
        c1.div_(self.batch_size)

        on_diag1 = torch.diagonal(c1).add_(-1).pow_(2).sum()
        off_diag1 = off_diagonal(c1).pow_(2).sum()
        loss1 = on_diag1 + self.lambd * off_diag1

        c2 = self.bn(z1).T @ self.bn(z3)  
        c2.div_(self.batch_size)

        on_diag2 = torch.diagonal(c2).add_(-1).pow_(2).sum()
        off_diag2 = off_diagonal(c2).pow_(2).sum()
        loss2 = on_diag2 + self.lambd * off_diag2

        c3 = self.bn(z2).T @ self.bn(z3)  
        c3.div_(self.batch_size)

        on_diag3 = torch.diagonal(c3).add_(-1).pow_(2).sum()
        off_diag3 = off_diagonal(c3).pow_(2).sum()
        loss3 = on_diag3 + self.lambd * off_diag3
        loss = (loss1 + loss2 + loss3) / 3  


        if torch.isnan(loss).item():
            raise Exception("loss is NaN")

        return loss