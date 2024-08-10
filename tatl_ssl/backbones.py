from typing import Optional

import torch
from torch import nn, Tensor
from torchvision import models




class ResNetBackbone(nn.Module):

    out_dim: int

    def __init__(
        self,
        resnet: str,
        num_classes: int = 8,
        with_classifier: bool = False,
        weights_path: Optional[str] = None,
        pretrained: bool = False,
    ):
        super(ResNetBackbone, self).__init__()
        kwargs = {}
        if hasattr(models, "ResNet18_Weights"): 
            kwargs["weights"] = (
                models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None

            )
        else:
            kwargs["pretrained"] = pretrained
        self.model = models.__dict__[resnet](num_classes=num_classes, **kwargs,) 

        if weights_path is not None and pretrained:
            raise Exception("Can't use both pretrained and weights_path")
        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)

        if not with_classifier:
            self.out_dim = self.model.fc.in_features
            self.model.fc = nn.Identity() 
        else:
            self.out_dim = num_classes


    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    