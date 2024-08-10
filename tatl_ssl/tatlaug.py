import functools
import random
from typing import Tuple, Union

import PIL
import numpy as np
import torch
from torch import Tensor
import torchvision.transforms.functional as F
import torchvision.transforms as T


def odd(x: Union[int, float]) -> int:
    x = int(x)
    if x % 2 == 0:
        return x + 1
    return x




leaf_color_range = [(35, 80, 25), (120, 220, 75)]

color_range = (random.randint(leaf_color_range[0][0], leaf_color_range[1][0]),
               random.randint(leaf_color_range[0][1], leaf_color_range[1][1]),
               random.randint(leaf_color_range[0][2], leaf_color_range[1][2]))
def NDA(img, v=0.4):  
    assert 0.0 <= v <= 0.5
    if v <= 0.:
        return img

    v = v * img.size[0]
    return NDAAbs(img, v, color_range=color_range)


def NDAAbs(img, v, color_range=(0, 255)):  
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color_range)
    return img

class TATLAugmentations:
    def __init__(self,
                 size: int,
                 s: float = 0.1,
                 mean=[0.4675, 0.4829, 0.4575],
                 std=[0.2256, 0.2229, 0.2537]):
        def adjust_gamma_img(img):
            gamma_value = 2
            return F.adjust_gamma(img, gamma_value)
        nda_fn = functools.partial(NDA, v=0.4) 
        augs = []
        augs.append(T.RandomResizedCrop(size=size, scale=(.2, 1.)))

        
        self.augment_f = T.Compose(augs + [
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.augment_s = T.Compose(augs + [
            T.RandomApply([T.ColorJitter(0.4*s, 0.4*s, 0.4*s, 0.1*s)], p=0.8),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.augment_t = T.Compose(augs + [
            T.Lambda(nda_fn),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.augment_f(x), self.augment_s(x), self.augment_t(x)