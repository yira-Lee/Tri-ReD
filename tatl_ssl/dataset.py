from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import Dataset


class MaskedDataset(Dataset):
    "A dataset that returns only ratio% of samples"

    def __init__(self, dataset: Dataset, ratio: float = 1.0):
        super(MaskedDataset, self).__init__()
        assert 0.0 < ratio <= 1.0

        self.ratio = ratio
        self.dataset = dataset

        n_samples = int(self.ratio * len(self.dataset))
        self.range = torch.arange(len(self.dataset))[torch.randperm(len(self.dataset))][
            :n_samples
        ]

    def __len__(self):
        return self.range.size(0)

    def __getitem__(self, idx):
        new_idx = self.range[idx].item()
        return self.dataset[new_idx]


@dataclass()
class DatasetSpec:
    num_classes: int
    size: int
    crop_size: int

    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]



def get_dataset_spec(dataset: str) -> DatasetSpec:
    "Returns the dataset spec for the given path"

    if "CSM" in str(dataset):   
        return DatasetSpec(
            num_classes=45,
            size=256,
            crop_size=224,
            mean=(0.4675, 0.4829, 0.4575),
            std=(0.2256, 0.2229, 0.2537),
        )
    if "BICGSV" in str(dataset): 
        return DatasetSpec(
            num_classes=8,  
            size=256,
            crop_size=224,
            mean=(0.4467, 0.4585, 0.4369),
            std=(0.2143, 0.2143, 0.2426),
        )
    if "BEAUTY" in str(dataset): 
        return DatasetSpec(
            num_classes=4,  
            size=256,
            crop_size=224,
            mean=(0.4373, 0.4459, 0.4317),
            std=(0.2147, 0.2127, 0.2253),
        )

    raise NotImplementedError(dataset)
