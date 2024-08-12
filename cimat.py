import os
import torch
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset


class CimatDataset(Dataset):
    def __init__(
        self,
        keys,
        features_path,
        features_ext,
        features_channels,
        labels_path,
        labels_ext,
        dimensions,
    ):
        super().__init__()
        self.keys = keys
        self.features_path = features_path
        self.labels_path = labels_path
        self.features_channels = features_channels
        self.features_ext = features_ext
        self.labels_ext = labels_ext
        self.dims = dimensions

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        x = np.zeros(self.dims, dtype=np.float32)
        key = self.keys[idx]
        # Load features
        for j, feature in enumerate(self.features_channels):
            filename = os.path.join(
                self.features_path, feature, key + self.features_ext
            )
            z = imread(filename, as_gray=True).astype(np.float32)

            if z.shape[0] == self.dims[0] and z.shape[1] == self.dims[1]:
                x[..., j] = z
        # Load label
        filename = os.path.join(self.labels_path, key + self.labels_ext)
        y = np.zeros((x.shape[0], x.shape[1], 1))
        z = imread(filename, as_gray=True).astype(np.float32) / 255.0

        if z.shape[0] == self.dims[0] and z.shape[1] == self.dims[1]:
            y[..., 0] = z

        # Make C,H,W
        x = torch.tensor(x).permute(2, 0, 1)
        y = torch.tensor(y).permute(2, 0, 1)
        return x, y
