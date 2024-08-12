import os
import torch
import numpy as np
import pandas as pd
from torchvision.io import ImageReadMode, read_image
from torch.utils.data import Dataset


class CimatDataset(Dataset):
    def __init__(
        self,
        base_dir,
        dataset,
        trainset,
        features_channels,
        features_extension,
        labels_extension,
    ):
        super().__init__()
        # Initialization
        self.data_dir = os.path.join(
            base_dir,
            "data",
            "projects",
            "consorcio-ia",
            "data",
            f"oil-spills_{dataset}",
            "augmented_dataset",
        )
        self.features_dir = os.path.join(self.data_dir, "features")
        self.labels_dir = os.path.join(self.data_dir, "labels")
        self.csv_datadir = os.path.join(self.data_dir, "learningCSV", "trainingFiles")
        csv_dataset = pd.read_csv(
            os.path.join(self.csv_datadir, f"train{trainset}.csv")
        )
        self.keys = csv_dataset["key"]
        self.features_channels = features_channels
        self.features_extension = features_extension
        self.labels_extension = labels_extension

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # x = np.zeros(self.dims, dtype=np.float32)
        key = self.keys[idx]
        # Load features
        features = []
        for j, feature in enumerate(self.features_channels):
            filename = os.path.join(
                self.features_dir, feature, key + self.features_extension
            )
            z = read_image(filename, ImageReadMode.GRAY)
            features.append(z)
        x = torch.stack(features)
        # z = imread(filename, as_gray=True).astype(np.float32)

        # if z.shape[0] == self.dims[0] and z.shape[1] == self.dims[1]:
        #    x[..., j] = z
        # Load label
        filename = os.path.join(self.labels_dir, key + self.labels_extension)
        y = read_image(filename)
        # y = np.zeros((x.shape[0], x.shape[1], 1))
        # z = imread(filename, as_gray=True).astype(np.float32) / 255.0

        # if z.shape[0] == self.dims[0] and z.shape[1] == self.dims[1]:
        #    y[..., 0] = z

        # Make C,H,W
        # x = torch.tensor(x).permute(2, 0, 1)
        # y = torch.tensor(y).permute(2, 0, 1)
        return x, y
