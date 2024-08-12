import os
import numpy as np
from torchvision.io import ImageReadMode, read_image
from torch.utils.data import Dataset


class KrestenitisDataset(Dataset):
    def __init__(self, base_dir) -> None:
        super().__init__()
        self.images_dir = os.path.join(base_dir, "images")
        self.labels_dir = os.path.join(base_dir, "labels_1D")

        self.ids = os.listdir(self.images_dir)
        self.images_fps = [
            os.path.join(self.images_dir, image_id.split(".")[0] + ".jpg")
            for image_id in self.ids
        ]
        self.labels_fps = [
            os.path.join(self.labels_dir, image_id.split(".")[0] + ".png")
            for image_id in self.ids
        ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Read Image
        image = read_image(self.images_fps[idx], ImageReadMode.GRAY)
        label = read_image(self.labels_fps[idx])

        return image, label
