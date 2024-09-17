import os
import torch
import numpy as np
import pandas as pd
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader


class CimatDataset(Dataset):
    def __init__(
        self,
        base_dir,
        dataset,
        trainset,
        features_channels,
        features_extension,
        labels_extension,
        learning_dir,
        mode,
    ):
        super().__init__()
        # Initialization
        self.data_dir = os.path.join(
            base_dir,
            "data",
            "projects",
            "consorcio-ia",
            "data",
            f"oil_spills_{dataset}",
            "augmented_dataset",
        )
        self.features_dir = os.path.join(self.data_dir, "features")
        self.labels_dir = os.path.join(self.data_dir, "labels")
        self.csv_datadir = os.path.join(self.data_dir, "learningCSV", learning_dir)
        csv_dataset = pd.read_csv(
            os.path.join(self.csv_datadir, f"{mode}{trainset}.csv"), nrows=1024
        )
        self.keys = csv_dataset["key"]
        channels = {"o": "ORIGIN", "v": "VAR", "w": "WIND"}
        self.features_channels = [channels[feat] for feat in features_channels]
        self.features_extension = features_extension
        self.labels_extension = labels_extension

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # x = np.zeros(self.dims, dtype=np.float32)
        key = self.keys[idx]
        # Load features
        # features = []
        # for j, feature in enumerate(self.features_channels):
        #    filename = os.path.join(
        #    self.features_dir, feature, key + self.features_extension
        # )
        #    z = torch.from_numpy(imread(filename, as_gray=True))
        #    features.append(z)
        x = torch.stack(
            [
                torch.from_numpy(
                    imread(
                        os.path.join(
                            self.features_dir, feature, key + self.features_extension
                        ),
                        as_gray=True,
                    )
                )
                for feature in self.features_channels
            ]
        )
        # z = imread(filename, as_gray=True).astype(np.float32)

        # if z.shape[0] == self.dims[0] and z.shape[1] == self.dims[1]:
        #    x[..., j] = z
        # Load label
        # filename = os.path.join(self.labels_dir, key + self.labels_extension)
        y = torch.from_numpy(
            np.expand_dims(
                imread(
                    os.path.join(self.labels_dir, key + self.labels_extension),
                    as_gray=True,
                ).astype(np.float32)
                / 255.0,
                0,
            )
        )
        # y = np.zeros((x.shape[0], x.shape[1], 1))
        # z = imread(filename, as_gray=True).astype(np.float32) / 255.0

        # if z.shape[0] == self.dims[0] and z.shape[1] == self.dims[1]:
        #    y[..., 0] = z

        # Make C,H,W
        # x = torch.tensor(x).permute(2, 0, 1)
        # y = torch.tensor(y).permute(2, 0, 1)
        return x, y


def prepare_dataloaders(base_dir, dataset, trainset, feat_channels):
    train_dataset = CimatDataset(
        base_dir=base_dir,
        dataset=dataset,
        trainset=trainset,
        features_channels=feat_channels,
        features_extension=".tiff",
        labels_extension=".pgm",
        learning_dir="trainingFiles",
        mode="train",
    )

    valid_dataset = CimatDataset(
        base_dir=base_dir,
        dataset=dataset,
        trainset=trainset,
        features_channels=feat_channels,
        features_extension=".tiff",
        labels_extension=".pgm",
        learning_dir="crossFiles",
        mode="cross",
    )

    test_dataset = CimatDataset(
        base_dir=base_dir,
        dataset=dataset,
        trainset=trainset,
        features_channels=feat_channels,
        features_extension=".tiff",
        labels_extension=".pgm",
        learning_dir="testingFiles",
        mode="test",
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=16, pin_memory=True, shuffle=True, num_workers=12
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=4, pin_memory=True, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=4, pin_memory=True, shuffle=False, num_workers=4
    )
    return train_dataloader, valid_dataloader, test_dataloader


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as F

    home_dir = os.path.expanduser("~")
    feat_channels = ["ORIGIN", "ORIGIN", "VAR"]

    train_dataset = CimatDataset(
        base_dir=home_dir,
        dataset="17",
        trainset="01",
        features_channels=feat_channels,
        features_extension=".tiff",
        labels_extension=".pgm",
        learning_dir="trainingFiles",
        mode="train",
    )
    image, label = train_dataset[0]
    print(f"Tensor image shape: {image.shape}")
    print(f"Tensor label shape: {label.shape}")
    np_image = image.numpy()
    np_label = label.numpy()
    print(f"Numpy image shape: {np_image.shape}")
    print(f"Numpy label shape: {np_label.shape}")

    print(
        f"Image: max: {np.max(np_image)}, min: {np.min(np_image)}, values: {np.unique(np_image)}"
    )
    print(
        f"Label: max: {np.max(np_label)}, min: {np.min(np_label)}, values: {np.unique(np_label)}"
    )

    fig, axs = plt.subplots(1, 2)
    pil_image = np.array(F.to_pil_image(image))
    pil_label = np.array(F.to_pil_image(label))
    axs[0].imshow(pil_image)
    axs[1].imshow(pil_label)
    plt.show()
