import os
import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader, random_split


class CHN6_CUGDataset(Dataset):
    def __init__(self, base_dir) -> None:
        super().__init__()
        self.images_dir = os.path.join(base_dir, "images")
        self.labels_dir = os.path.join(base_dir, "gt")

        self.ids = [fname.split("_")[0] for fname in os.listdir(self.images_dir)]
        self.images_fps = [
            os.path.join(self.images_dir, fname + "_sat.jpg") for fname in self.ids
        ]
        self.labels_fps = [
            os.path.join(self.labels_dir, fname + "_mask.png") for fname in self.ids
        ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = torch.from_numpy(
            np.transpose(imread(self.images_fps[idx]), (2, 0, 1))
        ).type(torch.float)
        label = torch.from_numpy(
            np.expand_dims(imread(self.labels_fps[idx], as_gray=True), 0)
        ).type(torch.float)
        label[label > 0] = 1.0

        return image, label


def prepare_dataloaders(base_dir):
    data_dir = os.path.join(base_dir, "data", "CHN6-CUG")

    train_dir = os.path.join(data_dir, "train")
    train_dataset = CHN6_CUGDataset(
        base_dir=train_dir,
    )

    valid_dir = os.path.join(data_dir, "val")
    valid_dataset = CHN6_CUGDataset(
        base_dir=valid_dir,
    )

    # test_dir = os.path.join(data_dir, "test")
    # test_dataset = SOSDataset(
    #    base_dir=test_dir,
    # )
    # Split valid into valid and test (40-60)
    valid_dataset, test_dataset = random_split(valid_dataset, [0.6, 0.4])
    print(f"Training dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(valid_dataset)}")
    print(f"Testing dataset length: {len(test_dataset)}")

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


def save_predictions(batch_idx, images, labels, predictions, directory):
    for idx_image, image in enumerate(images):
        print(f"\t{idx_image} image shape: ", image.shape)
        # Image in CHN6-CUG is color image (three channel)
        image = np.transpose(image, (1, 2, 0))
        image = image * 255
        image = image.astype(np.uint8)
        image_p = Image.fromarray(image)
        image_p = image_p.convert("L")
        image_p.save(os.path.join(directory, f"batch{batch_idx}_image{idx_image}.png"))
    for idx_label, label in enumerate(labels):
        print(f"\t{idx_label} label shape: ", label.shape)
        # label = np.transpose(label, (1, 2, 0))
        label = np.squeeze(label)
        label = label * 255
        label = label.astype(np.uint8)
        label_p = Image.fromarray(label)
        label_p = label_p.convert("L")
        label_p.save(os.path.join(directory, f"batch{batch_idx}_label{idx_label}.png"))
    for idx_prediction, prediction in enumerate(predictions):
        print(f"\t{idx_prediction} prediction shape: ", prediction.shape)
        # prediction = np.transpose(prediction, (1, 2, 0))
        prediction = np.squeeze(prediction)
        prediction_p = Image.fromarray((prediction * 255).astype(np.uint8))
        prediction_p = prediction_p.convert("L")
        prediction_p.save(
            os.path.join(
                directory,
                f"batch{batch_idx}_prediction{idx_prediction}.png",
            )
        )


def save_figures(batch_idx, images, labels, predictions, directory):
    fig, axs = plt.subplots(1, 3, figsize=(12, 8))
    axs[0].imshow(images[0, 0, :, :])
    axs[0].set_title("Imagen")
    axs[1].imshow(predictions[0, 0, :, :])
    axs[1].set_title("Prediction")
    axs[2].imshow(labels[0, 0, :, :])
    axs[2].set_title("Label")
    plt.savefig(os.path.join(directory, f"result_batch{batch_idx}.png"))
    plt.close()


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as F
    from torch.utils.data import DataLoader

    home_dir = os.path.expanduser("~")
    data_dir = os.path.join(home_dir, "data", "CHN6-CUG")
    train_dir = os.path.join(data_dir, "train")
    train_dataset = CHN6_CUGDataset(train_dir)

    image, label = train_dataset[0]
    i = 0
    while np.max(label.numpy()) == 0:
        image, label = train_dataset[i]
        i = i + 1
    print(f"Tensor image shape, type: {image.shape}, {image.type()}")
    print(f"Tensor label shape, type: {label.shape}, {image.type()}")
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
    axs[0].imshow(pil_image, cmap="gray")
    axs[1].imshow(pil_label)
    plt.show()

    # Validation of labels values
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=1)
    print("Checking label values")
    for image, label in train_loader:
        print(np.unique(label.numpy()))
