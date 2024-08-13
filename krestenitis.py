import os
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


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as F

    home_dir = os.path.expanduser("~")
    data_dir = os.path.join(home_dir, "data", "oil-spill-dataset")
    train_dir = os.path.join(data_dir, "train")

    train_dataset = KrestenitisDataset(train_dir)
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
