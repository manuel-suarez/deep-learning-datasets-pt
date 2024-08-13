import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from krestenitis import KrestenitisDataset
from cimat import CimatDataset

home_dir = os.path.expanduser("~")
# data_dir = os.path.join(home_dir, "data", "oil-spill-dataset_256")
# train_dir = os.path.join(data_dir, "train")

# dataset = KrestenitisDataset(train_dir)
# data_dir = os.path.join(
#    home_dir,
#    "data",
#    "projects",
#    "consorcio-ia",
#    "data",
#    "oil_spills_17",
#    "augmented_dataset",
# )
# feat_dir = os.path.join(data_dir, "features")
# labl_dir = os.path.join(data_dir, "labels")
# train_dir = os.path.join(data_dir, "learningCSV", "trainingFiles")
feat_channels = ["ORIGIN", "ORIGIN", "VAR"]

# Load CSV key files
# train_set = pd.read_csv(os.path.join(train_dir, f"train01.csv"))
# print(f"Training CSV file length: {len(train_set)}")

# Load generators
# train_keys = train_set["key"]
train_dataset = CimatDataset(
    base_dir=home_dir,
    dataset="17",
    trainset="01",
    features_channels=feat_channels,
    features_extension=".tiff",
    labels_extension=".pgm",
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
