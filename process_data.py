import os
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from sympy.physics.control.control_plots import matplotlib
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import normalize, to_tensor
from torchvision import transforms
from PIL import Image
from nn_class import Net



# locations of image, files
# benign_training_folder = "melanoma_cancer_dataset/train/benign/"
# malignant_training_folder = "melanoma_cancer_dataset/train/malignant/"
#
# benign_testing_folder = "melanoma_cancer_dataset/test/benign/"
# malignant_testing_folder = "melanoma_cancer_dataset/test/malignant/"
#
# train_transforms = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomRotation(degrees=15),
#     transforms.ColorJitter(
#         brightness=0.1,  # Ajustează luminozitatea cu ±10%
#         contrast=0.1,  # Ajustează contrastul cu ±10%
#         saturation=0.05,  # Ajustează saturația cu ±5%
#         hue=0.02  # Ajustează nuanța cu ±2%
#     ),
#     transforms.ToTensor(),
#     #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])

# def collate_fn_train(examples):
#     images = []
#     labels = []
#     for example in examples:
#         image, label = example
#         # image = to_tensor(image)
#         image = normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         image = image.unsqueeze(0)
#         label = torch.tensor(label).unsqueeze(0)
#         images.append(image)
#         labels.append(label)
#
#     images_batch = torch.cat(images)
#     labels_batch = torch.cat(labels)
#
#     return images_batch, labels_batch
#

class MelanomaImageDataset(Dataset):
    def __init__(self, folder, label, img_size=64):
        self.data = []
        self.label = label
        for filename in os.listdir(folder):
            try:
                path = os.path.join(folder, filename)
                img = Image.open(path).convert("RGB")
                img = img.resize((img_size, img_size))
                self.data.append([img, label])
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MelanomaDataset(MelanomaImageDataset):
    def __init__(self, folder, label, transform=None):
        super().__init__(folder, label)
        self.transform = transform

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)

        if self.transform:
            image = self.transform(image)

        return image, label

# folder_path = "melanoma_cancer_dataset/test/benign/"
# files = os.listdir(folder_path)
#
# # Numărăm doar imaginile (fișierele .jpg, .png etc.)
# image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
# num_images = sum(1 for file in files if os.path.splitext(file)[1].lower() in image_extensions)
#
# print(f"Număr de imagini benign: {num_images}")
#
# folder_path = "melanoma_cancer_dataset/test/malignant/"
# files = os.listdir(folder_path)
#
# # Numărăm doar imaginile (fișierele .jpg, .png etc.)
# image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
# num_images = sum(1 for file in files if os.path.splitext(file)[1].lower() in image_extensions)
#
# print(f"Număr de imagini melanoma: {num_images}")
#
# benign_training_data = MelanomaDataset(benign_training_folder, np.array([1, 0]), transform=train_transforms)
# benign_training_dataloader = DataLoader(benign_training_data, batch_size=100, shuffle=True, num_workers=2, collate_fn=collate_fn_train)
#
# import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
#
# img, label = benign_training_dataloader.dataset[0]
# img_np = img.numpy().transpose(1, 2, 0)  # [C,H,W] -> [H,W,C]
# plt.imshow(img_np)
# plt.title(f"Label: {label}")
# plt.axis('off')
# plt.show()
