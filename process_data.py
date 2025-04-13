import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import math
import torchvision.transforms.functional as TF
from nn_class import Net
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import normalize, to_tensor
from torchvision import transforms

# one-hot vectors
# [1, 0] = benign
# [0, 1] = melanoma

# made all images the same size 50x50 pixels
# img_size = 50

# locations of image, files
benign_training_folder = "melanoma_cancer_dataset/train/benign/"
malignant_training_folder = "melanoma_cancer_dataset/train/malignant/"

benign_testing_folder = "melanoma_cancer_dataset/test/benign/"
malignant_testing_folder = "melanoma_cancer_dataset/test/malignant/"

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.1,  # Ajustează luminozitatea cu ±10%
        contrast=0.1,  # Ajustează contrastul cu ±10%
        saturation=0.05,  # Ajustează saturația cu ±5%
        hue=0.02  # Ajustează nuanța cu ±2%
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def collate_fn_train(examples):
    images = []
    labels = []
    for example in examples:
        image, label = example
        image = to_tensor(image)
        image = normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        image = image.unsqueeze(0)
        label = torch.tensor(label).unsqueeze(0)
        images.append(image)
        labels.append(label)

    images_batch = torch.cat(images)
    labels_batch = torch.cat(labels)

    return images_batch, labels_batch


class MelanomaImageDataset(Dataset):
    def __init__(self, folder, label, img_size=50):
        self.data = []
        self.label = label
        for filename in os.listdir(folder):
            try:
                path = os.path.join(folder, filename)
                img = cv2.imread(path)
                img = cv2.resize(img, (img_size, img_size))
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


# def load_images(folder, label):
#     data = []
#     for filename in os.listdir(folder):
#         try:
#             path = os.path.join(folder, filename)
#             img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#             img = cv2.resize(img, (img_size, img_size))
#             data.append([img, label])
#         except Exception as e:
#             print(f"Error loading {filename}: {e}")
#     return data
benign_training_data = MelanomaDataset(benign_training_folder, np.array([1, 0]))
benign_training_dataloader = DataLoader(benign_training_data, batch_size=100, shuffle=True, num_workers=2, collate_fn=collate_fn_train)

print(benign_training_dataloader.dataset[0])
# benign_training_data = load_images(benign_training_folder, np.array([1, 0]))
# malignant_training_data = load_images(malignant_training_folder, np.array([0, 1]))
# benign_testing_data = load_images(benign_testing_folder, np.array([1, 0]))
# malignant_testing_data = load_images(malignant_testing_folder, np.array([0, 1]))
#
# #make the training data both the same size
# benign_training_data = benign_training_data[0 : len(malignant_training_data)]
#
# #convert the lists to numpy arrays for concatenation
# benign_training_data = np.array(benign_training_data, dtype=object)
# malignant_training_data = np.array(malignant_training_data, dtype=object)
# benign_testing_data = np.array(benign_testing_data, dtype=object)
# malignant_testing_data = np.array(malignant_testing_data, dtype=object)
#
# #print len
# print(f"Benign training count: {len(benign_training_data)}")
# print(f"Malignant training count: {len(malignant_training_data)}")
# # print(f"Benign testing count: {len(benign_testing_data)}")
# # print(f"Malignant testing count: {len(malignant_testing_data)}")
#
# #concatenate and shuffle the data
# training_data = np.concatenate((benign_training_data, malignant_training_data))
# np.random.shuffle(training_data)
# np.save("melanoma_training_data.npy", training_data)
#
# testing_data = np.concatenate((benign_testing_data, malignant_testing_data))
# np.random.shuffle(testing_data)
# np.save("melanoma_testing_data.npy", testing_data)
#
# # print(training_data[0][0])
# # print(testing_data[0][0])
