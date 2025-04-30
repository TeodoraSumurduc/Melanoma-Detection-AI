import os
from collections import Counter

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
from sympy.physics.control.control_plots import matplotlib
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.transforms.functional import normalize, to_tensor
from torchvision import transforms
from PIL import Image
from nn_class import Net

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

