import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import math
from torchvision.transforms.functional import normalize, to_tensor
from torchvision import transforms
from matplotlib import pyplot as plt

from nn_class import Net
from torch.utils.data import DataLoader, ConcatDataset
from process_data import MelanomaDataset

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

benign_training_dataset = MelanomaDataset(benign_training_folder, np.array([1, 0]), transform=train_transforms)
malignant_training_dataset = MelanomaDataset(malignant_training_folder, np.array([1, 0]), transform=train_transforms)

benign_testing_dataset = MelanomaDataset(benign_testing_folder, np.array([1, 0]))
malignant_testing_dataset = MelanomaDataset(malignant_testing_folder, np.array([0, 1]))

train_dataset = ConcatDataset([benign_training_dataset, malignant_training_dataset])
test_dataset = ConcatDataset([benign_testing_dataset, malignant_testing_dataset])

train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=2, collate_fn=collate_fn_train)
test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True, num_workers=2, collate_fn=collate_fn_train)

print(torch.cuda.is_available())

#sentdex neural networks from scratch
#50 x 50 pixels
img_size = 50

batch_size = 100
epochs = 10
lr = 1e-3

model = Net().cuda()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

train_losses = []
val_losses = []

# Training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for batch in train_dataloader:
        images, labels = batch
        images = images.cuda()
        labels = labels.cuda().long()  # Important pentru CrossEntropyLoss!

        optimizer.zero_grad()
        outputs = model(images)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_dataloader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in test_dataloader:
            images, labels = batch
            images = images.cuda()
            labels = labels.cuda().long()

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(test_dataloader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# dataloader = DataLoader(dataset=dataset, batch_size=100, shuffle=True, num_workers=2)
# first_data = dataset[0]
#
# print(f"First data:{first_data[0]} {first_data[1]}")

# compose = torchvision.transforms.Compose([ToTensor(), MulTransform(255)])
# dataset = MelanomaDataset(transform=compose)
# first_data = dataset[0]
#
# print(f"First data:{first_data[0]} {first_data[1]}")
#
# epochs = 2
# total_samples = len(dataset)
# nr_iterations = math.ceil(total_samples / 100)
# print(total_samples, nr_iterations)
#
# net = Net()
#
# optimizer = optim.Adam(net.parameters(), lr=0.001)
#
# loss_functions = nn.MSELoss()
#
# for epoch in range(epochs):
#     for i, (images, labels) in enumerate(dataloader):
#         print(f"Epoch {epoch + 1}, fraction complete: {i / len(train_X)}")
#         batch_X = train_X[i : i + batch_size].view(-1, 1, img_size, img_size)
#         batch_Y = train_Y[i : i + batch_size]
#
#         optimizer.zero_grad()
#         #reset gradients of model parameters to zero before this pass
#
#         outputs = net(batch_X)
#         loss = loss_functions(outputs, batch_Y)
#
#         loss.backward() #backpropagation
#
#         optimizer.step()

# dataiter = iter(dataloader)
# data = dataiter.next()
# features, labels = data
# print(features, labels)


#
# training_data = np.load("melanoma_training_data.npy", allow_pickle=True)
# # print(len(training_data))
# # for row in training_data:
# #     print(row)
# #     # print(row[0])
# #     # print(row[1])
# #     break
#
# # putting all the image arrays into this tensor
# train_X = torch.tensor(np.array([item[0] for item in training_data]), dtype=torch.float32)
# train_X = train_X / 255
#
# # for row in train_X:
# #     print(row)
# #     break
#
# #one-hot vector labesl tensor
# train_Y = torch.tensor(np.array([item[1] for item in training_data]), dtype=torch.float32)
#
# net = Net()
#
# optimizer = optim.Adam(net.parameters(), lr=0.001)
#
# #mean squared error loss function
# loss_functions = nn.MSELoss()
#
# #how many images to process at once
# batch_size = 100
#
# #TO DO HIGHLY RECOMMENDED: use DataLoader class to load data in batches
# epochs = 2
#
# for epoch in range(epochs):
#     for i in range(0, len(train_X), batch_size):
#         #print(f"Epoch {epoch + 1}, fraction complete: {i / len(train_X)}")
#         batch_X = train_X[i : i + batch_size].view(-1, 1, img_size, img_size)
#         batch_Y = train_Y[i : i + batch_size]
#
#         optimizer.zero_grad()
#         #reset gradients of model parameters to zero before this pass
#
#         outputs = net(batch_X)
#         loss = loss_functions(outputs, batch_Y)
#
#         loss.backward() #backpropagation
#
#         optimizer.step()
#

# torch.save(net.state_dict(), "saved_model.pth")