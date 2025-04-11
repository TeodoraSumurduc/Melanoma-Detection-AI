import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import math
from nn_class import Net
from torch.utils.data import DataLoader, Dataset



#sentdex neural networks from scratch

class MelanomaDataset(Dataset):
    def __init__(self):
        self.data = np.load("melanoma_training_data.npy", allow_pickle=True)
        self.train_X = torch.tensor(np.array([item[0] for item in self.data]), dtype=torch.float32)
        self.train_X = self.train_X / 255
        self.train_Y = torch.tensor(np.array([item[1] for item in self.data]), dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.train_X[idx]
        label = self.train_Y[idx]
        return img, label


#50 x 50 pixels
img_size = 50

dataset = MelanomaDataset()
dataloader = DataLoader(dataset=dataset, batch_size=100, shuffle=True, num_workers=2)
# first_data = dataset[0]
# print(first_data)
epochs = 2
total_samples = len(dataset)
nr_iterations = math.ceil(total_samples / 100)
print(total_samples, nr_iterations)

net = Net()

optimizer = optim.Adam(net.parameters(), lr=0.001)

loss_functions = nn.MSELoss()

for epoch in range(epochs):
    for i, (images, labels) in enumerate(dataloader):
        print(f"Epoch {epoch + 1}, fraction complete: {i / len(train_X)}")
        batch_X = train_X[i : i + batch_size].view(-1, 1, img_size, img_size)
        batch_Y = train_Y[i : i + batch_size]

        optimizer.zero_grad()
        #reset gradients of model parameters to zero before this pass

        outputs = net(batch_X)
        loss = loss_functions(outputs, batch_Y)

        loss.backward() #backpropagation

        optimizer.step()

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