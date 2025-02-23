import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from nn_class import Net


#sentdex neural networks from scratch

#50 x 50 pixels
img_size = 50

training_data = np.load("melanoma_training_data.npy", allow_pickle=True)

for row in training_data:
    print(row[0])
    print(row[1])
    break

# putting all the image arrays into this tensor
train_X = torch.tensor(np.array([item[0] for item in training_data]), dtype=torch.float32)
train_X = train_X / 255

# for row in train_X:
#     print(row)
#     break

#one-hot vector labesl tensor
train_Y = torch.tensor(np.array([item[1] for item in training_data]), dtype=torch.float32)

net = Net()

optimizer = optim.Adam(net.parameters(), lr=0.001)

#mean squared error loss function
loss_functions = nn.MSELoss()

#how many images to process at once
batch_size = 100

#TO DO HIGHLY RECOMMENDED: use DataLoader class to load data in batches
epochs = 2

for epoch in range(epochs):
    for i in range(0, len(train_X), batch_size):
        print(f"Epoch {epoch + 1}, fraction complete: {i / len(train_X)}")
        batch_X = train_X[i : i + batch_size].view(-1, 1, img_size, img_size)
        batch_Y = train_Y[i : i + batch_size]

        optimizer.zero_grad()
        #reset gradients of model parameters to zero before this pass

        outputs = net(batch_X)
        loss = loss_functions(outputs, batch_Y)

        loss.backward() #backpropagation

        optimizer.step()

torch.save(net.state_dict(), "saved_model.pth")