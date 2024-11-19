import torch
import torch.nn as nn
import torch.nn.functional as F

# Image size (50x50 pixels)
img_size = 50


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        self.fc1 = nn.Linear(128 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)

        x = x.view(-1, 128 * 2 * 2)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        x = F.softmax(x, dim=1)

        return x


net = Net()
test_img = torch.randn(1, 1, img_size, img_size)
output = net(test_img)
print("Network output for test image:", output)
