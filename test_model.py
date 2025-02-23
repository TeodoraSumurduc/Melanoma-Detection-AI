import numpy as np
import torch
from nn_class import Net

#50 x 50 pixels
img_size = 50

net = Net()
net.load_state_dict(torch.load("saved_model.pth"))
net.eval()

testing_data = np.load("melanoma_testing_data.npy", allow_pickle=True)

# for row in testing_data:
#     print(row[0])
#     print(row[1])
#     break
    


# putting all the image arrays into this tensor
test_X = torch.tensor(np.array([item[0] for item in testing_data]), dtype=torch.float32)
test_X = test_X / 255

for row in test_X:
    print(row)
    break

#one-hot vector labesl tensor
test_Y = torch.tensor(np.array([item[1] for item in testing_data]), dtype=torch.float32)


correct = 0
total = 0

with torch.no_grad():
    for i in range(len(test_X)):
        real_class = torch.argmax(test_Y[i])
        net_out = net(test_X[i].view(-1, 1, img_size, img_size))[0]

        predicted_class = torch.argmax(net_out)

        if predicted_class == real_class:
            correct += 1
        total += 1

print(f"Accuracy: {round(correct / total, 3)}")