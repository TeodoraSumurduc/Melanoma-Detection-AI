import cv2
import numpy as np
import torch
from nn_class import Net


def apply_model(path):
    #50 x 50 pixels
    img_size = 50

    #resize image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))

    #add to an array
    images_arr = np.array(img) 
    images_arr = images_arr / 255

    images_arr = torch.tensor(images_arr)

    net = Net()
    net.load_state_dict(torch.load("saved_model.pth"))
    net.eval()

    images_arr = images_arr.float() 
    net_out = net(images_arr.view(-1, 1, img_size, img_size))[0]

    if net_out[0] >= net_out[1]:
        print()
        print()
        print("prediction Benign")
        print(f"Confidence: {round(float(net_out[0]),3)}")
        print()
        print()
    else:
        print()
        print()
        print("prediction Melanoma")
        print(f"Confidence: {round(float(net_out[1]),3)}")
        print()
        print()


apply_model("demo_pics/melanoma_10117.jpg")