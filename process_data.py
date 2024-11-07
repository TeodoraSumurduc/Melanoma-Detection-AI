import os
import cv2
import matplotlib.pypllot as plt
import numpy as n

#made all images the same size 50x50 pixels
img_size = 50




#locations of image, files
benign_training_folder = "melanoma_cancer_dataset/train/benign/"
malignant_training_folder = "melanoma_cancer_dataset/train/malignant/"

benign_testing_folder = "melanoma_cancer_dataset/test/benign/"
malignant_testing_folder = "melanoma_cancer_dataset/test/malignant/"

for filename in os.listdir(benign_training_folder):
    img = cv2.imread(os.path.join(benign_training_folder, filename))
    img = cv2.resize(img, (img_size, img_size))
    cv2.imwrite(os.path.join(benign_training_folder, filename), img)