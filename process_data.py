import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

#one-hot vectors
#[1, 0] = benign
#[0, 1] = melanoma

#made all images the same size 50x50 pixels
img_size = 50

#locations of image, files
benign_training_folder = "melanoma_cancer_dataset/train/benign/"
malignant_training_folder = "melanoma_cancer_dataset/train/malignant/"

benign_testing_folder = "melanoma_cancer_dataset/test/benign/"
malignant_testing_folder = "melanoma_cancer_dataset/test/malignant/"


def load_images(folder, label):
    data = []
    for filename in os.listdir(folder):
        try:
            path = os.path.join(folder, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size))
            data.append([img, label])
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return data

benign_training_data = load_images(benign_training_folder, np.array([1, 0]))
malignant_training_data = load_images(malignant_training_folder, np.array([0, 1]))
benign_testing_data = load_images(benign_testing_folder, np.array([1, 0]))
malignant_testing_data = load_images(malignant_testing_folder, np.array([0, 1]))

#benign training data

#make the training data both the same size
benign_training_data = benign_training_data[0 : len(malignant_training_data)]

#convert the lists to numpy arrays for concatenation
benign_training_data = np.array(benign_training_data, dtype=object)
malignant_training_data = np.array(malignant_training_data, dtype=object)
benign_testing_data = np.array(benign_testing_data, dtype=object)
malignant_testing_data = np.array(malignant_testing_data, dtype=object)

#print len
# print(f"Benign training count: {len(benign_training_data)}")
# print(f"Malignant training count: {len(malignant_training_data)}")
# print(f"Benign testing count: {len(benign_testing_data)}")
# print(f"Malignant testing count: {len(malignant_testing_data)}")

#concatenate and shuffle the data
training_data = np.concatenate((benign_training_data, malignant_training_data))
np.random.shuffle(training_data)
np.save("melanoma_training_data.npy", training_data)

testing_data = np.concatenate((benign_testing_data, malignant_testing_data))
np.random.shuffle(testing_data)
np.save("melanoma_testing_data.npy", testing_data)