import random
import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report
from torchvision import transforms
from nn_class import Net
from torch.utils.data import DataLoader, ConcatDataset
from process_data import MelanomaDataset


def collate_fn_train(examples):
    images = []
    labels = []
    for example in examples:
        image, label = example
        # image = to_tensor(image)
        # image = normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        image = image.unsqueeze(0)
        label = torch.tensor(label).unsqueeze(0)
        images.append(image)
        labels.append(label)

    images_batch = torch.cat(images)
    labels_batch = torch.cat(labels)

    return images_batch, labels_batch

def main():
    benign_training_folder = "melanoma_cancer_dataset/train/benign/"
    malignant_training_folder = "melanoma_cancer_dataset/train/malignant/"

    benign_testing_folder = "melanoma_cancer_dataset/test/benign/"
    malignant_testing_folder = "melanoma_cancer_dataset/test/malignant/"

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    benign_training_dataset = MelanomaDataset(benign_training_folder, 0, transform=None)
    malignant_training_dataset = MelanomaDataset(malignant_training_folder, 1, transform=None)

    benign_testing_dataset = MelanomaDataset(benign_testing_folder, 0, transform=test_transforms)
    malignant_testing_dataset = MelanomaDataset(malignant_testing_folder, 1, transform=test_transforms)

    train_dataset = ConcatDataset([benign_training_dataset, malignant_training_dataset])
    test_dataset = ConcatDataset([benign_testing_dataset, malignant_testing_dataset])

    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=2,
                                  collate_fn=collate_fn_train)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True, num_workers=2, collate_fn=collate_fn_train)

    # print(f"Cuda available: {torch.cuda.is_available()}")

    # img_size = 100
    # batch_size = 100
    epochs = 30
    lr = 1e-3

    model = Net().cuda()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        flip_prob = random.random()
        rotation_deg = random.uniform(0, 25)
        brightness = np.random.uniform(0.0, 0.5)
        contrast = np.random.uniform(0.0, 0.25)
        saturation = np.random.uniform(0.0, 0.25)
        hue = np.random.uniform(-0.05, 0.05)
        hue_tuple = (0, hue) if hue >= 0 else (hue, 0)

        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=flip_prob),
            transforms.RandomVerticalFlip(p=flip_prob),
            transforms.RandomRotation(degrees=rotation_deg),
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue_tuple
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # === UPDATEAZĂ TRANSFORM-UL ÎN DATASETURI ===
        benign_training_dataset.transform = train_transforms
        malignant_training_dataset.transform = train_transforms

        # ... RESTUL TRAININGULUI (nu se schimbă față de ce ai deja) ...
        model.train()
        train_loss = 0.0

        for batch in train_dataloader:
            images, labels = batch
            images = images.cuda()
            labels = labels.cuda().long()

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

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(
            f"Epoca {epoch + 1} Augmentări: rotation={rotation_deg:.2f}, brightness={brightness:.2f}, contrast={contrast:.2f}, saturation={saturation:.2f}, hue={hue:.2f}")


    y_true = []
    y_pred = []

    # model = Net().cuda()

    model.eval()
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.cuda()
            labels = labels.cuda().long()
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            print("Labeluri:", labels[:10])
            print("Predicții:", torch.argmax(outputs, dim=1)[:10])
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print(classification_report(y_true, y_pred, target_names=["benign", "malignant"]))

    torch.save(model.state_dict(), "melanoma_model.pth")
    print("Model salvat cu succes.")

    # plt.plot(train_losses, label="Train Loss")
    # plt.plot(val_losses, label="Val Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()

# benign_training_dataset = MelanomaDataset(benign_training_folder, np.array([1, 0]), transform=train_transforms)
# malignant_training_dataset = MelanomaDataset(malignant_training_folder, np.array([1, 0]), transform=train_transforms)
#
# benign_testing_dataset = MelanomaDataset(benign_testing_folder, np.array([1, 0]))
# malignant_testing_dataset = MelanomaDataset(malignant_testing_folder, np.array([0, 1]))
#
# train_dataset = ConcatDataset([benign_training_dataset, malignant_training_dataset])
# test_dataset = ConcatDataset([benign_testing_dataset, malignant_testing_dataset])
#
# train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=2, collate_fn=collate_fn_train)
# test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True, num_workers=2, collate_fn=collate_fn_train)
#
# print(torch.cuda.is_available())
#
# #sentdex neural networks from scratch
# #50 x 50 pixels
# img_size = 50
#
# batch_size = 100
# epochs = 10
# lr = 1e-3
#
# model = Net().cuda()
#
# loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=lr)
#
# train_losses = []
# val_losses = []
#
# # Training loop
# for epoch in range(epochs):
#     model.train()
#     train_loss = 0.0
#
#     for batch in train_dataloader:
#         images, labels = batch
#         images = images.cuda()
#         labels = labels.cuda().long()  # Important pentru CrossEntropyLoss!
#
#         optimizer.zero_grad()
#         outputs = model(images)
#
#         loss = loss_fn(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         train_loss += loss.item()
#
#     train_loss /= len(train_dataloader)
#
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for batch in test_dataloader:
#             images, labels = batch
#             images = images.cuda()
#             labels = labels.cuda().long()
#
#             outputs = model(images)
#             loss = loss_fn(outputs, labels)
#             val_loss += loss.item()
#
#     val_loss /= len(test_dataloader)
#
#     train_losses.append(train_loss)
#     val_losses.append(val_loss)
#
#     print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
#
# plt.plot(train_losses, label="Train Loss")
# plt.plot(val_losses, label="Val Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()

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
