import random
from collections import Counter

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

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.05,  # Ajustează luminozitatea cu ±10%
            contrast=0.05,  # Ajustează contrastul cu ±10%
            saturation=0.05,  # Ajustează saturația cu ±5%
            hue=0.02  # Ajustează nuanța cu ±2%
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    benign_training_dataset = MelanomaDataset(benign_training_folder, 0, transform=train_transforms)
    malignant_training_dataset = MelanomaDataset(malignant_training_folder, 1, transform=train_transforms)

    benign_testing_dataset = MelanomaDataset(benign_testing_folder, 0, transform=test_transforms)
    malignant_testing_dataset = MelanomaDataset(malignant_testing_folder, 1, transform=test_transforms)

    train_dataset = ConcatDataset([benign_training_dataset, malignant_training_dataset])
    test_dataset = ConcatDataset([benign_testing_dataset, malignant_testing_dataset])

    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=2,
                                  collate_fn=collate_fn_train)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True, num_workers=2, collate_fn=collate_fn_train)

    print(f"Cuda available: {torch.cuda.is_available()}")

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
        # flip_prob = random.random()
        # rotation_deg = random.uniform(0, 25)
        # brightness = np.random.uniform(0.0, 0.5)
        # contrast = np.random.uniform(0.0, 0.25)
        # saturation = np.random.uniform(0.0, 0.25)
        # hue = np.random.uniform(-0.05, 0.05)
        # hue_tuple = (0, hue) if hue >= 0 else (hue, 0)
        #
        # train_transforms = transforms.Compose([
        #     transforms.RandomHorizontalFlip(p=flip_prob),
        #     transforms.RandomVerticalFlip(p=flip_prob),
        #     transforms.RandomRotation(degrees=rotation_deg),
        #     transforms.ColorJitter(
        #         brightness=brightness,
        #         contrast=contrast,
        #         saturation=saturation,
        #         hue=hue_tuple
        #     ),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # ])
        #
        # # === UPDATEAZĂ TRANSFORM-UL ÎN DATASETURI ===
        # benign_training_dataset.transform = train_transforms
        # malignant_training_dataset.transform = train_transforms
        #
        # # ... RESTUL TRAININGULUI (nu se schimbă față de ce ai deja) ...
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
        # print(
        #     f"Epoca {epoch + 1} Augmentări: rotation={rotation_deg:.2f}, brightness={brightness:.2f}, contrast={contrast:.2f}, saturation={saturation:.2f}, hue={hue:.2f}")


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
    print(Counter(y_pred))

    torch.save(model.state_dict(), "melanoma_model.pth")
    print("Model salvat cu succes.")


if __name__ == "__main__":
    main()
