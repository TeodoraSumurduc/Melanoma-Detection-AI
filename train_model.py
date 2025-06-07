from collections import Counter
import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from torchvision import transforms
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights

from nn_class import Net
from torch.utils.data import DataLoader, ConcatDataset
from process_data import MelanomaDataset


def collate_fn_melanoma(examples):
    images = []
    labels = []
    for example in examples:
        image, label = example
        image = image.unsqueeze(0)
        label = torch.tensor(label).unsqueeze(0)
        images.append(image)
        labels.append(label)

    images_batch = torch.cat(images)
    labels_batch = torch.cat(labels)

    return images_batch, labels_batch


class config:
    def __init__(self):
        self.img_size = 64
        self.batch_size = 100
        self.num_workers = 2
        self.num_classes = 2
        self.epochs = 30
        self.learning_rate = 1e-3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "melanoma_model.pth"
        # self.model = Net().to(self.device)


        # Load pre-trained ResNet50 with weights
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replace the final fully connected layer to match our number of classes
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, self.num_classes)

        # Only the final layer will be trainable
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

        # Move model to device
        self.resnet = self.resnet.to(self.device)


        self.benign_training_folder = "melanoma_cancer_dataset/train/benign/"
        self.malignant_training_folder = "melanoma_cancer_dataset/train/malignant/"
        self.benign_testing_folder = "melanoma_cancer_dataset/test/benign/"
        self.malignant_testing_folder = "melanoma_cancer_dataset/test/malignant/"
        self.benign_val_folder = "melanoma_cancer_dataset/val/benign/"
        self.malignant_val_folder = "melanoma_cancer_dataset/val/malignant/"

    def data_loader(self, dataset, collate_fn_melanoma=None):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          collate_fn=collate_fn_melanoma)

    def get_train_transform(self):
        return transforms.Compose([
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

    def get_test_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def get_dataset(self):
        benign_training_dataset = MelanomaDataset(self.benign_training_folder, 0, transform=self.get_train_transform())
        malignant_training_dataset = MelanomaDataset(self.malignant_training_folder, 1,
                                                     transform=self.get_train_transform())

        benign_testing_dataset = MelanomaDataset(self.benign_testing_folder, 0, transform=self.get_test_transform())
        malignant_testing_dataset = MelanomaDataset(self.malignant_testing_folder, 1,
                                                    transform=self.get_test_transform())

        benign_val_dataset = MelanomaDataset(self.benign_val_folder, 0, transform=self.get_test_transform())
        malignant_val_dataset = MelanomaDataset(self.malignant_val_folder, 1,
                                                transform=self.get_test_transform())

        train_dataset = ConcatDataset([benign_training_dataset, malignant_training_dataset])
        test_dataset = ConcatDataset([benign_testing_dataset, malignant_testing_dataset])
        val_dataset = ConcatDataset([benign_val_dataset, malignant_val_dataset])

        return train_dataset, test_dataset, val_dataset

    def train_model(self, train_dataset, test_dataset, val_dataset):
        train_dataloader = self.data_loader(train_dataset)
        test_dataloader = self.data_loader(test_dataset)
        val_dataloader = self.data_loader(val_dataset)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.resnet.fc.parameters(), lr=self.learning_rate)

        train_losses = []
        val_losses = []

        for epoch in range(self.epochs):

            self.resnet.train()
            train_loss = 0.0

            for batch in train_dataloader:
                images, labels = batch

                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.resnet(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_dataloader)

            self.resnet.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    images, labels = batch

                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.resnet(images)
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(val_dataloader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Epoch {epoch + 1}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        self.save_model()
        self.graphs(train_losses, val_losses, test_dataloader)
        print("Training completed.")

    def save_model(self):
        torch.save(self.resnet.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def graphs(self, train_losses, val_losses, test_dataloader):
        y_true = []
        y_pred = []

        y_scores = []

        self.resnet.eval()
        with torch.no_grad():
            for images, labels in test_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.resnet(images)
                preds = torch.argmax(outputs, dim=1)
                print("Labeluri:", labels[:10])
                print("Predicții:", torch.argmax(outputs, dim=1)[:10])
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

                probs = torch.softmax(outputs, dim=1)[:, 1]
                y_scores.extend(probs.cpu().numpy())

        print(classification_report(y_true, y_pred, target_names=["benign", "malignant"]))
        print(Counter(y_pred))

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color='blue', label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig("roc_curve.png")
        plt.close()

        # matricea de confuzie
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        plt.close()

        # train vs val loss
        plt.figure(figsize=(6, 4))
        plt.plot(train_losses, label="Train Loss", color='blue')
        plt.plot(val_losses, label="Val Loss", color='orange')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Evoluția pierderii (Loss)")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig("train_val_loss.png")
        plt.close()


def main():
    cfg = config()
    train_dataset, test_dataset = cfg.get_dataset()
    cfg.train_model(train_dataset, test_dataset)


if __name__ == "__main__":
    main()
