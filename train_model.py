from collections import Counter
import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from torchvision import transforms

from nn_class import Net
from torch.utils.data import DataLoader, ConcatDataset
from process_data import MelanomaDataset


def collate_fn_train(examples):
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
            brightness=0.05,  # luminozitatea cu ±10%
            contrast=0.05,  # contrastul cu ±10%
            saturation=0.05,  # saturația cu ±5%
            hue=0.02  #  nuanța cu ±2%
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

    epochs = 30
    lr = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)

    # model = Net().cuda()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0

        for batch in train_dataloader:
            images, labels = batch
            # images = images.cuda()
            # labels = labels.cuda().long()

            images = images.to(device)
            labels = labels.to(device)

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
                # images = images.cuda()
                # labels = labels.cuda().long()

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(test_dataloader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")


    y_true = []
    y_pred = []

    y_scores = []


    model.eval()
    with torch.no_grad():
        for images, labels in test_dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            print("Labeluri:", labels[:10])
            print("Predicții:", torch.argmax(outputs, dim=1)[:10])
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            probs = torch.softmax(outputs, dim=1)[:, 1]
            y_scores.extend(probs.cpu().numpy())

    print(classification_report(y_true, y_pred, target_names=["benign", "malignant"]))
    print(Counter(y_pred))

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

    # Generează matricea de confuzie
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])

    # Afișează și salvează imaginea
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    #train vs val loss
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

    # # histograma cu distributia probabilitatilor
    # plt.hist(probs, bins=20, color="purple", alpha=0.7)
    # plt.xlabel("Probabilitate malignitate")
    # plt.ylabel("Număr imagini")
    # plt.title("Distribuția scorurilor de malignitate")
    # plt.savefig("prob_distribution.png")
    # plt.close()

    torch.save(model.state_dict(), "melanoma_model.pth")
    print("Model salvat cu succes.")


if __name__ == "__main__":
    main()
