# This is a sample Python script.
import torch
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from sklearn.metrics import classification_report, confusion_matrix

from nn_class import Net


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    # print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


    y_true = []
    y_pred = []

    model = Net().cuda()

    model.eval()
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print(classification_report(y_true, y_pred, target_names=["benign", "malignant"]))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
