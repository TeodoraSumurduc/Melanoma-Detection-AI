import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
from PIL import Image
from nn_class import Net
import random
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from nn_class import Net  # asigură-te că importul e corect
import numpy as np

def apply_model(path, num_trials=10):
    img_size = 64

    def get_transform():
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
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

    def load_image(image_path):
        img = Image.open(image_path).convert("RGB")
        image = get_transform()(img).unsqueeze(0).cuda()
        return image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load("melanoma_model.pth"))
    model.eval()

    softmax_scores = []

    with torch.no_grad():
        for _ in range(num_trials):
            image = load_image(path)
            output = model(image)
            probs = F.softmax(output, dim=1).cpu().numpy()[0]
            softmax_scores.append(probs)

    mean_probs = np.mean(softmax_scores, axis=0)
    final_pred = np.argmax(mean_probs)
    label_names = ["benign", "malignant"]

    return {
        "image_path": path,
        "predictions": softmax_scores,
        "mean_probabilities": {
            "benign": round(float(mean_probs[0]) * 100, 2),
            "malignant": round(float(mean_probs[1]) * 100, 2)
        },
        "final_prediction": label_names[final_pred]
    }


print(apply_model("demo_pics/melanoma_10117.jpg"))
