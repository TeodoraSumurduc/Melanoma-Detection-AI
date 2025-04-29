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

def apply_model(path, num_trials=3):
    img_size = 64

    def get_transform():
        flip_prob = random.random()
        rotation_deg = random.uniform(0, 25)
        brightness = np.random.uniform(0.0, 0.5)
        contrast = np.random.uniform(0.0, 0.25)
        saturation = np.random.uniform(0.0, 0.25)
        hue = np.random.uniform(-0.05, 0.05)
        hue_tuple = (0, hue) if hue >= 0 else (hue, 0)

        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
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

    def load_image(image_path):
        img = Image.open(image_path).convert("RGB")
        image = get_transform()(img).unsqueeze(0).cuda()
        return image

    model = Net().cuda()
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
        "mean_probabilities": {
            "benign": round(float(mean_probs[0]) * 100, 2),
            "malignant": round(float(mean_probs[1]) * 100, 2)
        },
        "final_prediction": label_names[final_pred]
    }


print(apply_model("demo_pics/melanoma_10140.jpg"))