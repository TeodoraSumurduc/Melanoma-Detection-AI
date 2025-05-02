import os

from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
from nn_class import Net  # asigură-te că Net e modelul tău
import io

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
model.load_state_dict(torch.load("melanoma_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        image_file = request.files.get('image')
        if image_file is None or image_file.filename == "":
            return jsonify({"error": "no image"})

    image_file = request.files['image']
    image = Image.open(image_file).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence = torch.max(probabilities).item() * 100
        pred_class = torch.argmax(probabilities, dim=1).item()

    result = {
        'prediction': 'malignant' if pred_class == 1 else 'benign',
        'confidence': round(confidence, 2)
    }

    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, port=port)
