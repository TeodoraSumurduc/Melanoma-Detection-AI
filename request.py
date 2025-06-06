import requests

# url = "https://melanoma-detection-ai-563719261463.us-central1.run.app/predict"
# image_path = "demo_pics/melanoma_10117.jpg"  # pune aici calea unei imagini reale
#
# with open(image_path, 'rb') as img:
#     files = {'image': img}
#     response = requests.post(url, files=files)
#
# print("Status code:", response.status_code)
# print("Response:", response.text)

# resp = requests.post("https://melanoma-detection-ai.onrender.com/predict", files={'image': open("demo_pics/melanoma_10179.jpg", 'rb')})
#
# print("Status code:", resp.status_code)
# print("Response:", resp.text)
import os
import shutil
import random

# Calea către folderul original de test
original_test_dir = 'demo_pics/test'
# Calea unde vom salva noile foldere
base_output_dir = 'melanoma_cancer_dataset'

# Folderele noi
val_dir = os.path.join(base_output_dir, 'val')
new_test_dir = os.path.join(base_output_dir, 'new_test')  # redenumim pentru a evita confuzii

# Clasele
classes = ['benign', 'malignant']

# Creează folderele de ieșire
for target_dir in [val_dir, new_test_dir]:
    for cls in classes:
        os.makedirs(os.path.join(target_dir, cls), exist_ok=True)

# Mutarea fișierelor
for cls in classes:
    cls_dir = os.path.join(original_test_dir, cls)
    images = os.listdir(cls_dir)
    random.shuffle(images)

    split_point = len(images) // 2
    val_images = images[:split_point]
    test_images = images[split_point:]

    # Mutăm în val
    for img in val_images:
        src = os.path.join(cls_dir, img)
        dst = os.path.join(val_dir, cls, img)
        shutil.move(src, dst)

    # Mutăm în new_test
    for img in test_images:
        src = os.path.join(cls_dir, img)
        dst = os.path.join(new_test_dir, cls, img)
        shutil.move(src, dst)

print("Împărțirea în seturi de validare și testare a fost realizată cu succes.")
