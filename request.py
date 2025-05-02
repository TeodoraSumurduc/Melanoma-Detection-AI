import requests

url = "https://melanoma-detection-ai-563719261463.us-central1.run.app/predict"
image_path = "demo_pics/melanoma_10117.jpg"  # pune aici calea unei imagini reale

with open(image_path, 'rb') as img:
    files = {'image': img}
    response = requests.post(url, files=files)

print("Status code:", response.status_code)
print("Response:", response.text)

# resp = requests.post("http://127.0.0.1:5000/predict", files={'image': open("demo_pics/melanoma_10179.jpg", 'rb')})
#
# print(resp.json())