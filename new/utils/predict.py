import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

def normalize_class_name(name):
    return name.strip().lower().replace(" ", "_")

# Define your classes
class_names = [normalize_class_name(name) for name in [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]]


# Load the JSON file
with open("model/plant_disease_tips.json", "r") as f:
    class_info_dict = json.load(f)


class_info_map = {
    normalize_class_name(class_name): {
        "PreventionTips": info.get("PreventionTips", "Not available"),
        "FertilizerRecommendations": info.get("FertilizerRecommendation", "Not available")
    }
    for class_name, info in class_info_dict.items()
}

print("🗂 Normalized keys in class_info_map:")
for k in list(class_info_map.keys())[:10]:  # Just preview 10
    print(k)




# Define the image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load the trained model
def load_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 256),  
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, len(class_names))  # 38 classes
    )
    state_dict = torch.load("model/best_model.pth", map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# Predict function
def predict_image(image_path):
    if not os.path.exists(image_path):
        return {
            "PredictedClass": "❌ Image not found",
            "PreventionTips": "N/A",
            "FertilizerRecommendations": "N/A"
        }

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = normalize_class_name(class_names[predicted.item()])



    info = class_info_map.get(predicted_class, {
        "PreventionTips": "Not available",
        "FertilizerRecommendations": "Not available"
    })

    return {
        "PredictedClass": predicted_class,
        "PreventionTips": info["PreventionTips"],
        "FertilizerRecommendations": info["FertilizerRecommendations"]
    }
