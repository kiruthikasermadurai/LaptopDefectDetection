

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Define transformations (matching training pipeline)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match MobileNetV2 input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as per ImageNet standards
])

# Define the Custom MobileNetV2 Model
class CustomMobileNet(nn.Module):
    def __init__(self):
        super(CustomMobileNet, self).__init__()
        self.mobilenet = models.mobilenet_v2()
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # 2 Classes: Defectless & Defective
        )

    def forward(self, x):
        return self.mobilenet(x)

# Load trained model
MODEL_NAME = "MobileNetV2"
MODEL_PATH = r"C:\Users\SANJITHA\Projects\HPE\LaptopDefectDetection\backend\models\mobilenetv2_model.pth"
model = CustomMobileNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()  # Set model to evaluation mode

@app.route("/")
def hello():
    return "Hello World"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    
    # Load and preprocess the image
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Get prediction
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)  # Apply softmax to get confidence scores
        confidence, predicted_class = torch.max(probabilities, 1)  # Get max confidence & class index

    # Convert to frontend-friendly format
    has_defect = bool(predicted_class.item())  # True if class 1 (Defective), False if class 0 (No Defect)
    confidence = round(confidence.item(), 4)  # Round confidence for better readability

    return jsonify({
        "hasDefect": has_defect,
        "confidence": confidence,
        "model": MODEL_NAME
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
