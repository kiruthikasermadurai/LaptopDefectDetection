

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Define transformations (matching training pipeline)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match MobileNetV2 input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as per ImageNet standards
])

class CustomResNet50(nn.Module):
    def __init__(self):
        super(CustomResNet50, self).__init__()
        self.resnet = models.resnet50(weights=None)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.resnet(x)

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
# -------------------------------
# Model Loader
# -------------------------------
def load_model(model_name):
    model_name = model_name.lower()
    if model_name == "mobilenet":
        model = CustomMobileNet()
        path =  os.path.join("models", "mobilenetv2_model.pth")

    elif model_name == "resnet-50":
        model = CustomResNet50()
        path =  os.path.join("models", "resnet50_model.pth")

    elif model_name == "yolo":
        raise NotImplementedError("YOLO integration coming soon!")
    else:
        raise ValueError("Unsupported model selected.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model weights not found at {path}")

    try:
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_name}': {str(e)}")


def process_image(image_file, model):
    try:
        image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        return {
            "filename": image_file.filename,
            "hasDefect": bool(predicted_class.item()),
            "confidence": round(confidence.item(), 4)
        }

    except Exception as e:
        return {
            "filename": image_file.filename,
            "error": f"Processing failed: {str(e)}"
        }

# -------------------------------
# API Routes
# -------------------------------
@app.route("/")
def hello():
    return "Hello from Defect Detection API "

@app.route("/api/detect", methods=["POST"])
def detect_single():
    if "image" not in request.files or "model_name" not in request.form:
        return jsonify({"error": "Missing image or model_name"}), 400

    image_file = request.files["image"]
    model_name = request.form["model_name"]

    try:
        model = load_model(model_name)
        result = process_image(image_file, model)
        result["model"] = model_name
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/detect/batch", methods=["POST"])
def detect_batch():
    if "images" not in request.files or "model_name" not in request.form:
        return jsonify({"error": "Missing images or model_name"}), 400

    image_files = request.files.getlist("images")
    model_name = request.form["model_name"]

    try:
        model = load_model(model_name)
        results = [process_image(file, model) for file in image_files]
        for res in results:
            res["model"] = model_name
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

