from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import random
import torchvision.models as models

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load models
model_ids = {
    "resnet50": {
        "path":torch.load(r"C:\Users\SANJITHA\Projects\HPE\backend\models\resnet50_model.pth", map_location="cpu"),
        "model": models.mobilenet_v2()
        }
    
}

# Define image preprocessing
transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.RandomHorizontalFlip(p=0.5),  # Flip 50% of the time
        transforms.ColorJitter(
            brightness=random.uniform(0.85, 1.15),  # Brightness: 85%-115% (avoid extreme brightness)
            contrast=random.uniform(0.85, 1.15)  # Contrast: 85%-115% (avoid too much darkness/brightness)
        ),
    ])

@app.route("/")
def hello():
    return "Hello World"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files or "model" not in request.form:
        print("No image found")
        return jsonify({"error": "Missing image or model selection"}), 400

    file = request.files["image"]
    model_name = request.form["model"]

    if model_name not in model_ids:
        return jsonify({"error": "Invalid model selection"}), 400

    # Load image
    image = Image.open(io.BytesIO(file.read()))
    image = transform(image)  # Add batch dimension
    print(model_name)
    # Get prediction
    model = model_ids[model_name]["model"]
    model.load_state_dict(model_ids[model_name]["path"])
    model.eval()
    with torch.no_grad():
        output = model(image)

    # Example: Assume 0 = No defect, 1 = Defect
    predicted_class = "Defective Laptop" if torch.argmax(output) == 1 else "No Defect"

    return jsonify({"prediction": predicted_class})

if __name__ == "__main__":
    app.run(debug=True,port=5000)
