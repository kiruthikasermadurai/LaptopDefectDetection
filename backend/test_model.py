import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -------------------------------
# Define the exact same model used for training
# -------------------------------
class CustomResNet50(nn.Module):
    def __init__(self):
        super(CustomResNet50, self).__init__()
        self.resnet = models.resnet50(weights=None)  # No pretrained weights for inference
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
            nn.Linear(256, 2)  # 2 Classes: Defectless (0), Defective (1)
        )

    def forward(self, x):
        return self.resnet(x)

# -------------------------------
# Load the model
# -------------------------------
model = CustomResNet50()
model.load_state_dict(torch.load("models/resnet50_model.pth", map_location="cpu"))
model.eval()

# -------------------------------
# Image preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Match your training normalization
])

# -------------------------------
# Load image for testing
# -------------------------------
image_path = "img6.jpg"  # Replace with your test image
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)  # Add batch dimension

# -------------------------------
# Make prediction
# -------------------------------
with torch.no_grad():
    output = model(image)
    probs = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(probs, 1)

    class_names = ["Defectless", "Defective"]
    print(f"Prediction: {class_names[predicted.item()]}")
    print(f"Confidence: {confidence.item() * 100:.2f}%")
