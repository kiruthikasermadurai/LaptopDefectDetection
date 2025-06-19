from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import torchvision.models as models
import torch.nn as nn
import os
from ultralytics import YOLO
import numpy as np
import cv2
import base64

app = Flask(__name__)
CORS(app)

class_colors = {
    0: (0, 0, 255),    
    1: (0, 255, 255),  
    2: (255, 0, 0)     
}


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


class CustomResNet50MultiLabel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.resnet(x)

class CustomMobileNetMultiLabel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.base = models.mobilenet_v2(weights=None)
        self.base.classifier[1] = nn.Linear(self.base.last_channel, num_classes)

    def forward(self, x):
        return self.base(x)


def load_model(model_name):
    model_name = model_name.lower()
    base_dir = os.path.dirname(__file__)

    if model_name == "mobilenet":
        model = CustomMobileNetMultiLabel()
        path = os.path.join(base_dir, 'models', 'best_model_final1.pth')
        model.load_state_dict(torch.load(path, map_location="cpu"))
        label_map = {
            0: 'crack',
            1: 'others',
            2: 'lines',
            3: 'defectless'
        }
        model.eval()
        return model, label_map

    elif model_name == "resnet-50":
        model = CustomResNet50MultiLabel()
        path = os.path.join(base_dir, 'models', 'best_model_50_4.pth')
        model.load_state_dict(torch.load(path, map_location="cpu"))
        label_map = {
            0: 'crack',
            1: 'lines',
            2: 'defectless',
            3: 'others'
        }
        model.eval()
        return model, label_map

    elif model_name == "yolo":
        path = os.path.join(base_dir, 'models', 'best.pt')
        label_map = {
            0: 'crack',
            1: 'lines',
            2: 'others',
            3: 'defectless'
        }
        return YOLO(path), label_map

    else:
        raise ValueError("Unsupported model selected.")


def process_image(image_file, model, model_name, label_map):
    image_file.seek(0)
    image = Image.open(io.BytesIO(image_file.read())).convert("RGB")

    if isinstance(model, YOLO):
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        image_height, image_width = image_bgr.shape[:2]
        image_overlay = image_bgr.copy()

        results = model(image)[0]
        masks = results.masks.data if results.masks else []
        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []
        raw_classes = [int(cls.item()) for cls in results.boxes.cls] if results.boxes else []
        confs = [float(conf.item()) for conf in results.boxes.conf] if results.boxes else []


        if len(masks) == 0:
            has_defect = False
            predicted_labels = ["defectless"]
            confidence=0.0
        else:
            has_defect = True
            predicted_labels = []
            confidences=[]
            for i, class_id in enumerate(raw_classes):
                if class_id not in label_map:
                    continue
                if label_map[class_id] not in predicted_labels:
                    predicted_labels.append(label_map[class_id])
                confidences.append(confs[i])

                mask = masks[i].cpu().numpy().astype(np.uint8) * 255
                mask_resized = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                color = class_colors.get(class_id, (255, 255, 255))
                colored_mask = np.zeros_like(image_bgr)
                for c in range(3):
                    colored_mask[:, :, c] = color[c]

                alpha = 0.4
                image_overlay = np.where(mask_resized[:, :, None] == 255,
                                         (1 - alpha) * image_overlay + alpha * colored_mask,
                                         image_overlay).astype(np.uint8)
                x1, y1, x2, y2 = [int(v) for v in boxes[i]]
                cv2.putText(image_overlay, f"{label_map[class_id]}", (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, lineType=cv2.LINE_AA)
            confidence=max(confidences) if confidences else 1.0

        _, buffer = cv2.imencode('.jpg', image_overlay)
        img_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')

        return {
            "filename": image_file.filename,
            "hasDefect": has_defect,
            "confidence": round(confidence, 4),
            "label": ", ".join(predicted_labels),
            "model": model_name,
            "predicted_class": ", ".join(predicted_labels),
            "annotated_image_base64": img_base64
        }

    else:
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            raw_output = model(input_tensor)
            output = torch.sigmoid(raw_output.squeeze(0))


        
        threshold = 0.5
        detected_indices = (output >= threshold).nonzero(as_tuple=True)[0].tolist()

        if detected_indices:
            predicted_labels_raw = [label_map[i] for i in detected_indices]
            confidences = output[detected_indices].tolist()
            predicted_labels = [lbl for lbl in predicted_labels_raw if lbl != "defectless"]

            if not predicted_labels:
                predicted_labels = ["defectless"]
                has_defect = False
                confidence = confidences[predicted_labels_raw.index("defectless")]
            else:
                has_defect = True
                confidence = max(confidences)
        else:
            predicted_labels = ["defectless"]
            has_defect = False
            confidence = 0.0

        return {
            "filename": image_file.filename,
            "hasDefect": has_defect,
            "confidence": round(confidence, 4),
            "label": ", ".join(predicted_labels),
            "model": model_name,
            "predicted_class": ", ".join(predicted_labels)
        }


@app.route("/api/detect", methods=["POST"])
def detect_single():
    if "image" not in request.files or "model_name" not in request.form:
        return jsonify({"error": "Missing image or model_name"}), 400

    image_file = request.files["image"]
    model_name = request.form["model_name"]

    try:
        model, label_map = load_model(model_name)
        result = process_image(image_file, model, model_name, label_map)
        return jsonify(result)
    except Exception as e:
        import traceback
        print(f"Error in detect_single: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    
@app.route("/api/detect/all", methods=["POST"])
def detect_all_models_single():
    if "image" not in request.files:
        return jsonify({"error": "Missing image"}), 400

    image_file = request.files["image"]
    results = {}

    try:
        for model_name in ["yolo", "resnet-50", "mobilenet"]:
            image_file.stream.seek(0)  # Reset file pointer
            model, label_map = load_model(model_name)
            result = process_image(image_file, model, model_name, label_map)
            results[model_name] = result
        return jsonify(results)
    except Exception as e:
        import traceback
        print(f"Error in detect_all_models_single: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/detect/batch/all", methods=["POST"])
def detect_all_models_batch():
    if "images" not in request.files:
        return jsonify({"error": "Missing images"}), 400

    image_files = request.files.getlist("images")
    all_results = {"yolo": [], "resnet-50": [], "mobilenet": []}

    try:
        for model_name in ["yolo", "resnet-50", "mobilenet"]:
            model, label_map = load_model(model_name)
            for image_file in image_files:
                image_file.stream.seek(0)  # Reset file pointer
                result = process_image(image_file, model, model_name, label_map)
                all_results[model_name].append(result)
        return jsonify(all_results)
    except Exception as e:
        import traceback
        print(f"Error in detect_all_models_batch: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/detect/batch", methods=["POST"])
def detect_batch():
    if "images" not in request.files or "model_name" not in request.form:
        return jsonify({"error": "Missing images or model_name"}), 400

    image_files = request.files.getlist("images")
    model_name = request.form["model_name"]

    try:
        model, label_map = load_model(model_name)
        results = [process_image(f, model, model_name, label_map) for f in image_files]
        return jsonify(results)
    except Exception as e:
        import traceback
        print(f"Error in detect_batch: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})

@app.route("/api/classes", methods=["GET"])
def get_classes():
    model_name = request.args.get("model_name", "").lower()
    try:
        _, label_map = load_model(model_name)
        return jsonify({"classes": label_map})
    except Exception:
        return jsonify({"error": "Invalid model name"}), 400

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    app.run(debug=True, port=5000)
