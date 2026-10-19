from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, resnet18

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def get_model(name="mobilenet_v3_small", num_classes=3):
    name = name.lower()
    if name == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model
    elif name == "resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif name in ["simplecnn", "simple_cnn"]:
        return SimpleCNN(num_classes=num_classes)
    raise ValueError(f"Unknown model name: {name}")

def load_trained_model(weights_path, model_name=None, num_classes=3, device="cpu"):
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    class_names = ["cat", "dog", "raven"]
    img_size = 224

    if isinstance(checkpoint, dict):
        model_name = checkpoint.get("model_name", model_name)
        class_names = checkpoint.get("class_names", class_names)
        img_size = checkpoint.get("img_size", img_size)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    else:
        state_dict = checkpoint

    if model_name is None:
        model_name = "mobilenet_v3_small" if "mobilenet" in str(weights_path).lower() else ("resnet18" if "resnet" in str(weights_path).lower() else "simple_cnn")

    if model_name in ["simplecnn", "simple_cnn"] and "img_size" not in (checkpoint if isinstance(checkpoint, dict) else {}):
        img_size = 128

    model = get_model(name=model_name, num_classes=len(class_names))
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, model_name, class_names, img_size
