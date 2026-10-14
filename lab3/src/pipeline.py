from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision.transforms as T

class ImageStream:
    def __init__(self, source):
        self.source = Path(source)
        self.image_paths = []
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        if self.source.is_dir():
            self.image_paths = [p for p in sorted(self.source.rglob("*")) if p.suffix.lower() in valid_exts]
        elif self.source.is_file() and self.source.suffix.lower() in valid_exts:
            self.image_paths = [self.source]
        else:
            raise FileNotFoundError(f"Image or directory not found: {source}")

        if not self.image_paths:
            raise ValueError(f"No valid images found in: {source}")

    def __len__(self):
        return len(self.image_paths)

    def __iter__(self):
        for path in self.image_paths:
            frame = cv2.imread(str(path))
            if frame is not None:
                yield frame, path

_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def preprocess_frame(frame, img_size=(224, 224), device="cpu"):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, img_size, interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    tensor = _NORM(tensor).unsqueeze(0).to(device)
    return tensor

def postprocess_output(outputs, class_names):
    probabilities = torch.softmax(outputs, dim=1)[0].detach().cpu().numpy()
    top_idx = int(np.argmax(probabilities))
    top_class = class_names[top_idx]
    top_confidence = float(probabilities[top_idx])
    prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    return top_class, top_confidence, prob_dict
