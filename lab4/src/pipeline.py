from pathlib import Path
import cv2
import numpy as np

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

def preprocess_image(frame, img_size=(224, 224)):
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, img_size, interpolation=cv2.INTER_LINEAR)
    img = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = np.transpose((img - mean) / std, (2, 0, 1))
    return np.expand_dims(tensor, axis=0).astype(np.float32)

def postprocess_output(outputs, class_names):
    logits = np.squeeze(outputs)
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / np.sum(exp_logits)
    top_idx = int(np.argmax(probabilities))
    return class_names[top_idx], float(probabilities[top_idx]), {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
