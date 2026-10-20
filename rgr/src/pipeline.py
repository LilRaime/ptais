import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort


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
            raise FileNotFoundError(f"Source not found: {source}")

        if not self.image_paths:
            raise ValueError(f"No valid images found in: {source}")

    def __len__(self):
        return len(self.image_paths)

    def __iter__(self) -> Generator[tuple[np.ndarray, Path], None, None]:
        for path in self.image_paths:
            frame = cv2.imread(str(path))
            if frame is not None:
                yield frame, path

def preprocess_image(frame: np.ndarray, img_size: int = 224) -> np.ndarray:
    if isinstance(img_size, int):
        target_size = (img_size, img_size)
    else:
        target_size = img_size
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR)
    img = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = np.transpose((img - mean) / std, (2, 0, 1))
    return np.expand_dims(tensor, axis=0).astype(np.float32)

def postprocess_output(outputs: np.ndarray, class_names: list[str]) -> tuple[str, float, dict[str, float]]:
    logits = np.squeeze(outputs)
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    top_idx = int(np.argmax(probs))
    return class_names[top_idx], float(probs[top_idx]), {class_names[i]: float(probs[i]) for i in range(len(class_names))}

class BaseEngine:
    def infer(self, tensor: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class OrtEngine(BaseEngine):
    def __init__(self, model_path: Path, provider: str = "CPUExecutionProvider"):
        self.model_path = Path(model_path)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=[provider])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        return self.session.run([self.output_name], {self.input_name: tensor})[0]

class OpenVinoEngine(BaseEngine):
    def __init__(self, model_path: Path, device: str = "CPU"):
        import openvino as ov
        self.core = ov.Core()
        model = self.core.read_model(str(model_path))
        self.compiled = self.core.compile_model(model, device)
        self.output_layer = self.compiled.output(0)

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        return self.compiled([tensor])[self.output_layer]

class EndToEndPipeline:
    def __init__(self, engine: BaseEngine, class_names: list[str], img_size: int = 224):
        self.engine = engine
        self.class_names = class_names
        self.img_size = img_size

    def process_frame(self, frame: np.ndarray) -> tuple[str, float, dict[str, float], dict[str, float]]:
        t0 = time.perf_counter()
        tensor = preprocess_image(frame, self.img_size)
        t_pre = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        outputs = self.engine.infer(tensor)
        t_inf = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        top_cls, conf, probs = postprocess_output(outputs, self.class_names)
        t_post = (time.perf_counter() - t2) * 1000.0

        timing = {
            "preprocess_ms": t_pre,
            "inference_ms": t_inf,
            "postprocess_ms": t_post,
            "total_ms": t_pre + t_inf + t_post,
        }
        return top_cls, conf, probs, timing

def analyze_bottleneck(stage_stats: dict[str, float]) -> dict[str, Any]:
    pre = stage_stats.get("preprocess_ms", 0.0)
    inf = stage_stats.get("inference_ms", 0.0)
    post = stage_stats.get("postprocess_ms", 0.0)
    total = pre + inf + post
    if total <= 0:
        total = 1e-6

    pct_pre = (pre / total) * 100.0
    pct_inf = (inf / total) * 100.0
    pct_post = (post / total) * 100.0

    stages = {"Preprocessing": pct_pre, "Inference": pct_inf, "Postprocessing": pct_post}
    bottleneck_stage = max(stages, key=stages.get)

    if bottleneck_stage == "Preprocessing":
        diagnosis = "Pipeline is I/O & Preprocessing-bound. Image decoding, resizing, and float normalization consume more time than neural network execution."
        recommendations = [
            "Use hardware-accelerated image decoders (libjpeg-turbo / NVJPEG / OpenCV with IPP).",
            "Perform color conversion and normalization asynchronously or in parallel threads.",
            "Offload resizing and normalization to GPU/NPU before inference if input stream is in VRAM."
        ]
    elif bottleneck_stage == "Inference":
        diagnosis = "Pipeline is Compute-bound. Neural network weight multiplications dominate total latency."
        recommendations = [
            "Apply model quantization (INT8 / FP16) to reduce memory bandwidth and arithmetic cost.",
            "Enable hardware accelerators (GPU / NPU) or optimize CPU threads and execution providers.",
            "Use batching if multiple input frames or streams can be processed concurrently."
        ]
    else:
        diagnosis = "Postprocessing represents an unexpected overhead."
        recommendations = ["Optimize vector operations in NumPy using vectorized softmax."]

    return {
        "bottleneck_stage": bottleneck_stage,
        "percentages": {"preprocess": pct_pre, "inference": pct_inf, "postprocess": pct_post},
        "diagnosis": diagnosis,
        "recommendations": recommendations,
    }
