from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime.quantization as oq
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType


class DatasetCalibrationDataReader(CalibrationDataReader):
    def __init__(self, data_dir: Path, input_name: str, img_size: int = 224, max_per_class: int = 15):
        self.input_name = input_name
        self.img_size = img_size
        self.files = []
        for c in ["cat", "dog", "raven"]:
            c_dir = Path(data_dir) / c
            if c_dir.exists():
                self.files.extend(list(c_dir.glob("*.jpg"))[:max_per_class])
        if not self.files:
            valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            self.files = [p for p in sorted(Path(data_dir).rglob("*")) if p.suffix.lower() in valid_exts][:max_per_class * 3]
        self.idx = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.idx >= len(self.files):
            return None
        file_path = self.files[self.idx]
        self.idx += 1
        img = cv2.imread(str(file_path))
        if img is None:
            return self.get_next()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img_f = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = np.transpose((img_f - mean) / std, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0).astype(np.float32)
        return {self.input_name: tensor}

def quantize_dynamic_onnx(
    input_model_path: Path,
    output_model_path: Path,
    weight_type: QuantType = QuantType.QInt8,
    op_types_to_quantize: list[str] | None = None,
    per_channel: bool = True,
) -> Path:
    input_model_path = Path(input_model_path)
    output_model_path = Path(output_model_path)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    oq.quantize_dynamic(
        model_input=str(input_model_path),
        model_output=str(output_model_path),
        weight_type=weight_type,
        op_types_to_quantize=op_types_to_quantize,
        per_channel=per_channel,
    )
    return output_model_path

def quantize_static_onnx(
    input_model_path: Path,
    output_model_path: Path,
    dataset_dir: Path,
    img_size: int = 224,
    max_per_class: int = 15,
) -> Path:
    input_model_path = Path(input_model_path)
    output_model_path = Path(output_model_path)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(input_model_path))
    input_name = model.graph.input[0].name
    reader = DatasetCalibrationDataReader(dataset_dir, input_name, img_size, max_per_class)
    oq.quantize_static(
        model_input=str(input_model_path),
        model_output=str(output_model_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )
    return output_model_path
