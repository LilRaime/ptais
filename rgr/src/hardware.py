import platform
from typing import Any

import psutil


def get_system_specs() -> dict[str, Any]:
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu": platform.processor(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
    }

def audit_hardware_devices() -> dict[str, Any]:
    devices = {
        "CPU": {"available": True, "details": platform.processor(), "reason": "Host CPU available"},
        "GPU": {"available": False, "details": None, "reason": "Not detected"},
        "NPU": {"available": False, "details": None, "reason": "Not detected"},
    }

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            devices["GPU"]["available"] = True
            devices["GPU"]["details"] = f"PyTorch CUDA: {gpu_name} (Count: {torch.cuda.device_count()})"
            devices["GPU"]["reason"] = "NVIDIA CUDA driver and GPU operational"
    except Exception:
        pass

    try:
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        devices["ort_providers"] = available_providers

        if "CUDAExecutionProvider" in available_providers and not devices["GPU"]["available"]:
            devices["GPU"]["available"] = True
            devices["GPU"]["details"] = "ONNX Runtime CUDAExecutionProvider"
            devices["GPU"]["reason"] = "CUDA Execution Provider available"
        elif "DmlExecutionProvider" in available_providers and not devices["GPU"]["available"]:
            devices["GPU"]["available"] = True
            devices["GPU"]["details"] = "ONNX Runtime DirectML (DirectX 12 GPU)"
            devices["GPU"]["reason"] = "DirectML provider available"
    except Exception:
        devices["ort_providers"] = ["CPUExecutionProvider"]

    try:
        import openvino as ov
        core = ov.Core()
        ov_devices = core.available_devices
        devices["openvino_devices"] = ov_devices

        gpu_devices = [d for d in ov_devices if d.startswith("GPU")]
        if gpu_devices and not devices["GPU"]["available"]:
            gpu_dev = gpu_devices[0]
            try:
                name = core.get_property(gpu_dev, "FULL_DEVICE_NAME")
            except Exception:
                name = gpu_dev
            devices["GPU"]["available"] = True
            devices["GPU"]["details"] = f"OpenVINO {gpu_dev}: {name}"
            devices["GPU"]["reason"] = "OpenVINO GPU target operational"

        if "NPU" in ov_devices:
            try:
                npu_name = core.get_property("NPU", "FULL_DEVICE_NAME")
            except Exception:
                npu_name = "Intel AI Boost / NPU"
            devices["NPU"]["available"] = True
            devices["NPU"]["details"] = f"OpenVINO NPU: {npu_name}"
            devices["NPU"]["reason"] = "Dedicated NPU hardware detected"
        else:
            devices["NPU"]["available"] = False
            devices["NPU"]["details"] = None
            devices["NPU"]["reason"] = "No Intel NPU / AMD XDNA NPU hardware detected on this host processor"
    except Exception:
        devices["openvino_devices"] = ["CPU"]

    if not devices["GPU"]["available"]:
        devices["GPU"]["reason"] = "No CUDA/DirectML/OpenVINO GPU detected"

    return devices
