
import cv2


def draw_overlay(
    frame,
    top_class: str,
    confidence: float,
    class_probs: dict[str, float],
    fps: float,
    timing: dict[str, float],
    engine_name: str = "ONNX Runtime",
    model_name: str = "MobileNetV3",
):
    frame = frame.copy()
    h, w = frame.shape[:2]
    panel_w = min(400, w - 20)
    panel_h = 220
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (16, 18, 24), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (75, 80, 95), 1)

    header = f"{engine_name.upper()} | {model_name}"
    cv2.putText(frame, header, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 175, 200), 1, cv2.LINE_AA)

    status_color = (60, 220, 90) if confidence >= 0.7 else (0, 200, 255)
    cv2.putText(frame, f"{top_class.upper()}  {confidence * 100:.1f}%", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2, cv2.LINE_AA)
    
    total_ms = timing.get("total_ms", 0.0)
    cv2.putText(frame, f"FPS: {fps:5.1f}   Total: {total_ms:5.2f} ms", (20, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.line(frame, (20, 94), (10 + panel_w - 10, 94), (60, 65, 80), 1)

    y = 114
    max_bar_w = panel_w - 150
    for c_name, prob in class_probs.items():
        cv2.putText(frame, f"{c_name[:7]:<7} {prob * 100:4.1f}%", (20, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (120, y - 6), (120 + max_bar_w, y + 6), (35, 40, 50), -1)
        bar_len = int(prob * max_bar_w)
        if bar_len > 0:
            color = (50, 210, 90) if c_name == top_class else (200, 140, 50)
            cv2.rectangle(frame, (120, y - 6), (120 + bar_len, y + 6), color, -1)
        y += 20

    cv2.line(frame, (20, y), (10 + panel_w - 10, y), (60, 65, 80), 1)
    y += 16
    pre_ms = timing.get("preprocess_ms", 0.0)
    inf_ms = timing.get("inference_ms", 0.0)
    post_ms = timing.get("postprocess_ms", 0.0)
    stages_text = f"Pre: {pre_ms:.2f}ms | Inf: {inf_ms:.2f}ms | Post: {post_ms:.2f}ms"
    cv2.putText(frame, stages_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 210, 255), 1, cv2.LINE_AA)

    return frame
