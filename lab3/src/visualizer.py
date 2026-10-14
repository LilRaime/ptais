import cv2
import numpy as np

def draw_overlay(frame, top_class, confidence, class_probs, fps, latency_ms, model_name=""):
    h, w = frame.shape[:2]
    panel_w = min(360, w - 20)
    panel_h = 175
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (18, 18, 24), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (80, 80, 100), 1)

    title_text = f"MODEL: {model_name.upper()}" if model_name else "CLASSIFICATION"
    cv2.putText(frame, title_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 180), 1, cv2.LINE_AA)

    status_color = (60, 220, 90) if confidence >= 0.7 else (0, 200, 255)
    class_label = f"{top_class.upper()}  {confidence * 100:.1f}%"
    cv2.putText(frame, class_label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2, cv2.LINE_AA)

    metrics_str = f"FPS: {fps:5.1f}   Latency: {latency_ms:5.1f} ms"
    cv2.putText(frame, metrics_str, (20, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    cv2.line(frame, (20, 94), (10 + panel_w - 10, 94), (60, 60, 75), 1)

    y = 112
    max_bar_w = panel_w - 150
    for c_name, prob in class_probs.items():
        name_str = f"{c_name[:7]:<7} {prob * 100:4.1f}%"
        cv2.putText(frame, name_str, (20, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA)
        bar_len = int(prob * max_bar_w)
        cv2.rectangle(frame, (120, y - 6), (120 + max_bar_w, y + 6), (40, 40, 50), -1)
        bar_color = (50, 210, 90) if c_name == top_class else (200, 140, 50)
        if bar_len > 0:
            cv2.rectangle(frame, (120, y - 6), (120 + bar_len, y + 6), bar_color, -1)
        y += 22

    return frame
