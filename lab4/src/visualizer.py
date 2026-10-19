import cv2

def draw_overlay(frame, top_class, confidence, class_probs, fps, latency_ms, engine_name="ONNX Runtime", model_name=""):
    frame = frame.copy()
    h, w = frame.shape[:2]
    panel_w = min(380, w - 20)
    panel_h = 185
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (18, 18, 24), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (80, 80, 100), 1)

    header = f"ENGINE: {engine_name.upper()}" + (f" | {model_name}" if model_name else "")
    cv2.putText(frame, header, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 200), 1, cv2.LINE_AA)

    status_color = (60, 220, 90) if confidence >= 0.7 else (0, 200, 255)
    cv2.putText(frame, f"{top_class.upper()}  {confidence * 100:.1f}%", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:5.1f}   Latency: {latency_ms:5.2f} ms", (20, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.line(frame, (20, 94), (10 + panel_w - 10, 94), (60, 60, 75), 1)

    y = 114
    max_bar_w = panel_w - 150
    for c_name, prob in class_probs.items():
        cv2.putText(frame, f"{c_name[:7]:<7} {prob * 100:4.1f}%", (20, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (120, y - 6), (120 + max_bar_w, y + 6), (40, 40, 50), -1)
        bar_len = int(prob * max_bar_w)
        if bar_len > 0:
            color = (50, 210, 90) if c_name == top_class else (200, 140, 50)
            cv2.rectangle(frame, (120, y - 6), (120 + bar_len, y + 6), color, -1)
        y += 22

    return frame
