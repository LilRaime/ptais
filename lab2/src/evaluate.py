from dataclasses import dataclass
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
import torch

@dataclass
class PredictionSample:
    path: str
    true_name: str
    pred_name: str
    confidence: float
    subset: str
    explanation: str

def explain_misclassification(true_name, pred_name, conf):
    pair = (true_name.lower(), pred_name.lower())
    reasons = {
        ("cat", "dog"): "Similar fur texture, curled posture, or facial structure",
        ("dog", "cat"): "Small breed size, pointed ears, or close-up cropping",
        ("raven", "cat"): "Dark feathers on shadowed background mistaken for fur",
        ("raven", "dog"): "Dark silhouette on ground mistaken for canine",
        ("cat", "raven"): "Dark monochrome coat or deep cast shadows",
        ("dog", "raven"): "Dark monochrome coat or deep cast shadows",
    }
    desc = reasons.get(pair, "Atypical camera angle or cluttered background")
    return f"{desc} (conf: {conf * 100:.1f}%)"

def evaluate_model(model, test_loader, class_names, device="cpu", val_loader=None, desired_samples=5):
    device = torch.device(device)
    model.to(device).eval()

    def predict(loader, subset):
        trues, preds, probs, paths = [], [], [], []
        with torch.no_grad():
            for x, y, p in loader:
                out = model(x.to(device))
                trues.extend(y.tolist())
                preds.extend(out.argmax(1).cpu().tolist())
                probs.extend(torch.softmax(out, 1).cpu().tolist())
                paths.extend(p)
        return trues, preds, probs, paths

    y_t, y_p, p_all, paths = predict(test_loader, "test")
    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, average="macro", zero_division=0))
    rec = float(recall_score(y_t, y_p, average="macro", zero_division=0))
    cm = confusion_matrix(y_t, y_p, labels=list(range(len(class_names))))
    rep = classification_report(y_t, y_p, target_names=class_names, output_dict=True, zero_division=0)

    correct, incorrect = [], []
    for t, p, probs, path in zip(y_t, y_p, p_all, paths):
        conf = float(probs[p])
        tn, pn = class_names[t], class_names[p]
        exp = f"Correctly predicted (conf: {conf * 100:.1f}%)" if t == p else explain_misclassification(tn, pn, conf)
        item = PredictionSample(path, tn, pn, conf, "test", exp)
        (correct if t == p else incorrect).append(item)

    if len(incorrect) < desired_samples and val_loader:
        vy_t, vy_p, vp_all, vpaths = predict(val_loader, "val")
        for t, p, probs, path in zip(vy_t, vy_p, vp_all, vpaths):
            if t != p and len(incorrect) < desired_samples:
                conf = float(probs[p])
                tn, pn = class_names[t], class_names[p]
                incorrect.append(PredictionSample(path, tn, pn, conf, "val", explain_misclassification(tn, pn, conf)))

    correct.sort(key=lambda s: s.confidence, reverse=True)
    incorrect.sort(key=lambda s: s.confidence, reverse=True)
    return acc, prec, rec, cm, rep, correct[:desired_samples], incorrect[:desired_samples]
