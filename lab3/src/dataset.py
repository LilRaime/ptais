from pathlib import Path
import random
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, samples, transform=None, preload=True, target_size=(224, 224)):
        self.samples = samples
        self.transform = transform
        self.cached = [] if preload else None
        if preload:
            for path, _ in self.samples:
                with Image.open(path) as img:
                    self.cached.append(img.convert("RGB").resize(target_size))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.cached[idx] if self.cached is not None else Image.open(path).convert("RGB")
        tensor = self.transform(img) if self.transform else transforms.ToTensor()(img)
        return tensor, label, str(path)

def create_dataloaders(dataset_dir="../lab1/dataset", batch_size=16, img_size=(224, 224), seed=42, preload=True):
    random.seed(seed)
    root = Path(dataset_dir)
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    
    train_s, val_s, test_s = [], [], []
    for idx, c in enumerate(classes):
        files = [p for p in (root / c).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        random.shuffle(files)
        n = len(files)
        n_tr, n_val = int(n * 0.70), int(n * 0.15)
        train_s.extend([(p, idx) for p in files[:n_tr]])
        val_s.extend([(p, idx) for p in files[n_tr:n_tr + n_val]])
        test_s.extend([(p, idx) for p in files[n_tr + n_val:]])

    for s in (train_s, val_s, test_s):
        random.shuffle(s)

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    t_train = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        norm
    ])
    t_eval = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        norm
    ])

    ds_tr = ImageDataset(train_s, t_train, preload=preload, target_size=img_size)
    ds_val = ImageDataset(val_s, t_eval, preload=preload, target_size=img_size)
    ds_test = ImageDataset(test_s, t_eval, preload=preload, target_size=img_size)

    loaders = [
        DataLoader(ds_tr, batch_size=batch_size, shuffle=True),
        DataLoader(ds_val, batch_size=batch_size, shuffle=False),
        DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    ]
    return *loaders, classes, (ds_tr, ds_val, ds_test)
