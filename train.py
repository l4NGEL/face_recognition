import os, random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torch.nn import TripletMarginLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FaceNet(nn.Module):
    def __init__(self, embedding_size=128):  
        super(FaceNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)  
        self.fc = nn.Linear(512, embedding_size)

    def _make_layer(self, in_c, out_c, blocks, stride=1):
        layers = [nn.Conv2d(in_c, out_c, 3, stride, 1),
                  nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]
        for _ in range(1, blocks):
            layers += [nn.Conv2d(out_c, out_c, 3, 1, 1),
                       nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x).view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        centers_batch = self.centers[labels]
        return ((features - centers_batch) ** 2).sum() / features.size(0)


class TripletDataset(Dataset):
    def __init__(self, folder_dataset, transform=None):
        self.dataset = folder_dataset
        self.transform = transform
        self.label_to_images = {}
        for path, label in folder_dataset.samples:
            self.label_to_images.setdefault(label, []).append(path)
        self.labels = list(self.label_to_images.keys())
        self.triplets = self.create_triplets()

    def create_triplets(self):
        triplets = []
        for label in self.labels:
            imgs = self.label_to_images[label]
            if len(imgs) < 2: continue
            for anchor in imgs:
                pos = random.choice([img for img in imgs if img != anchor])
                neg_label = random.choice([l for l in self.labels if l != label])
                neg = random.choice(self.label_to_images[neg_label])
                triplets.append((anchor, pos, neg))
        return triplets

    def __getitem__(self, idx):
        a, p, n = self.triplets[idx]
        a, p, n = Image.open(a).convert("RGB"), Image.open(p).convert("RGB"), Image.open(n).convert("RGB")
        if self.transform:
            a, p, n = self.transform(a), self.transform(p), self.transform(n)
        return a, p, n

    def __len__(self):
        return len(self.triplets)

def train_one_epoch(model, loader, triplet_loss, center_loss, optimizer, lambda_c=0.01):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for a, p, n in tqdm(loader):
        a, p, n = a.to(device), p.to(device), n.to(device)
        emb_a, emb_p, emb_n = model(a), model(p), model(n)
        labels = torch.arange(a.size(0)).to(device)

        loss_tri = triplet_loss(emb_a, emb_p, emb_n)
        loss_cent = center_loss(emb_a, labels)
        norm_penalty = 1e-4 * (emb_a.norm(2) + emb_p.norm(2) + emb_n.norm(2))
        loss = loss_tri + lambda_c * loss_cent + norm_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (F.pairwise_distance(emb_a, emb_p) < F.pairwise_distance(emb_a, emb_n)).sum().item()
        total += a.size(0)

    return total_loss / len(loader), correct / total

def validate(model, loader, triplet_loss):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for a, p, n in tqdm(loader):
            a, p, n = a.to(device), p.to(device), n.to(device)
            emb_a, emb_p, emb_n = model(a), model(p), model(n)
            loss = triplet_loss(emb_a, emb_p, emb_n)
            total_loss += loss.item()
            correct += (F.pairwise_distance(emb_a, emb_p) < F.pairwise_distance(emb_a, emb_n)).sum().item()
            total += a.size(0)
    return total_loss / len(loader), correct / total


def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.grid(True); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.grid(True); plt.legend()

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()

# ====== MAIN ======
def main():
    train_dir = "/content/dataset/dataset_5/train"
    val_dir = "/content/dataset/dataset_5/val"

    batch_size = 32
    epochs = 100
    margin = 0.6
    embedding_size = 128
    lr = 5e-4
    patience = 5
    no_improve_epochs = 0

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.2),
        transforms.RandomAffine(20, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2),  
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_ds = TripletDataset(datasets.ImageFolder(train_dir), transform)
    val_ds = TripletDataset(datasets.ImageFolder(val_dir), transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = FaceNet(embedding_size=embedding_size).to(device)
    triplet_loss = TripletMarginLoss(margin=margin, p=2)
    center_loss = CenterLoss(num_classes=len(train_ds.labels), feat_dim=embedding_size).to(device)

    optimizer = AdamW(list(model.parameters()) + list(center_loss.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_acc = 0

    for epoch in range(epochs):
        triplet_loss.margin = max(0.2, 0.6 * (0.93 ** epoch))  
        print(f"\n📦 Epoch {epoch+1}/{epochs} | Margin: {triplet_loss.margin:.3f}")

        train_loss, train_acc = train_one_epoch(model, train_loader, triplet_loss, center_loss, optimizer)
        val_loss, val_acc = validate(model, val_loader, triplet_loss)
        scheduler.step()

        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_acc); val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_facenet_model.pth")
            print(f"✅ Model saved. Best Val Acc: {best_acc:.4f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"⏳ No improvement: {no_improve_epochs}/{patience}")
            if no_improve_epochs >= patience:
                print("🚓 Early stopping triggered.")
                break

    print("🎉 Training completed.")
    plot_metrics(train_losses, val_losses, train_accs, val_accs)

if __name__ == "__main__":
    main()
