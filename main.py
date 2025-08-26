import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_ratio = 0.9
train_ds = datasets.ImageFolder("data/cats_dogs/train", transform=transform_train)
val_ds   = datasets.ImageFolder("data/cats_dogs/val",   transform=transform_val)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet18(pretrained=True)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(3):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        logits = model(imgs)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * imgs.size(0)
        preds = (logits.sigmoid() > 0.5).long()
        correct += (preds.cpu() == labels.long().cpu()).sum().item()
        total += imgs.size(0)

    train_acc = correct / total
    train_loss = loss_sum / total

    model.eval()
    v_total, v_correct, v_loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            v_loss_sum += loss.item() * imgs.size(0)
            preds = (logits.sigmoid() > 0.5).long()
            v_correct += (preds.cpu() == labels.long().cpu()).sum().item()
            v_total += imgs.size(0)

    val_acc = v_correct / v_total
    val_loss = v_loss_sum / v_total

    print(f"Epoch {epoch+1}: "
          f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
          f"val_loss={val_loss:.4f} acc={val_acc:.3f}")

torch.save(model.state_dict(), "resnet18_cats_dogs.pth")
print("save resnet18_cats_dogs.pth")
