import torch
import torch.nn as nn
import torch.optim as optim
from mlp_model import MLP  # Model tanımı dışarıdaysa
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import matplotlib.pyplot as plt
import json

# --- VERİYİ HAZIRLA (önceki adımlardan) ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
subset = Subset(full_train, list(range(1000)))
train_set, valid_set = random_split(subset, [800, 200], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=64, shuffle=False)

# --- MODEL, LOSS, OPTIMIZER ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- EĞİTİM VE VALIDASYON DÖNGÜSÜ ---
train_losses = []
valid_losses = []

for epoch in range(5):
    model.train()
    total_train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    train_losses.append(total_train_loss / len(train_loader))

    model.eval()
    total_valid_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_valid_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    valid_losses.append(total_valid_loss / len(valid_loader))
    acc = 100 * correct / total

    print(f"Epoch {epoch+1}/5 | Train Loss: {train_losses[-1]:.4f} | Valid Loss: {valid_losses[-1]:.4f} | Valid Acc: {acc:.2f}%")

# --- GRAFİĞİ KAYDET (mlp_loss.png) ---
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('MLP Loss')
plt.savefig("mlp_loss.png")
plt.close()

# --- DOĞRULUK METRİKLERİ (mlp_metrics.json) ---
metrics = {
    "accuracy": round(acc, 2),
}
with open("mlp_metrics.json", "w") as f:
    json.dump(metrics, f)

print("Eğitim tamamlandı. mlp_loss.png ve mlp_metrics.json dosyaları kaydedildi.")
