import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split

# Görselleri tensöre çevir + normalize et
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# MNIST verisini indir ve yükle
full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# İlk 1000 örneği seç
subset_indices = list(range(1000))
subset = Subset(full_train, subset_indices)

# 800 eğitim, 200 valid olarak ayır
train_set, valid_set = random_split(subset, [800, 200], generator=torch.Generator().manual_seed(42))

# DataLoader'lar
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=64, shuffle=False)

# Test için ekrana yaz
print(f"Eğitim örnek sayısı: {len(train_set)}")
print(f"Valid örnek sayısı: {len(valid_set)}")
