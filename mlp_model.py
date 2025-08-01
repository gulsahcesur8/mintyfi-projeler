import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),           # 28x28 -> 784 (düzleştir)
            nn.Linear(784, 128),    # Girdi katmanı
            nn.ReLU(),              # Aktivasyon
            nn.Linear(128, 64),     # Gizli katman
            nn.ReLU(),
            nn.Linear(64, 10)       # Çıktı katmanı (10 sınıf)
        )

    def forward(self, x):
        return self.layers(x)
if __name__ == "__main__":
    model = MLP()
    test_input = torch.randn(1, 1, 28, 28)  # 1 örnek, 1 kanal, 28x28
    output = model(test_input)
    print("Çıktı boyutu:", output.shape)  # Beklenen: [1, 10]
