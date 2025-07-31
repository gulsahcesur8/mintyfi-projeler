import torch

# -------------------------
# CİHAZ SEÇİMİ (CPU ya da GPU)
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Kullanılan cihaz:", device)

# -------------------------
# 2B RASTGELE TENSÖRLER OLUŞTURMA
# -------------------------
a = torch.rand(3, 3, device=device)  # 3x3 rastgele tensör
b = torch.rand(3, 3, device=device)  # 3x3 rastgele tensör

print("\nTensör A:\n", a)
print("Tensör B:\n", b)

# -------------------------
# TEMEL TENSÖR İŞLEMLERİ
# -------------------------

# 1. Toplama
sum_result = a + b
print("\nToplam (A + B):\n", sum_result)

# 2. Matris çarpımı
matmul_result = a @ b  # ya da torch.matmul(a, b)
print("\nMatris Çarpımı (A @ B):\n", matmul_result)

# 3. Element-wise fonksiyon (örnek: sinüs)
sin_result = torch.sin(a)
print("\nElement-wise torch.sin(A):\n", sin_result)

# -------------------------
# İLERİ VE GERİ HESAPLAMA (AUTOGRAD)
# -------------------------

# requires_grad=True ile tensör oluşturuyoruz (türev hesaplamak için)
x = torch.tensor([[2.0, 3.0], [1.0, 4.0]], requires_grad=True)

# İleri hesaplama: y = x^2 + 2x
y = x**2 + 2 * x
loss = y.mean()  # ortalama değerini kayıp fonksiyonu olarak kabul ediyoruz

print("\nİleri hesaplama sonucu (y):\n", y)
print("Loss (ortalama):", loss.item())

# Geri yayılım: loss'a göre x'in türevlerini hesapla
loss.backward()

print("\nx'in gradyanı (∂loss/∂x):\n", x.grad)
