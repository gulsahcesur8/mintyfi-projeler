import sys
import joblib

# Komut satırından argüman al (mesaj metni)
if len(sys.argv) < 2:
    print("⚠️ Lütfen tahmin edilecek metni tırnak içinde girin.")
    print("Örnek: python predict.py \"bu ürünü çok beğendim\"")
    sys.exit()

message = sys.argv[1]

# Kaydedilmiş modeli yükle
model = joblib.load("best_model.pkl")

# Tahmin yap
prediction = model.predict([message])[0]

# Sonucu yazdır
etiket = "Olumlu" if prediction == 1 else "Olumsuz"
print(f"🔍 Metin: \"{message}\"\n📊 Tahmin: {etiket}")
