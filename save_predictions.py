import joblib
import json

# Modeli yükle
model = joblib.load("best_model.pkl")

# 5 örnek mesaj
sample_texts = [
    "Bu ürün gerçekten harika!",
    "Hiç memnun kalmadım.",
    "Fena değil ama daha iyilerini gördüm.",
    "Muhteşem bir alışverişti.",
    "Berbat bir kalite, asla tavsiye etmem."
]

# Tahminleri al
results = []
for text in sample_texts:
    pred = model.predict([text])[0]
    label = "Olumlu" if pred == 1 else "Olumsuz"
    results.append({
        "text": text,
        "prediction": label
    })

# JSON dosyasına yaz
with open("pred_samples.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("✅ Tahminler pred_samples.json dosyasına kaydedildi.")
