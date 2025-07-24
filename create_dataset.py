import pandas as pd

data = {
    "text": [
        "Bu ürün harika, bayıldım!",
        "Kesinlikle tavsiye ederim.",
        "Berbat bir deneyimdi.",
        "Çok kötü, hiç beğenmedim.",
        "Fena değil, idare eder.",
        "Çok güzel bir kaliteye sahip.",
        "Yine alırım, memnun kaldım.",
        "İade etmek zorunda kaldım.",
        "Kötü kargo, paket yırtıktı.",
        "Harika, hızlı teslimat!"
    ],
    "label": [
        "positive", "positive", "negative", "negative", "neutral",
        "positive", "positive", "negative", "negative", "positive"
    ]
}

df = pd.DataFrame(data)
df.to_csv("dataset.csv", index=False, encoding="utf-8")
print("dataset.csv dosyası oluşturuldu.")
