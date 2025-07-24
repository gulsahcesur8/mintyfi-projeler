import pandas as pd

# Dosya yolunu tanımla
file_path = "data/SMSSpamCollection"

# Veri kümesini oku (tab ayracına dikkat!)
df = pd.read_csv(file_path, sep='\t', header=None, names=["label", "message"])

# İlk 5 satırı göster
print("İlk 5 satır:")
print(df.head())

# Veri seti bilgisi
print("\nVeri seti boyutu:", df.shape)

# Kolon bilgileri
print("\nSütun isimleri:", df.columns.tolist())

# Sınıf sayımları
print("\nSınıf dağılımı:")
print(df["label"].value_counts())
