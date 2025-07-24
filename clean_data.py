import pandas as pd

# Veriyi oku
df = pd.read_csv("data/SMSSpamCollection", sep="\t", header=None, names=["label", "message"])

# 1. Eksik değer var mı?
print("Eksik değer sayısı:\n", df.isnull().sum())

# 2. Yinelenen kayıt sayısı
print("\nYinelenen kayıt sayısı:", df.duplicated().sum())

# Yinelenenleri kaldır
df = df.drop_duplicates()

# 3. Encoding sorunu var mı? (örnek amaçlı ilk bozuk karakterleri bulmaya çalışalım)
# Garip karakter varsa görürüz
print("\nRastgele 10 mesaj:")
print(df["message"].sample(10, random_state=42))

# 4. Temiz veri boyutu
print("\nTemizlenmiş veri boyutu:", df.shape)

# 5. Temiz veriyi kaydet
df.to_csv("sms_clean.csv", index=False)

print("\nTemiz veri sms_clean.csv olarak kaydedildi.")
