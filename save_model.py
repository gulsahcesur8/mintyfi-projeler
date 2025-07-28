from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Eğitim verisi (örnek olumlu/olumsuz yorumlar)
texts = [
    "çok güzel film", "berbattı", "harikaydı", 
    "çok kötü", "mükemmel", "iğrenç", "bayıldım"
]
labels = [1, 0, 1, 0, 1, 0, 1]  # 1: olumlu, 0: olumsuz

# Metni sayısal hale getir
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Lojistik Regresyon Modeli eğit
model = LogisticRegression()
model.fit(X, labels)

# Model ve vectorizer'ı kaydet
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model ve vectorizer kaydedildi!")
