from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

# Eğitim verisi: Basit olumlu/olumsuz cümleler
X = [
    "Bu harika bir ürün!", 
    "Hiç beğenmedim çok kötüydü", 
    "Gerçekten mükemmel bir deneyimdi", 
    "Berbat, iade ettim", 
    "Çok memnun kaldım, teşekkürler"
]
y = [1, 0, 1, 0, 1]  # 1 = olumlu, 0 = olumsuz

# Model oluştur ve eğit
model = make_pipeline(TfidfVectorizer(), LogisticRegression())
model.fit(X, y)

# Modeli kaydet
joblib.dump(model, "best_model.pkl")
print("✅ Model kaydedildi: best_model.pkl")
