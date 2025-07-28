from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# FastAPI uygulamasını başlat
app = FastAPI()

# Veri giriş formatı
class InputText(BaseModel):
    text: str

# Uygulama başladığında modeli ve vectorizer'ı yükle
@app.on_event("startup")
def load_model():
    global model, vectorizer
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    print("✅ Model ve vectorizer yüklendi.")

# /predict endpoint'i
@app.post("/predict")
def predict(input: InputText):
    text = input.text
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0].max()
    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 4)
    }
