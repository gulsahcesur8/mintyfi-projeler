import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# 1. Veriyi yükle
df = pd.read_csv("dataset.csv")

# 2. Train-valid split (%80-%20, stratified)
X_train, X_valid, y_train, y_valid = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)
# 3. TF-IDF ile metinleri sayısallaştır
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_valid_tfidf = tfidf.transform(X_valid)

# 4. Modelleri tanımla
models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "MultinomialNB": MultinomialNB(),
    "RandomForest": RandomForestClassifier(random_state=42),
}

# 5. Modelleri eğit ve değerlendir
results = []

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_valid_tfidf)
    
    acc = accuracy_score(y_valid, preds)
    prec = precision_score(y_valid, preds, average="weighted", zero_division=0)
    rec = recall_score(y_valid, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_valid, preds, average="weighted", zero_division=0)
    cm = confusion_matrix(y_valid, preds)
    
    print(f"Model: {name}")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("-" * 30)
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ConfusionMatrix": cm
    })

# 6. En iyi modeli bul (F1 skoru bazlı)
best_model = max(results, key=lambda x: x["F1"])
print(f"En iyi model: {best_model['Model']}")

# 7. Sonuçları results_day3.md dosyasına yazma (basit tablo ve yorum)

with open("results_day3.md", "w", encoding="utf-8") as f:
    f.write("# Gün 3 Sonuçları\n\n")
    f.write("| Model | Accuracy | Precision | Recall | F1 Score |\n")
    f.write("|-------|----------|-----------|--------|----------|\n")
    for r in results:
        f.write(f"| {r['Model']} | {r['Accuracy']:.4f} | {r['Precision']:.4f} | {r['Recall']:.4f} | {r['F1']:.4f} |\n")
    f.write("\n")
    f.write(f"En iyi model: **{best_model['Model']}**\n")
    f.write("\n")
    f.write("Model parametreleri:\n\n")
    for name, model in models.items():
        if name == best_model["Model"]:
            f.write(f"- {name}: {model}\n")

print("Sonuçlar results_day3.md dosyasına yazıldı.")
