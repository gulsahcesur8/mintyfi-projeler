import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Tüm veriyle eğitim ve tahmin
model = LogisticRegression(max_iter=1000)
model.fit(X, y)
y_pred = model.predict(X)

acc = accuracy_score(y, y_pred)
print(f"Accuracy on full data: {acc:.2f}")
print("\nClassification Report on full data:")
print(classification_report(y, y_pred))
