import mlflow
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

import mlflow.sklearn

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Küçük örnek veri seti (veri indirimi yok, çok hızlı çalışır)
X_train = ["uzayda yaşam var", "beyin tümörü tedavisi", "nasa marsa gitti", "kanser tedavisi", "güneş sistemi"]
y_train = [1, 0, 1, 0, 1]

X_test = ["uzayda yaşam", "tedavi hastalık"]
y_test = [1, 0]


# Parametre kombinasyonları
param_list = [
    {"ngram_range": (1, 1), "C": 1.0, "max_features": 1000},
    {"ngram_range": (1, 2), "C": 0.5, "max_features": 1500},
    {"ngram_range": (1, 2), "C": 2.0, "max_features": 2000}
]

best_acc = 0
best_run_id = ""

for params in param_list:
    with mlflow.start_run() as run:
        vectorizer = CountVectorizer(ngram_range=params["ngram_range"], max_features=params["max_features"])
        clf = LogisticRegression(C=params["C"], max_iter=200)

        pipe = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', clf)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Parametre ve metrikleri kaydet
        mlflow.log_param("ngram_range", params["ngram_range"])
        mlflow.log_param("C", params["C"])
        mlflow.log_param("max_features", params["max_features"])
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(pipe, "model")

        print(f"Run ID: {run.info.run_id} - Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_run_id = run.info.run_id

# En iyi run_id'yi best_run.txt'e yaz
with open("best_run.txt", "w") as f:
    f.write(best_run_id)
# En iyi run_id'yi best_run.txt'e yaz
with open("best_run.txt", "w", encoding="utf-8") as f:
    f.write(best_run_id)

# En iyi run linkini exp_notes.md'ye yaz
best_run_link = f"http://localhost:5000/#/experiments/0/runs/{best_run_id}"

with open("exp_notes.md", "w", encoding="utf-8") as f:
    f.write(f"En iyi run linki:\n{best_run_link}\n")
