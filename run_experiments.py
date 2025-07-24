import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score

# Veri setini al
data = fetch_20newsgroups(subset='all', categories=['rec.autos', 'sci.med'])
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Parametre setleri
param_grid = [
    {'ngram_range': (1, 1), 'C': 1.0, 'max_features': 1000},
    {'ngram_range': (1, 2), 'C': 0.1, 'max_features': 2000},
    {'ngram_range': (1, 3), 'C': 10.0, 'max_features': 3000},
]

best_acc = 0
best_run_id = None

for params in param_grid:
    with mlflow.start_run():
        vectorizer = CountVectorizer(ngram_range=params['ngram_range'], max_features=params['max_features'])
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        clf = LogisticRegression(C=params['C'], max_iter=1000)
        clf.fit(X_train_vec, y_train)
        preds = clf.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)

        print(f"Params: {params}, Accuracy: {acc}")

        if acc > best_acc:
            best_acc = acc
            best_run_id = mlflow.active_run().info.run_id

# En iyi run_id'yi dosyaya yaz
with open("best_run.txt", "w") as f:
    f.write(best_run_id)

print(f"Best run id: {best_run_id}")
