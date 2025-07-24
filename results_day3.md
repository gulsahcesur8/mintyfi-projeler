# Gün 3 Sonuçları

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| LogisticRegression | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| MultinomialNB | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| RandomForest | 0.5000 | 0.2500 | 0.5000 | 0.3333 |

En iyi model: **LogisticRegression**

Model parametreleri:

- LogisticRegression: LogisticRegression(max_iter=200)
## Kısa Yorum

Logistic Regression ve Multinomial Naive Bayes modelleri, elimizdeki küçük ve dengeli veri seti üzerinde mükemmel performans göstermiştir. Random Forest modeli ise düşük performans sergilemiştir. En iyi sonuç Logistic Regression modelinden alınmıştır.  
Veri seti küçük olduğundan gerçek dünya verilerinde performans farklı olabilir, bu nedenle veri setinin büyütülmesi ve modellerin parametre optimizasyonu önerilir.

