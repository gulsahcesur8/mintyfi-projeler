# Fine-Tuning Araştırması ve Örnek Uygulama (TinyLlama & DistilGPT-2)

Bu çalışmada, düşük parametreli dil modelleri (TinyLlama-1.1B-Chat-v1.0-q4 ve DistilGPT-2) üzerinde fine-tuning (yeniden eğitim) sürecini inceledim. Amaç, hem teorik olarak süreci anlamak hem de pratikte küçük bir model üzerinde uygulama yapabilecek bir temel oluşturmaktı.

## Fine-Tuning Nedir?
Fine-tuning, önceden eğitilmiş bir modeli küçük ve amaca uygun bir veri kümesi üzerinde yeniden eğiterek, belirli bir göreve uyarlama yöntemidir.  

Avantajları:
- Daha az veriyle yüksek doğruluk elde edebilme
- Eğitim süresini kısaltma
- Daha düşük maliyet

Dezavantajları:
- Aşırı uyum (overfitting) riski
- Veri seti uyumsuzluğu (domain shift)

## Neden Düşük Parametreli Model?
Benim çalışmamda TinyLlama ve DistilGPT-2 gibi küçük boyutlu modelleri tercih ettim.  

Bunun sebepleri:
- Daha hızlı eğitim yapabilmek
- Daha az GPU belleği kullanmak
- Deneme-yanılma sürecini hızlandırmak

| Model | Parametre Sayısı | Kullanım Alanı |
|-------|-----------------|----------------|
| TinyLlama-1.1B-Chat-v1.0-q4 | ~1.1B (quantized) | Sohbet tabanlı görevler |
| DistilGPT-2               | ~82M            | Metin üretimi |

## PEFT ve LoRA
Araştırmam sırasında Parameter-Efficient Fine-Tuning (PEFT) ve LoRA tekniklerini de inceledim.  

Kısaca:
- **PEFT**, tüm modeli değil, küçük ek katmanları eğiterek hafızadan ve zamandan tasarruf sağlar.
- **LoRA**, modelin belirli kısımlarına düşük dereceli adaptörler ekleyerek hızlı ve verimli eğitim imkanı verir.

## Proje Yapısı (Planlanan)
```
fine-tuning-research/
│
├── README.md          # Bu dosya
├── research.md        # Daha detaylı araştırma notları
├── train.py           # DistilGPT-2 + LoRA ile fine-tuning kodu
├── inference.py       # Eğitim sonrası test kodu
├── requirements.txt   # Gerekli Python kütüphaneleri
└── data/
    └── sample.txt     # Örnek küçük veri seti
```

## Sonraki Adımlar
- `train.py` ve `inference.py` dosyalarını eklemek
- Küçük bir veri seti ile DistilGPT-2 üzerinde LoRA fine-tuning yapmak
- Sonuçları değerlendirmek

---

Kaynaklar:
- Hugging Face Transformers dokümantasyonu
- TinyLlama ve DistilGPT-2 model kartları
- PEFT ve LoRA üzerine yapılan açıklamalar
