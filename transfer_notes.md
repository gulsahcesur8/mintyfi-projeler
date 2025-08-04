## Transfer Learning Notları

**Model**: ResNet18 (pretrained=True)

**Girdi boyutu**: (1, 3, 224, 224)  
**Çıktı boyutu**: (1, 512)

**Gözlemler**:
- ResNet18, ImageNet üzerinde eğitilmiş bir modeldir.
- Son katman çıkarıldığında 512 boyutlu bir embedding üretir.
- Bu embedding, görsel özellikleri (kenarlar, şekiller, dokular) temsil eder.
- Kendi sınıflandırma ya da benzerlik görevlerinde kullanılabilir.
