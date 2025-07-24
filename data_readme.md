# Veri Kümesi Açıklaması

Bu veri kümesi SMS mesajlarından oluşmaktadır. Her satırda bir mesaj ve bu mesajın spam (istenmeyen) ya da ham (normal) olduğu bilgisi yer almaktadır.

## Dosya: sms_clean.csv

Veri temizleme işlemleri sonucunda elde edilen dosyadır. Aşağıdaki işlemler uygulanmıştır:

- Eksik veri kontrolü yapıldı, herhangi bir eksik değere rastlanmadı.
- Yinelenen kayıtlar kontrol edildi, toplam 403 adet tekrar eden satır tespit edilip silindi.
- Kodlama (encoding) kaynaklı bozuk karakter kontrolü yapıldı, gözle görülür bir sorun bulunmadı.

## Kolonlar

- **label**: Mesajın sınıf etiketi. İki farklı değer alır: `ham` (normal mesaj) veya `spam` (istenmeyen mesaj).
- **message**: SMS metninin kendisi.

## Sınıf Dağılımı

- ham: 4516 adet
- spam: 653 adet

Veri kümesi, veri analizi veya makine öğrenmesi uygulamaları için hazır hale getirilmiştir.
