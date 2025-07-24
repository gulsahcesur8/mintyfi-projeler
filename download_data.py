import requests
import zipfile
import io
import os

# URL'den veri setini indir
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
response = requests.get(url)

# Zip dosyasını bellekte aç
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    zip_ref.extractall("data")  # "data" klasörüne çıkar
    print("Veri başarıyla indirildi ve çıkarıldı.")
