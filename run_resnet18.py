import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

def main():
    # 1. Modeli indir ve hazırla
    model = models.resnet18(pretrained=True)
    model.eval()

    # 2. Son katmanı çıkar (embedding modeli oluştur)
    embedding_model = torch.nn.Sequential(*list(model.children())[:-1])

    # 3. Test görüntüsünü indir
    url = "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")

    # 4. Görüntü ön işleme
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(img).unsqueeze(0)  # (1, 3, 224, 224)

    # 5. Embedding çıkar
    with torch.no_grad():
        out = embedding_model(input_tensor)       # (1, 512, 1, 1)
        embedding = out.view(out.size(0), -1)     # (1, 512)

    # 6. Sonuçları göster
    print("Input tensor shape:", input_tensor.shape)
    print("Embedding shape:", embedding.shape)

    # 7. transfer_notes.md dosyasını yaz
    with open("transfer_notes.md", "w", encoding="utf-8") as f:
        f.write("## Transfer Learning Notları\n\n")
        f.write("**Model**: ResNet18 (pretrained=True)\n\n")
        f.write(f"**Girdi boyutu**: {tuple(input_tensor.shape)}  \n")
        f.write(f"**Çıktı boyutu**: {tuple(embedding.shape)}\n\n")
        f.write("**Gözlemler**:\n")
        f.write("- ResNet18, ImageNet üzerinde eğitilmiş bir modeldir.\n")
        f.write("- Son katman çıkarıldığında 512 boyutlu bir embedding üretir.\n")
        f.write("- Bu embedding, görsel özellikleri (kenarlar, şekiller, dokular) temsil eder.\n")
        f.write("- Kendi sınıflandırma ya da benzerlik görevlerinde kullanılabilir.\n")

if __name__ == "__main__":
    main()
