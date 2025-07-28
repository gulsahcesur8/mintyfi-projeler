import requests

url = "http://127.0.0.1:8000/predict"

test_texts = [
    "çok kötüydü",
    "mükemmel bir yapım",
    "orta karar bir film"
]

for text in test_texts:
    data = {"text": text}
    response = requests.post(url, json=data)
    print(f"Input: {text}")
    print(f"Response: {response.json()}\n")
