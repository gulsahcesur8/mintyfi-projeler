from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import pipeline

# PDF yükleme
loader = PyPDFLoader("Nutuk.pdf")
docs = loader.load()

# Vektör oluşturma
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")

# Text generation pipeline (offline)
text_gen = pipeline(
    "text-generation",
    model="bigscience/bloom-560m",
    device=-1,  # CPU kullanımı
    max_new_tokens=256
)

# Soru-cevap fonksiyonu
def ask_question(question):
    results = vectordb.similarity_search(question, k=3)
    context = " ".join([r.page_content for r in results])
    prompt = f"Context: {context}\n\nSoru: {question}\nCevap:"
    answer = text_gen(prompt)[0]['generated_text']
    return answer

# Soru sorma döngüsü
while True:
    q = input("Sorunuz (çıkmak için 'q'): ")
    if q.lower() == "q":
        break
    print("Cevap:", ask_question(q))
