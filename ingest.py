from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# PDF dosyasını yükle
pdf_path = "data/nutuk.pdf"  # PDF dosyanın yolu
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# Embeddings oluştur
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Chroma vektör veritabanını oluştur
vectordb = Chroma.from_documents(docs, embeddings, persist_directory="chroma_db")

print("PDF başarıyla işlendi ve vektör veritabanı oluşturuldu.")
