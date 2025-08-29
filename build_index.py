from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1) PDF'i yükle
loader = PyPDFLoader("data/nutuk.pdf")
documents = loader.load()

# 2) Metni parçalara ayır
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# 3) Embedding modeli
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4) FAISS index oluştur
db = FAISS.from_documents(docs, embeddings)
db.save_local("nutuk_faiss_index")

print("✅ nutuk_faiss_index başarıyla oluşturuldu!")
