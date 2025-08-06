from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

print("Kod çalışmaya başladı...")

load_dotenv()
print(".env dosyası yüklendi.")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not found")

# Langchain'in Gemini entegrasyonu için GOOGLE_API_KEY ortam değişkenini ayarlayın.
os.environ["GOOGLE_API_KEY"] = api_key
print("API anahtarı ortam değişkenine ayarlandı.")

print("PDF dosyası yükleniyor...")
try:
    loader = PyPDFLoader("musteri_destek_faq.pdf")
    documents = loader.load()
    print(f"PDF dosyası başarıyla yüklendi. Sayfa sayısı: {len(documents)}")
except Exception as e:
    print(f"HATA: PDF yükleme sırasında bir sorun oluştu: {e}")
    # Bu satır, PDF dosyasının bulunamaması gibi hataları yakalamamızı sağlar.
    exit() # Hata oluşursa programı durdurur.

# ... (kodun geri kalanı)
print("Metinler parçalara ayrılıyor...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)
print(f"Toplam {len(docs)} adet metin parçası oluşturuldu.")

print("Embedding modeli oluşturuluyor...")
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

print("Vektör veritabanı oluşturuluyor ve kaydediliyor...")
vectordb = FAISS.from_documents(docs, embedding)
vectordb.save_local("faq_vectorstore")

print("Embedding ve vektör veri tabanı başarılı bir şekilde oluşturuldu")