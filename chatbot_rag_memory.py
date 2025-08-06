from langchain.chains import ConversationalRetrievalChain  # RAG + sohbet zinciri
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory  # konuşma geçmişini tutan hafıza yapısı
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
import os




load_dotenv()  # ortam değişkenlerini .env dosyasından yükle

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not found")


os.environ["GOOGLE_API_KEY"] = api_key

# embedding 
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# vektör veritabanı yükle
vectordb = FAISS.load_local(
    "faq_vectorstore",  
    embedding, 
    allow_dangerous_deserialization=True  # pickle güvenlik uyarısını bastır
)

# memory oluşturma
memory = ConversationBufferMemory(
    memory_key="chat_history",  # konuşma geçmişi bu anahtar ile saklanır
    return_messages=True  # geçmiş mesajlar tam haliyle geri döner
)

# sıfır rastlantısallık ile çalışır, sabit cevaplar verir
llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",  # Bu kısmı gemini ile değiştirdim 
    temperature=0  # bu değeri arttırırsak daha değişken cevaplar verir
)

# rag + memory zincir oluştur
# llm
# faiss retriever: en benzer 3 belge getirilsin (k=3)
# memory

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    verbose=True
)

print("Müşteri destek botuna hoş geldiniz")

while True:
    # kullanıcı girişi
    user_input = input("Siz: ")
    if user_input.lower() == "çık":
        break

    # kullanıcı sorusu llm + rag + memory zincirine verilir
    response = qa_chain.run(user_input)
    print("Müşteri Destek Botu: ", response)
