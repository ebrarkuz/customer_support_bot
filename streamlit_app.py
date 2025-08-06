import streamlit as st
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import tempfile
from langchain.memory import ConversationBufferMemory
from google.api_core.client_options import ClientOptions

# buraları gpt ekledi 
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage 

load_dotenv()


# Gemini API key 
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY bulunamadı")
os.environ["GOOGLE_API_KEY"] = api_key

# Streamlit kısmı 
st.set_page_config(page_title="Müşteri Destek Botu") # ikon ekle bi ara 

st.title("PDF Destek Botu (RAG + Memory)")
st.write("Bir PDF yükleyin, içeriğine dair sorular sorun. Türkçe desteklidir.")

# PDF yükleme 
uploaded_file = st.file_uploader("PDF Dosyanızı yükleyin", type="pdf", key="pdf_uploader")


if uploaded_file is not None:
    if "last_uploaded_name" not in st.session_state or uploaded_file.name != st.session_state.last_uploaded_name:
        
        with st.spinner("PDF işleniyor..."):
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name 
            
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

            # Metinleri parçalara yani chunk'lara böl
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = splitter.split_documents(documents)

            # Gemini embedding ile metin vektörleştirme
            embedding = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                transport='grpc' 
            )
            
            # vektör veri tabanı
            vectordb = FAISS.from_documents(docs, embedding)
            
            # LLM oluşturma
            llm = GoogleGenerativeAI(
                model="gemini-1.5-flash", 
                temperature=0,
                transport='grpc' 
            )

            #bu kısmı da gpt ekledi 
            
            # Sohbet geçmişi için memory oluşturma (Streamlit session_state ile uyumlu)
            # LangChain'in yeni zincir API'sinde memory doğrudan zincire verilmez,
            # MessagesPlaceholder ile prompt'a dahil edilir.
            # st.session_state.chat_history, bu placeholder için kullanılacak.

            # 1. Aşama: Sohbet geçmişine duyarlı retriever oluşturma
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "Aşağıdaki sohbet geçmişi ve yeni bir soru göz önüne alındığında, retriever'ın araması için standalone bir soru oluşturun. Sohbet geçmişi yoksa, soruyu olduğu gibi kullanın."),
                    MessagesPlaceholder(variable_name="chat_history"), # Sohbet geçmişi buraya gelecek
                    ("human", "{input}"),
                ]
            )
            
            history_aware_retriever = create_history_aware_retriever(
                llm, vectordb.as_retriever(search_kwargs={"k": 3}), contextualize_q_prompt
            )

            # 2. Aşama: Cevap zincirini oluşturma (belgeleri kullanarak)
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "Aşağıdaki bağlamı kullanarak soruyu cevaplayın. Eğer cevap bağlamda yoksa, 'Bilmiyorum.' deyin.\n\n{context}"),
                    MessagesPlaceholder(variable_name="chat_history"), # Sohbet geçmişi buraya gelecek
                    ("human", "{input}"),
                ]
            )

            document_chain = create_stuff_documents_chain(llm, qa_prompt)

            # 3. Aşama: İki aşamayı birleştirerek tam zinciri oluşturma
            qa_chain = create_retrieval_chain(history_aware_retriever, document_chain)

            st.session_state.qa_chain = qa_chain
            st.session_state.chat_history = [] # Sohbet geçmişini sıfırla
            st.session_state.last_uploaded_name = uploaded_file.name # Aynı dosyanın yeniden işlenmesini engellemek için

        st.success("PDF başarıyla işlendi!")

# Sohbet geçmişini görüntüleme
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Kullanıcı girişi
user_query = st.chat_input("Sorunuzu buraya yazın...")

if user_query:
    if "qa_chain" not in st.session_state:
        st.warning("Lütfen önce bir PDF dosyası yükleyin.")
    else:
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.spinner("Cevap aranıyor..."):
            # LangChain zincirini çağır
            # Yeni zincir API'sinde invoke metodu kullanılır
            response = st.session_state.qa_chain.invoke(
                {"input": user_query, "chat_history": st.session_state.chat_history}
            )
            
            ai_response = response["answer"]
            with st.chat_message("assistant"):
                st.markdown(ai_response)
            st.session_state.chat_history.append(AIMessage(content=ai_response))

