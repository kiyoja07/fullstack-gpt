import time
import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder

st.set_page_config(
    page_title="Document GPT",
    page_icon="📄",
    layout="wide",
)

class ChatCallbackHandler(BaseCallbackHandler):

    def __init__(self):
        self.message = ""
        self.message_box = None

    def on_llm_start(self, *args, **kwargs):
        try:
            self.message = ""
            self.message_box = st.empty()
        except Exception:
            pass
        
    def on_llm_end(self, *args, **kwargs):
        try:
            save_message(self.message, "ai")
        except Exception:
            pass

    def on_llm_new_token(self, token: str, **kwargs):
        try:
            self.message += token
            if self.message_box is not None:
                self.message_box.markdown(self.message)
        except Exception:
            pass

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.1, streaming=True, callbacks=[ChatCallbackHandler()])

# 메모리를 session_state에 저장
if "memory" not in st.session_state:
    try:
        st.session_state.memory = ConversationBufferMemory(
            llm=llm,
            max_token_limit=120,
            return_messages=True,
            memory_key="history"
        )
    except Exception:
        pass

# messages도 session_state에 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# memory 변수 안전하게 할당
try:
    memory = st.session_state.memory
except Exception:
    memory = ConversationBufferMemory(
        llm=llm,
        max_token_limit=120,
        return_messages=True,
        memory_key="history"
    )
    st.session_state.memory = memory


@st.cache_data(show_spinner="Embedding file..." )
def embed_file(file):
    try:
        # OpenAI API 키 확인
        if not os.getenv("OPENAI_API_KEY"):
            st.error("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다!")
            return None
        
        # 디렉토리 생성
        os.makedirs("./.cache/files", exist_ok=True)
        os.makedirs("./.cache/embeddings", exist_ok=True)
        
        file_content = file.read()
        file_path = f"./.cache/files/{file.name}"
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
        
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        
        with st.spinner("📄 문서를 로딩하고 분할하는 중..."):
            loader = UnstructuredFileLoader(file_path)
            docs = loader.load_and_split(text_splitter=splitter)
            st.success(f"✅ {len(docs)}개의 청크로 분할 완료!")
        
        with st.spinner("🔤 임베딩을 생성하는 중..."):
            embeddings = OpenAIEmbeddings()
            cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
            vectorstore = FAISS.from_documents(docs, cached_embeddings)
            retriever = vectorstore.as_retriever()
            st.success("✅ 임베딩 생성 완료!")
        
        return retriever
        
    except Exception as e:
        st.error(f"❌ 임베딩 생성 중 오류가 발생했습니다: {str(e)}")
        return None


st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload a file on the sidebar to get started!
"""
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )
    
    # 메모리 초기화 버튼
    if st.button("🗑️ 대화 기록 초기화"):
        try:
            st.session_state.memory.clear()
            st.session_state["messages"] = []
            st.success("대화 기록이 초기화되었습니다!")
        except Exception:
            pass

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def save_message(message, role):
    st.session_state["messages"].append({"role": role, "message": message})

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)



prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that can answer questions about documents and previous conversations. 

Use both the document context and conversation history to provide accurate answers. If the user asks about something we discussed before, refer to that information.

Document context: {context}
Conversation history: {history}

Always answer in Korean and be helpful and informative."""),
    ("user", "{question}"),
])

def docs_to_context(docs):
    return "\n\n".join([doc.page_content for doc in docs])


if file:
    retriever = embed_file(file)

    if retriever:
        st.success("🎉 파일이 성공적으로 처리되었습니다!")
        send_message("I'm ready to answer your questions!", "ai", save=False)
        paint_history()

        def ask(question):
            # 메모리에서 이전 대화 로드
            try:
                memory_vars = st.session_state.memory.load_memory_variables({})
                history = memory_vars.get("history", [])
            except Exception:
                history = []
            
            # 관련 문서 검색
            docs = retriever.invoke(question)
            context = docs_to_context(docs)
            
            # 간단한 프롬프트 사용
            result = prompt.invoke({
                "question": question, 
                "context": context,
                "history": history
            })
            
            # LLM으로 답변 생성
            response = llm.invoke(result)
            
            # 메모리에 대화 저장
            try:
                st.session_state.memory.save_context({"input": question}, {"output": response.content})
            except Exception:
                pass
            
            return response.content

        message = st.chat_input("Ask me anything!")
        if message:
            send_message(message, "human")
            with st.chat_message("ai"):
                response = ask(message)
    else:
        st.warning("파일 처리를 완료할 수 없습니다. 위의 오류 메시지를 확인해주세요.")
else:
    st.session_state["messages"] = []