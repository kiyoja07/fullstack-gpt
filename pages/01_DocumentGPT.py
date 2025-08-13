from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty() # LLM 응답 시작 시 빈 메시지 박스 생성

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai") # LLM 응답 종료 시 메시지 저장

    # 토큰 단위 실시간 출력 -> 사용자 경험 향상
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token # 새로운 토큰이 생성될 때마다 메시지 박스 업데이트
        self.message_box.markdown(self.message)  # 메시지 박스에 업데이트된 현재 메시지 표시

llm = ChatOpenAI(
    temperature=0.1, # 낮은 온도 → 일관된 답변
    streaming=True, # 실시간 스트리밍 활성화
    callbacks=[
        ChatCallbackHandler(), # 커스텀 콜백 연결
    ],
)


@st.cache_data(show_spinner="Embedding file...") # 동일한 파일 재업로드 시 캐시 사용
def embed_file(file):
    # 1. 파일 저장
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    # 2. 캐시 디렉토리 설정
    # 각 파일마다 별도의 캐시 디렉토리 생성 -> 파일 변경 시에만 재처리
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    # 3. 텍스트 분할
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,      # 청크당 600 토큰
        chunk_overlap=100,   # 100 토큰 중복
    )
    # 4. 문서 로드 및 분할
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    # 5. 임베딩 생성 (캐시 포함)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    # 6. FAISS 벡터스토어 생성
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False, # 중복 저장 방지
        )

# 문서 포맷팅 : 검색된 여러 문서를 하나의 문자열로 결합
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human") # 사용자 메시지를 박스에 표시하고 세션 상태에 저장
        # RAG 체인 구성
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        # AI 응답 실행
        with st.chat_message("ai"):
            chain.invoke(message)


else:
    st.session_state["messages"] = [] # 파일이 없으면 메시지 초기화