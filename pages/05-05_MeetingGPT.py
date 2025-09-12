from operator import itemgetter
import streamlit as st
import subprocess
import math
from pydub import AudioSegment
import openai
import glob
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.output_parser import StrOutputParser
from langchain.storage import LocalFileStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import CacheBackedEmbeddings
from langchain.prompts import MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler


class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self, *args, **kwargs):
        self.message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


st.set_page_config(
    page_title="MeetingGPT",
    page_icon="🤝",
)

st.title("MeetingGPT")
st.markdown(
    "MeetingGPT에 오신 것을 환영합니다. 사이드바에서 동영상을 업로드하면 대화의 요약과 대화에 대한 질문을 할 수 있는 챗봇을 제공해 드립니다."
)

llm = ChatOpenAI(
    temperature=0.1,
)

streaming_llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=2400,
    chunk_overlap=300,
)


@st.cache_data(show_spinner="Embedding..")
def embed_file(file_name):
    file_path = f"./.cache/meeting_files/{file_name}"
    cache_dir = LocalFileStore(f"./.cache/meeting_embeddings/{file_name}")
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vectorstore = FAISS.from_documents(docs, cached_embeddings)

    retriever = vectorstore.as_retriever()

    return retriever


@st.cache_data(show_spinner=False)
def extract_audio_from_video(video_path):
    audio_path = (
        video_path.replace(".mp4", ".mp3")
        .replace(".avi", ".mp3")
        .replace(".mkv", ".mp3")
        .replace(".mov", ".mp3")
    )
    if os.path.exists(audio_path):
        return
    command = ["ffmpeg", "-y", "-i", video_path, "-vn", audio_path]
    subprocess.run(command)


@st.cache_data(show_spinner=False)
def cut_audio_in_chunks(video_name, audio_path, chunk_size, chunks_folder):
    if os.path.exists(f"./.cache/chunks/{video_name}/00_chunk.mp3"):
        return
    chunk_len = chunk_size * 60 * 1000
    track = AudioSegment.from_mp3(audio_path)
    chunks = math.ceil(len(track) / chunk_len)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        chunk = track[start_time:end_time]
        path = os.path.join("./.cache/chunks/", video_name)
        os.mkdir(path)
        chunk.export(
            f"{chunks_folder}/{str(i).zfill(2)}_chunk.mp3",
            format="mp3",
        )


@st.cache_data(show_spinner=False)
def transcribe_chunks(chunks_folder, destination):
    if os.path.exists(destination):
        return
    files = glob.glob(f"{chunks_folder}/*.mp3")
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transcript = openai.Audio.transcribe(
                "whisper-1",
                audio_file,
            )
            text_file.write(transcript["text"])


@st.cache_data(show_spinner=False)
def process_video(video):
    st.write("영상 불러오는 중..")
    video_content = video.read()
    video_path = f"./.cache/meeting_files/{video.name}"
    audio_path = (
        video_path.replace(".mp4", ".mp3")
        .replace(".avi", ".mp3")
        .replace(".mkv", ".mp3")
        .replace(".mov", ".mp3")
    )
    transcript_path = (
        video_path.replace(".mp4", ".txt")
        .replace(".avi", ".txt")
        .replace(".mkv", ".txt")
        .replace(".mov", ".txt")
    )
    with open(video_path, "wb") as f:
        f.write(video_content)
    st.write("소리 추출 중..")
    extract_audio_from_video(video_path)
    st.write("소리 분할 중..")
    cut_audio_in_chunks(video.name, audio_path, 10, chunks_folder)
    st.write("대본 추출 중..")
    transcribe_chunks(
        chunks_folder,
        transcript_path,
    )
    return transcript_path


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def save_memory(input, output):
    st.session_state["memory"].save_context({"input": input}, {"output": output})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def invoke_chain(question):
    result = chain.invoke(question)
    save_memory(question, result.content)


@st.cache_data(show_spinner=False)
def generate_summary(transcript_path):
    loader = TextLoader(transcript_path)
    docs = loader.load_and_split(text_splitter=splitter)

    first_summary_prompt = ChatPromptTemplate.from_template(
        """
        당신은 문서 요약 전문가입니다.
        다음 문서를 정확하게 요약하세요.
        
        {text}
    """
    )

    progress_text = "요약본 생성하는 중.."

    first_summary_chain = first_summary_prompt | llm | StrOutputParser()

    my_bar = st.progress(0, text=f"{progress_text} (0/{len(docs)})")

    summary = first_summary_chain.invoke({"text": docs[0].page_content})

    my_bar.progress(1 / len(docs), text=f"{progress_text} (1/{len(docs)})")

    refine_prompt = ChatPromptTemplate.from_template(
        """
        당신은 요약본을 개선하는 일의 전문가입니다.
        이전 요약본을 새로운 정보를 보고 개선이 필요하다면 (예: 새로운 정보의 출현, 기존 정보에 대한 자세한 설명 등) 개선하세요. 
        이전 요약본:
        -----------------
        {existing_summary}
        -----------------
        새로운 정보:
        -----------------
        {context}
        -----------------
        만약 개선할 점이 없다면 이전 요약본을 반환하세요.
    """
    )

    refine_chain = refine_prompt | llm | StrOutputParser()
    for i, doc in enumerate(docs[1:]):
        summary = refine_chain.invoke(
            {
                "existing_summary": summary,
                "context": doc.page_content,
            }
        )
        my_bar.progress((i + 2) / len(docs), f"{progress_text} ({i+2}/{len(docs)})")
    return summary


with st.sidebar:
    video = st.file_uploader(
        "Video",
        type=["mp4", "avi", "mkv", "mov"],
    )

if video:
    chunks_folder = f"./.cache/chunks/{video.name}"
    with st.status("영상 처리 중.."):
        transcript_path = process_video(video)
    transcript_tab, summary_tab, chat_tab = st.tabs(["대본", "요약", "대화"])

    with transcript_tab:
        with open(transcript_path, "r") as f:
            st.write(f.read())
    with summary_tab:
        start = st.button("요약 생성하기")
        if start or st.session_state["isSummaryGenerated"]:
            summary = generate_summary(transcript_path)
            st.write(summary)
            st.session_state["isSummaryGenerated"] = True
    with chat_tab:
        retriever = embed_file(transcript_path.split("/")[-1])
        query = st.chat_input("회의에서 일어난 궁금한 일들을 물어보세요.")
        paint_history()
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
당신은 회의의 대본 기록을 이용해 사용자의 질문에 답변하는 전문적인 AI입니다.
당신이 기존에 알고 있던 지식을 사용하지 마세요.
모르는 내용이라면 모른다고 하고 지어내지 마세요.
사용자의 질문은 보통 회의에 대해 질문 하는 것이니 당신의 생각을 이야기 하지 마세요.
------대본------
{context}
---------------
""",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )
        if query:
            send_message(query, "human")
            chain = (
                {
                    "context": retriever,
                    "question": RunnablePassthrough(),
                    "history": RunnableLambda(
                        st.session_state.memory.load_memory_variables
                    )
                    | itemgetter("history"),
                }
                | chat_prompt
                | streaming_llm
            )
            with st.chat_message("ai"):
                invoke_chain(query)
else:
    st.session_state["messages"] = []
    st.session_state["memory"] = ConversationBufferMemory(
        llm=llm, max_token_limit=1000, return_messages=True
    )
    st.session_state["isSummaryGenerated"] = False