from langchain.storage import LocalFileStore
import streamlit as st
import subprocess
import math
from pydub import AudioSegment
import glob
import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings

llm = ChatOpenAI(
    temperature=0.1,
)

has_transcript = os.path.exists("./.cache/podcast.txt")

# RecursiveCharacterTextSplitter는 텍스트를 의미 단위를 가능한 한 보존하면서 잘라주는 도구이고, 검색용 문서 전처리, LLM 입력 최적화 등에 자주 쓰입니다.
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=100,
)

# 임베딩이란? 텍스트를 숫자 벡터로 변환하여 컴퓨터가 의미를 이해할 수 있게 만드는 과정
@st.cache_data()
def embed_file(file_path):
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    # retriever = vectorstore.as_retriever()
    # return retriever

    # Save the vectorstore to disk instead of returning the retriever
    vectorstore.save_local(f"./.cache/vectorstore/{os.path.basename(file_path)}")
    return file_path  # Return file path instead of retriever

# 저장된 벡터 데이터베이스를 불러와서 질문에 관련된 내용을 찾을 수 있는 검색기를 반환합니다.
def get_retriever(file_path):
    """Load the vectorstore and return retriever (not cached)"""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        f"./.cache/vectorstore/{os.path.basename(file_path)}", 
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever()

def find_cached_vectorstore(file_path):
    """Check if vectorstore exists for the given file_path"""
    return os.path.exists(f"./.cache/vectorstore/{os.path.basename(file_path)}")


# 오디오 파일을 텍스트로 변환하는 함수
@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    if has_transcript:  # 이미 대본이 있는 경우
        return
    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="ko"
            )
            text_file.write(transcript.text)


# 비디오에서 오디오를 추출하는 함수
@st.cache_data()
def extract_audio_from_video(video_path):
    if has_transcript:
        return
    audio_path = video_path.replace("mp4", "mp3")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        audio_path,
    ]
    subprocess.run(command)

# 긴 오디오 파일을 청크로 나누는 함수
@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    if has_transcript:
        return
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        chunk = track[start_time:end_time]
        chunk.export(
            f"./{chunks_folder}/chunk_{i}.mp3",
            format="mp3",
        )


st.set_page_config(
    page_title="MeetingGPT",
    page_icon="💼",
)

st.markdown(
    """
# MeetingGPT
            
Welcome to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any questions about it.

Get started by uploading a video file in the sidebar.
"""
)

with st.sidebar:
    video = st.file_uploader(
        "Video",
        type=["mp4", "avi", "mkv", "mov"],
    )

if video:
    chunks_folder = "./.cache/chunks"
    with st.status("Loading video...") as status:
        # 1. 파일 저장
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        audio_path = video_path.replace("mp4", "mp3")
        transcript_path = video_path.replace("mp4", "txt")
        with open(video_path, "wb") as f:  # 비디오 파일 저장
            f.write(video_content)
        # 2. 오디오 추출
        status.update(label="Extracting audio...")
        extract_audio_from_video(video_path)
        # 3. 오디오 분할
        status.update(label="Cutting audio segments...")
        cut_audio_in_chunks(audio_path, 10, chunks_folder)
        # 4. 오디오를 텍스트로 변환
        status.update(label="Transcribing audio...")
        transcribe_chunks(chunks_folder, transcript_path)

    transcript_tab, summary_tab, qa_tab = st.tabs(
        [
            "Transcript",
            "Summary",
            "Q&A",
        ]
    )

    with transcript_tab:
        with open(transcript_path, "r") as file:
            st.write(file.read())
    
    with summary_tab:
        start = st.button("Generate summary")

        if start:
            loader = TextLoader(transcript_path)

            docs = loader.load_and_split(text_splitter=splitter)

            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                Write a concise summary of the following:
                "{text}"
                CONCISE SUMMARY:                
            """
            )
            
            # 1. 첫 번째 요약 생성
            first_summary_chain = first_summary_prompt | llm | StrOutputParser()

            summary = first_summary_chain.invoke(
                {"text": docs[0].page_content},
            )

            refine_prompt = ChatPromptTemplate.from_template(
                """
                Your job is to produce a final summary.
                We have provided an existing summary up to a certain point: {existing_summary}
                We have the opportunity to refine the existing summary (only if needed) with some more context below.
                ------------
                {context}
                ------------
                Given the new context, refine the original summary.
                If the context isn't useful, RETURN the original summary.
                """
            )

            refine_chain = refine_prompt | llm | StrOutputParser()

            # 점진적 요약 개선
            with st.status("Summarizing...") as status:
                for i, doc in enumerate(docs[1:]):
                    status.update(label=f"Processing document {i+1}/{len(docs)-1} ")
                    summary = refine_chain.invoke(
                        {
                            "existing_summary": summary,
                            "context": doc.page_content,
                        }
                    )
                    st.write(summary)
            st.write(summary)
        
    with qa_tab:
        # retriever = embed_file(transcript_path)

        # docs = retriever.invoke("누구에 관한 내용인가요?")  # Example question in Korean

        # st.write(docs)

        # Create the embeddings if they don't exist
        if not find_cached_vectorstore(transcript_path):
            embed_file(transcript_path)
        # Get the retriever (not cached)
        retriever = get_retriever(transcript_path)
        
        docs = retriever.invoke("누구를 대상으로 하는 내용이라고 생각해요?")
        st.write(docs)