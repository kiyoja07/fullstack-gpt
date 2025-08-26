import streamlit as st
import json
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chat_models.openai import ChatOpenAI
import random

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

function = {
    "name": "get_questions",
    "description": "질문과 여러개의 보기로 이루어져 있는 questions array를 필요로 하는 function입니다.",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answer"],
                },
            },
        },
        "required": ["questions"],
    },
}

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-0125",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
).bind(
    function_call={
        "name": "get_questions",
    },
    functions=[function],
)

prompt = PromptTemplate.from_template(
    """
당신은 주어진 문서들을 기반으로 학생들의 지식 수준을 시험하는 문제를 출제하는 프로 출제자입니다.
주어질 Context에 등장하는 정보들을 바탕으로 10개의 문제를 출제하세요.
모든 문제는 총 4개의 보기가 있으며 그 중 한개만 정답입니다.
모든 문제는 짧고 유니크하게 출제하세요.

--------Context--------
{context}
-----------------------
"""
)


def foramt_document(docs):
    return "\n\n".join(document.page_content for document in docs)


@st.cache_data(show_spinner='"위키피디아"에 검색 중..')
def get_from_wikipedia(topic):
    retriever = WikipediaRetriever(lang="ko")
    return retriever.get_relevant_documents(topic)


@st.cache_data(show_spinner="문제 생성 중..")
def generate_questions(_docs, topic):
    chain = {"context": foramt_document} | prompt | llm
    response = chain.invoke(_docs)
    arguments = json.loads(response.additional_kwargs["function_call"]["arguments"])
    for index in range(len(arguments["questions"])):
        random.shuffle(arguments["questions"][index]["answers"])
    return arguments


@st.cache_data(show_spinner="로딩 중..")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n", chunk_size=600, chunk_overlap=100
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


with st.sidebar:
    docs = None
    topic = None
    choice = st.selectbox(
        "어떤 정보를 사용 하실지 선택해 주세요.",
        (
            "파일",
            "위키피디아",
        ),
    )
    if choice == "파일":
        file = st.file_uploader("문서를 업로드해 주세요.", type=["pdf", "docx", "txt"])
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input(
            "위키피디아에서 검색", placeholder="검색할 내용을 입력해 주세요."
        )
        if topic:
            docs = get_from_wikipedia(topic)
    show_answer = st.toggle("틀렸을 때 답 표시하기", False)


if not docs:
    st.markdown(
        """
QuizGPT에 오신 것을 환영합니다.

저는 위키피디아의 자료나 당신이 업로드한 파일을 이용해서 당신의 공부를 도울 것입니다.

사이드바에서 위키피디아에 검색하거나 당신의 파일을 업로드해서 시작해보세요.
"""
    )
else:
    response = generate_questions(docs, topic if topic else file.name)
    with st.form("questions_form"):
        for idx, question in enumerate(response["questions"]):
            value = st.radio(
                f"{idx+1}: {question['question']}",
                [
                    f"{index+1}: {answer['answer']}"
                    for index, answer in enumerate(question["answers"])
                ],
                index=None,
            )
            isCorrect = False
            if value:
                isCorrect = {"answer": value[3:], "correct": True} in question[
                    "answers"
                ]
            if isCorrect:
                st.success("✅ 정답입니다!")
            elif value:
                if show_answer:
                    for index, answer in enumerate(question["answers"]):
                        if "correct" in answer and answer["correct"]:
                            answer_number = index + 1
                            break
                    st.error(f"❌ 오답입니다. (정답: {answer_number}번)")
                else:
                    st.error("❌ 오답입니다.")
            st.divider()
        button = st.form_submit_button()