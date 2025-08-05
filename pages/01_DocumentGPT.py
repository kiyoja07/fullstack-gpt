import time
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

st.title("DocumentGPT")

if "messages" not in st.session_state:
    st.session_state["messages"] = [] # session_state는 페이지 새로 고침 후에도 데이터가 유지됨


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message) # 메시지 표시
    if save:
        st.session_state["messages"].append({"message": message, "role": role}) # 메시지 저장

# 이전 메시지 복원
for message in st.session_state["messages"]:
    send_message(
        message["message"],
        message["role"],
        save=False, # 🔑 중요: 중복 저장 방지
    )


message = st.chat_input("Send a message to the ai ")

if message:
    send_message(message, "human")
    time.sleep(2)
    send_message(f"You said: {message}", "ai")

    with st.sidebar:
        st.write(st.session_state)