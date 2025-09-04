import streamlit as st
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda


from pages.SiteGPT.chat_handler import get_answers, choose_answer, memory
from pages.SiteGPT.data_loader import load_website, find_history
from pages.SiteGPT.utils import paint_history, send_message


st.set_page_config(page_title="Quiz GPT", page_icon="❓")
st.title("Quiz GPT")
st.markdown(
    """
    Ask questions about the content of a website.
    Start by writing the URl of the website on the sidebar.
"""
)


def load_memory(_):
    return memory.load_memory_variables({})["chat_history"]


with st.sidebar:
    url = st.text_input("Write down a URL", placeholder="https://example.com")

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        retriever = load_website(url)
        send_message("I'm ready! Ask away!", "ai", False)
        paint_history()

        query = st.chat_input("Ask a question to the website.")
        if query:
            send_message(query, "human")

            found = find_history(query)
            if found:
                send_message(found, "ai")
            else:
                chain = (
                    {
                        "docs": retriever,
                        "question": RunnablePassthrough(),
                    }
                    | RunnablePassthrough.assign(chat_history=load_memory)
                    | RunnableLambda(get_answers)
                    | RunnableLambda(choose_answer)
                )

                with st.chat_message("ai"):
                    result = chain.invoke(query)
                memory.save_context(
                    {"input": query},
                    {"output": result.content},
                )
else:
    st.session_state["messages"] = []


# https://github.com/hwna00/fullstack-gpt/commit/4cac8704cf05b8a88379e7ea9ecef5968e51eff9