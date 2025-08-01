import streamlit as st

st.title("Hello world!")

st.subheader("Welcome to Streamlit!!")

st.markdown(
    """
    #### I love it!
"""
)

st.selectbox(
    "Choose your model",
    (
        "GPT-3",
        "GPT-4",
    ),
)