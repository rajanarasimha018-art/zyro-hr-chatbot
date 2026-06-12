import streamlit as st

st.title("Zyro Dynamics HR Chatbot")

question = st.text_input("Ask an HR Question")

if question:
    st.write("Question:", question)
    st.write("Response will appear here.")