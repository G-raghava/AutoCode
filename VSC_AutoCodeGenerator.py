import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/query/"

st.title("Interactive Chatbot for Code")
st.markdown("---")

user_input = st.chat_input("Enter a query:")
hide_source = st.checkbox("Hide source documents", value=True, key="hide_source")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if user_input:
    response = requests.post(API_URL, json={"question": user_input, "chat_history": st.session_state.chat_history})
    if response.status_code == 200:
        data = response.json()
        answer = data["answer"]
        docs = [] if hide_source else data["docs"]

        st.session_state.chat_history.append({"user": user_input, "answer": answer})

        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat['user'])
            with st.chat_message("Assistant"):
                st.write(chat['answer'])

        if not hide_source:
            st.write("**Source Documents:**")
            for index, document in enumerate(docs, start=1):
                source_label = f"**Source {index}:**"
                st.write(source_label)
                st.write(f"{document['metadata']['source']}")
                st.write(document['page_content'])
    else:
        st.error("Error in fetching response from API")
