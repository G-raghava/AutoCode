import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
 
# Load environment variables
# os.environ["GOOGLE_API_KEY"] ="AIzaSyA0COlvhMaj6k6_6zNmwF_ShC-nImEHVBo"
 
# Constants

EMBEDDINGS_MODEL_NAME = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
TARGET_SOURCE_CHUNKS = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))
PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY', 'db_vsc')
 
 
def main():
    # Initialize Streamlit
    st.title("Interactive Chatbot for code")
    st.markdown("---")
    
    if 'api_key' not in st.session_state:
        with st.form(key='api_key_form'):
            api_key = st.text_input("Enter your Google API key:", type="password")
            submit_button = st.form_submit_button(label='Submit')
            if submit_button and api_key:
                st.session_state.api_key = api_key
                os.environ["GOOGLE_API_KEY"] = api_key
    else:
        os.environ["GOOGLE_API_KEY"] = st.session_state.api_key
   
    user_input = st.chat_input("Enter a query:")
    hide_source = st.checkbox("Hide source documents", value=True, key="hide_source")
 
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
 
    # Initialize embeddings and Chroma database (only on first run)
    if not st.session_state.get("embeddings"):
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
        db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        st.session_state["embeddings"] = embeddings
        st.session_state["db"] = db
    else:
        embeddings = st.session_state["embeddings"]
        db = st.session_state["db"]
 
    # Initialize the conversational components
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    retriever = db.as_retriever(search_kwargs={"k": TARGET_SOURCE_CHUNKS})
 
    # Define the template for prompts with context placeholder
    template = """
    You are a helpful AI Assistant that follows instructions extremely well.
    Think step by step before answering the question.
    Answer the question
 
    CONTEXT: {context}
    {history}
    </s>
 
    {question}
    </s>
    """
    PROMPT = PromptTemplate(
        template=template, input_variables=["context", "history", "question"]
    )
 
    # Set up memory for conversation history
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="history", input_key="question")
 
    def build_retrieval_qa(llm, prompt, retriever):
        dbqa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=retriever,
            return_source_documents=not hide_source,
            chain_type_kwargs={"prompt": PROMPT, "memory": st.session_state.memory},
            verbose=True
        )
        return dbqa
 
    dbqa = build_retrieval_qa(llm=llm, prompt=PROMPT, retriever=retriever)
 
    if user_input:
        # Process user query
        res = dbqa(user_input)
        answer, docs = res['result'], [] if hide_source else res['source_documents']
 
        # Update conversation history
        st.session_state.chat_history.append({"user": user_input, "answer": answer})
 
        # Display chat history
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat['user'])
            with st.chat_message("assistant"):
                st.write(chat['answer'])
 
        if not hide_source:
            st.write("**Source Documents:**")
            for index, document in enumerate(docs, start=1):  # Start index from 1
                source_label = f"**Source {index}:**"  # Create custom label using index
                st.write(source_label)
                st.write(f"{document.metadata['source']}")  # Print metadata
                st.write(document.page_content)  # Print content
 
 
if __name__ == "__main__":
    main()