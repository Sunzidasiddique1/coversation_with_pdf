import os
import json
import streamlit as st
from PyPDF2 import PdfReader
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from ollama import pull

# Pull the embedding model
embedding_model = pull("mxbai-embed-large")

# Initialize the Llama 3.2 model with Ollama
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0,
)

# Create prompts for summarization and question answering
summary_prompt = ChatPromptTemplate.from_messages(
    [("system", "You are an AI that specializes in summarizing documents."),
     ("human", "Summarize the following document: {document}")]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [("system", "You are an AI that answers questions based on document content."),
     ("human", "Document: {document}\nQuestion: {question}")]
)

# Function to load and extract text from PDF documents
def load_document(file_path):
    pdf_reader = PdfReader(open(file_path, "rb"))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""  # Ensure we don't get None
    return text

# Function to save chat history to a JSON file
def save_chats(chat_history, file_path="chat_history.json"):
    with open(file_path, "w") as f:
        json.dump(chat_history, f)

# Function to load chat history from a JSON file
def load_chats(file_path="chat_history.json"):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return []

# Function to create embeddings for the PDF chunks
def create_embeddings(chunks, embedding_model, storing_path="vectorstore"):
    os.makedirs(storing_path, exist_ok=True)
    embeddings = [embedding_model(chunk) for chunk in chunks]
    with open(os.path.join(storing_path, 'embeddings.json'), 'w') as f:
        json.dump(embeddings, f)

# Streamlit app
st.title("Chat with Your Document Using Meta Llama 3.2 (Ollama & LangChain)")

# File uploader for the PDF
file_uploader = st.file_uploader("Choose a PDF file:", type=["pdf"])

if file_uploader is not None:
    # Save and extract text from the uploaded PDF file
    with open("uploaded_document.pdf", "wb") as f:
        f.write(file_uploader.getvalue())
    document_text = load_document("uploaded_document.pdf")
    
    if document_text.strip():  # Proceed only if text is not empty
        # Summarize the document content
        st.write("Summarizing the document...")
        summary_chain = summary_prompt | llm
        summary = summary_chain.invoke({"document": document_text})
        # Hide the summary display
        # st.write("Summary:")
        # st.write(summary["content"])

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        # Load chat history from file
        st.session_state["messages"] = load_chats()

        # Create embeddings for document chunks
        chunks = [document_text[i:i+1000] for i in range(0, len(document_text), 1000)]
        create_embeddings(chunks, embedding_model)

        # Chat interface for question answering
        st.write("You can now ask questions about the document.")

        # Text input for the user question
        user_input = st.text_input("Ask a question about the document:")

        if st.button("Send") and user_input:
            # Add the user's question to the chat history
            st.session_state.messages.append(("user", user_input))

            # Generate the answer using LangChain and Ollama
            qa_chain = qa_prompt | llm
            answer = qa_chain.invoke({"document": document_text, "question": user_input})

            # Add the AI's answer to the chat history
            st.session_state.messages.append(("ai", answer.content if hasattr(answer, 'content') else answer))

            # Save chat history to file
            save_chats(st.session_state.messages)

        # Display the chat history
        st.write("### Chat History:")
        for message in st.session_state.messages:
            if message[0] == "user":
                st.write(f"**You:** {message[1]}")
            else:
                st.write(f"**AI:** {message[1]}")
    else:
        st.write("The document appears to be empty or not properly extracted. Please try another document.")
