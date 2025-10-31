# Use pip installed package instead of the built-in sqlite3
try:
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    # On macOS, the built-in sqlite3 should work fine
    pass

import streamlit as st
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()


def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()

    # Split the documents into chunks.
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # Create a vectorstore from the chunks.
    vector_store = Chroma.from_documents(
        document_chunks,
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
        ),
    )

    return vector_store


def get_context_retriever_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
            ),
        ]
    )

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain(retriever_chain):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                "Answer the user's questions based on the below context:\n\n{context}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_query):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke(
        {"chat_history": st.session_state.chat_history, "input": user_query}
    )

    return response["answer"]


# Configuration
st.set_page_config(page_title="Chat with website", page_icon="🤖")

st.title("Chat with websites")

# Sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if website_url is None or website_url == "":
    st.info("Please enter a website URL")

else:
    # Session states
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! How can I help you?")
        ]

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

    # Input handling
    user_query = st.chat_input("Type your message here ...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Conversation handler
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("ai"):
                st.write(message.content)

        elif isinstance(message, HumanMessage):
            with st.chat_message("human"):
                st.write(message.content)
