import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory 
from langchain_core.runnables import RunnableWithMessageHistory
import streamlit as st
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Debug: Check if API key is loaded
if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found in environment variables. Please check your .env file.")
    st.stop()
elif len(groq_api_key) < 10:
    st.error("❌ GROQ_API_KEY appears to be invalid (too short). Please check your .env file.")
    st.stop()
else:
    st.success(f"✅ API Key loaded successfully (ends with: ...{groq_api_key[-8:]})")
   
st.set_page_config(page_title="LLAMA Chatbot", page_icon="🤖")
st.title("🤖 LLAMA Chatbot")
st.markdown("Ask me anything (type something below 👇)")

model=ChatGroq(model="llama-3.1-8b-instant",groq_api_key=groq_api_key)

if 'chat_store' not in st.session_state:
    st.session_state.chat_store = {}

def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in st.session_state.chat_store:
        st.session_state.chat_store[session_id] = ChatMessageHistory()
    return st.session_state.chat_store[session_id]
    
prompt=ChatPromptTemplate.from_messages(
    [
    ("system","You are a helpful chat assistant. Respond naturally to the user's questions and have a normal conversation. Do not translate messages unless specifically asked to do so."),
    MessagesPlaceholder(variable_name="history"),
    ("human","{input}")
    ]
)
chain=prompt|model

with_message_history=RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)


if "session_id" not in st.session_state:
    st.session_state.session_id = "chat_ui"

# Initialize or get the message history for this session
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Clear any existing history for this session
    if st.session_state.session_id in st.session_state.chat_store:
        st.session_state.chat_store[st.session_state.session_id] = ChatMessageHistory()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_prompt = st.chat_input("Ask your question here...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    config = {"configurable": {"session_id": st.session_state.session_id}}
    response = with_message_history.invoke({"input": user_prompt}, config=config)

    st.chat_message("assistant").markdown(response.content)
    st.session_state.messages.append({"role": "assistant", "content": response.content})