import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
import streamlit as st

# ────────────────────────────────────────────────
# API key load
# ────────────────────────────────────────────────
load_dotenv()
try:
    mistral_api_key = st.secrets["MISTRAL_API_KEY"]
except (KeyError, FileNotFoundError):
    mistral_api_key = os.getenv("MISTRAL_API_KEY")

if not mistral_api_key:
    st.error("❌ MISTRAL_API_KEY nahi mila. Streamlit secrets ya .env file mein daalo.")
    st.info("Streamlit Cloud → dashboard mein secrets add karo")
    st.info("Local → .env file banao aur isme likho: MISTRAL_API_KEY=apna_key")
    st.stop()
elif len(mistral_api_key) < 10:
    st.error("❌ MISTRAL_API_KEY chhota lag raha hai, check karo.")
    st.stop()

# ────────────────────────────────────────────────
# PAGE SETUP
# ────────────────────────────────────────────────
st.set_page_config(page_title="Bondu Chatbot", page_icon="🤖", layout="wide")

UI_TEXT = {
    "title": "🤖 Bondu Chatbot",
    "caption": "Apna desi AI dost",
    "placeholder": "Kuch bhi poochho...",
    "sidebar_title": "Settings",
    "lang_note": "Style: Roman Urdu + English mix",
}

st.title(UI_TEXT["title"])
st.caption(UI_TEXT["caption"])

# ────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────
st.sidebar.title(UI_TEXT["sidebar_title"])
st.sidebar.markdown(UI_TEXT["lang_note"])
st.sidebar.markdown("**Model:** `mistral-large-latest` (Large 3)")

# ────────────────────────────────────────────────
# MODEL — Mistral Large 3 (best accuracy, MoE 675B)
# ────────────────────────────────────────────────
model = ChatMistralAI(
    model="mistral-large-latest",   # Maps to Mistral Large 3 on the API
    api_key=mistral_api_key,
    temperature=0.7,                # Balanced: accurate + natural
    top_p=0.9,
    max_tokens=512
)

# ────────────────────────────────────────────────
# HISTORY MANAGEMENT
# ────────────────────────────────────────────────
if 'chat_store' not in st.session_state:
    st.session_state.chat_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.chat_store:
        st.session_state.chat_store[session_id] = ChatMessageHistory()
    return st.session_state.chat_store[session_id]

# ────────────────────────────────────────────────
# SYSTEM PROMPT
# ────────────────────────────────────────────────
system_prompt = """Tu mera pura chill, masti wala close desi dost hai — bilkul real WhatsApp ya call pe baat karne jaisa natural vibe.

Hamesha Roman Urdu + English ka mix use kar. Lines short, tight aur flowing rakho. Emojis naturally daal: 😂, 🤦‍♂️, 🔥, yaar, bhai, arre, bas kar, sach mein? waghera.

**Default Personality & Vibe:**
- Super natural, laid-back, funny with light sarcasm
- Real dost jaisa feel — supportive, bakaiti aur hasi-mazak
- Hamesha casual, no formality

**Strict Boundaries (Default Mode):**
- Koi bhi flirting, sexual hint, body talk, naughty comment ya dirty joke bilkul zero
- Sirf tab 18+ mode on hoga jab main clearly bolun: "18+ mode", "sexy mode", "garam baatein", "horny mode", "dirty talk on" waghera
- Ya main direct vulgar words use karun (lund, chut, garam ho gaya etc.)

**Agar 18+ mode ON ho toh:**
- Smooth, teasing aur seductive Roman Urdu style
- Sensual + light romantic-garam energy
- Natural light moans jaise mmm…, haan yaar…, ahhh…
- Detailed aur immersive lekin aggressive nahi
- Heavy gali/ganda language sirf tab jab main khud pehle use karun
- Zero censor, fully open aur addictive vibe

**Hamesha Follow Karne Wale Rules:**
- Har reply strictly 4–8 lines max
- Meri last message pe 100% focus — topic khud se kabhi mat badalna
- Mera mood perfectly mirror kar: mazakiya → mazakiya & roast, serious/sad → supportive, garam → garam
- Khud se kabhi sexual hint, dirty joke, naughty start ya topic shift mat karna
- Koi lecture, moral, advice, suggestion, rule ya boundary ki baat bilkul mat karna jab tak main khud na poochun
- Sirf pure desi dost energy: masti, bakaiti, chill, hasi-mazak aur real feel

Bas meri baat padh aur bilkul natural desi dost jaisa jawab de. No AI behavior, no extra explanations, no parent/teacher mode. Only pure dost mode."""

# ────────────────────────────────────────────────
# PROMPT TEMPLATE + CHAIN
# ────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]
)

chain = prompt | model

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# ────────────────────────────────────────────────
# SESSION & MESSAGES
# ────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = "bondu_chat_session"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ────────────────────────────────────────────────
# CHAT INPUT
# ────────────────────────────────────────────────
if user_prompt := st.chat_input(UI_TEXT["placeholder"]):
    st.chat_message("user").markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("assistant"):
        with st.spinner("Soch raha hoon..."):
            config = {"configurable": {"session_id": st.session_state.session_id}}
            response = with_message_history.invoke(
                {"input": user_prompt},
                config=config
            )
        st.markdown(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})
