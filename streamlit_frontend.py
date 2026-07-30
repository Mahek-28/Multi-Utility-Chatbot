import streamlit as st
from langgraph_backend import (chatbot, ingest_pdf,retrieve_all_threads, thread_document_metadata,)
from langchain_core.messages import AIMessage, HumanMessage,ToolMessage
import uuid  # generate new unique thread IDs

#................Utility Function..................#

def generate_thread_id():
    """Return a new UUID for a chat thread."""
    thread_id = uuid.uuid4()
    return thread_id


def reset_chat():
    """Start a fresh conversation by creating a new thread."""
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []


def add_thread(thread_id):
    """Ensure a thread ID is tracked in the sidebar list."""
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def load_conversation(thread_id):
    """Load stored messages from the backend for a thread."""
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])


def get_thread_title(thread_id):
    """Return a human-friendly title for a thread, caching it in session state."""
    if 'thread_titles' not in st.session_state:
        st.session_state['thread_titles'] = {}

    if thread_id in st.session_state['thread_titles']:
        return st.session_state['thread_titles'][thread_id]

    messages = load_conversation(thread_id)
    title = 'New chat'
    for msg in messages:
        if isinstance(msg, HumanMessage) and msg.content.strip():
            title = msg.content.strip().replace('\n', ' ')
            break

    if len(title) > 40:
        title = title[:37].rstrip() + '...'

    st.session_state['thread_titles'][thread_id] = title
    return title

def extract_text(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text = ""
        for part in content:
            if isinstance(part, dict):
                text += part.get("text", "")
        return text

    return str(content)

#................session setup.....................#
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

if 'thread_titles' not in st.session_state:
    st.session_state['thread_titles'] = {}

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state['thread_id'])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key,{})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

#................Sidebar UI........................#
st.sidebar.title('🧠 LangGraph PDF Chatbot')

if st.sidebar.button('🆕 New Chat', use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_docs = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using`{latest_docs.get('filename')}`"
        f"({latest_docs.get('chunks')} chunks from {latest_docs.get('documents')} pages)"
    )
else:
    st.sidebar.info("⚠️ No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("📄 Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"📚 `{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("🔍 Indexing PDF...",expanded= True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete",expanded=False)

st.sidebar.header("💬 My Conversations")

with st.sidebar.container(height=450):   # optional scrollable area
    for thread_id in st.session_state['chat_threads'][::-1]:
        thread_label = get_thread_title(thread_id)

        if st.button(
            thread_label,
            key=f"thread_{thread_id}",
            use_container_width=True
        ):
            st.session_state["thread_id"] = thread_id
            messages = load_conversation(thread_id)

            temp_messages = []
            for msg in messages:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                temp_messages.append(
                    {"role": role, "content": msg.content}
                )

            st.session_state["message_history"] = temp_messages
            st.rerun()

#................Main UI...........................#
st.title("🤖 Multi Utility Chatbot")

# display the loaded conversation in the chat window
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

# user input
user_input = st.chat_input('Type Here...')

if user_input:
    # append the user message locally right away
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']},
              "metadata":{
                  "thread_id":st.session_state['thread_id']},
                  "run_name":"chat_turn",
            }

    # stream assistant tokens from the backend
    with st.chat_message('assistant'):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box":None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages'
            ):
                
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    # yield only assistant tokens
                    yield extract_text(message_chunk.content)

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )


    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')}"
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

    st.divider()

    if selected_thread:
        st.session_state["thread_id"] = selected_thread
        messages = load_conversation(selected_thread)

        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({'role':role,"content":msg.content})
        st.session_state["message_history"] = temp_messages
        st.session_state["igested_docs"].setdefault(str(selected_thread),{})
        st.rerun()