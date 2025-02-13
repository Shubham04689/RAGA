import streamlit as st
import time

def streamlit_chat_loop(chain, retriever):
    """
    Streamlit-based chat interface.
    Run with: streamlit run streamlit_chat.py
    """
    st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")
    
    # Sidebar for instructions
    with st.sidebar:
        st.title("AI Chat Interface 🤖")
        st.markdown("""
        **How to Use:**
        1. Type your query in the text box.
        2. Click 'Send' or press Enter.
        3. View the bot's response and sources.
        """)
        st.markdown("---")
        st.write("💡 Tip: The bot retrieves information from the document database.")
    
    st.title("Chat with AI 🧠")
    st.write("Type your question below and press **Send**.")

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chat input form
    query = st.text_input("Enter your question:", key="query_input")
    send_button = st.button("Send")

    if send_button and query:
        start_time = time.time()

        with st.spinner("Thinking... 🤔"):
            # Get retrieved context and generate answer
            retrieved_context = retriever.invoke(query)
            result = chain.invoke({
                "context": retrieved_context,
                "input": query
            })
        
        elapsed_time = time.time() - start_time

        # Process response
        if isinstance(result, dict):
            answer = result.get("answer", result)
            sources = result.get("sources", [])
        else:
            answer = result
            sources = []

        # Store user input and bot response in session state
        st.session_state.chat_history.append({"speaker": "User", "text": query})
        st.session_state.chat_history.append({
            "speaker": "Bot",
            "text": answer,
            "elapsed": elapsed_time,
            "sources": sources
        })

    # Display chat history
    st.subheader("Chat History 📝")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["speaker"] == "User":
                st.markdown(f"**🧑‍💻 User:** {message['text']}")
            else:
                st.markdown(f"**🤖 AI:** {message['text']}")
                st.markdown(f"_Response time: {message.get('elapsed', 0):.2f} seconds_")
                if message.get("sources"):
                    with st.expander("Sources Used 🔍"):
                        for idx, source in enumerate(message["sources"], start=1):
                            preview = source.get("content", "No content available")[:200]
                            metadata = source.get("metadata", {})
                            st.markdown(f"**Source {idx}:**")
                            st.markdown(f"📄 Preview: {preview}...")
                            st.markdown(f"🔗 **Source:** {metadata.get('source', 'Unknown')}")
                            st.markdown(f"📌 **Page:** {metadata.get('page', 'Unknown')}")

    # Auto-scroll to the latest message
    st.markdown("---")
    st.write("💡 Type another question above to continue the conversation.")

