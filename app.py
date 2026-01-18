import streamlit as st
from src.rag_pipeline import RAGPipeline
import time

# Page config
st.set_page_config(
    page_title="Ask Himanshu - AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1E88E5;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2em;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize RAG pipeline (cached)
@st.cache_resource
def load_pipeline():
    return RAGPipeline()

# Header
st.markdown('<p class="main-header">🤖 Ask Himanshu</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Your AI Assistant for Learning About Himanshu Ramteke</p>',
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This AI assistant can answer questions about:
    
    - 📄 Professional experience
    - 🎓 Education & certifications
    - 🚀 Projects & publications
    - 💻 Technical skills
    - 🏆 Achievements
    
    **Privacy Note:** Personal contact information is protected.
    For inquiries, please use email: hsramteke21@gmail.com
    """)
    
    st.header("Sample Questions")
    st.code("""
    - What is Himanshu's experience?
    - Tell me about the forecasting project
    - What technologies does he know?
    - What are his research publications?
    - Where did he study?
    """)
    
    st.header("Technology Stack")
    st.text("🧠 LLM: Llama-3.1-8B (Groq)\n📚 Vector DB: Qdrant\n🔍 RAG: LangChain\n🎨 Frontend: Streamlit")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources"):
                for src in message["sources"]:
                    st.text(f"• {src['source']} (Page: {src['page']}, Relevance: {src['score']:.2f})")

# Chat input
if prompt := st.chat_input("Ask me anything about Himanshu..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Load pipeline
            pipeline = load_pipeline()
            
            # Get answer
            result = pipeline.answer_query(prompt)
            
            # Display answer
            st.markdown(result['answer'])
            
            # Display sources if available
            if result['sources']:
                with st.expander("📚 Sources"):
                    for src in result['sources']:
                        st.text(f"• {src['source']} (Page: {src['page']}, Relevance: {src['score']:.2f})")
    
    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result['answer'],
        "sources": result['sources']
    })

# Footer
st.markdown("---")
st.markdown(
    """
    <p style='text-align: center; color: #666;'>
        Built with ❤️ by Himanshu Ramteke | 
        <a href='https://github.com/Himanshu-hub21' target='_blank'>GitHub</a> | 
        <a href='https://www.linkedin.com/in/himanshuramtekehr21/' target='_blank'>LinkedIn</a>
    </p>
    """,
    unsafe_allow_html=True
)

