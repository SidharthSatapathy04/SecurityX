"""
capstone_streamlit.py — SecurityX Cybersecurity Assistant Streamlit UI
Run: streamlit run capstone_streamlit.py
"""

import streamlit as st
import uuid
from agent import ask, DOCUMENTS

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SecurityX - Cybersecurity Assistant",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🛡️ SecurityX — Cybersecurity Assistant")
st.caption("AI-powered threat detection and cybersecurity guidance")

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("📋 About SecurityX")
    
    st.write(
        """SecurityX is an AI cybersecurity assistant that helps security analysts, 
developers, and IT administrators understand cyber threats, attack techniques, and 
defense strategies. The agent uses RAG (Retrieval-Augmented Generation) with a 
knowledge base of 11 cybersecurity topics, and includes an active threat detection 
tool that analyzes security incidents."""
    )
    
    st.divider()
    
    st.header("🔍 Knowledge Base Topics")
    kb_topics = [d["topic"] for d in DOCUMENTS]
    for topic in kb_topics:
        st.write(f"• {topic}")
    
    st.divider()
    
    st.header("⚙️ Session Info")
    st.write(f"**Thread ID:** `{st.session_state.thread_id}`")
    st.write(f"**Messages:** {len(st.session_state.messages)}")
    
    if st.button("🗑️ New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()
    
    st.divider()
    st.caption("**Example questions:**")
    st.caption("• What is SQL injection?")
    st.caption("• We're seeing 500 failed logins/min from 10 IPs")
    st.caption("• How does ransomware work?")
    st.caption("• What should we do in a DDoS attack?")

# ============================================================
# CHAT HISTORY DISPLAY
# ============================================================

st.header("💬 Conversation")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ============================================================
# CHAT INPUT & RESPONSE
# ============================================================

if prompt := st.chat_input("Ask about cybersecurity threats, vulnerabilities, or defenses..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing..."):
            try:
                result = ask(prompt, thread_id=st.session_state.thread_id)
                answer = result.get("answer", "Sorry, I could not generate an answer.")
                threat_type = result.get("threat_type", "None")
                severity = result.get("severity", "Low")
                route = result.get("route", "unknown")
                sources = result.get("sources", [])
                faithfulness = result.get("faithfulness", 0.0)
            except Exception as e:
                answer = f"Error: {str(e)}"
                threat_type = "Error"
                severity = "Unknown"
                route = "error"
                sources = []
                faithfulness = 0.0
        
        # Display answer
        st.write(answer)
        
        # Display metadata
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"📍 Route: `{route}`")
        with col2:
            st.caption(f"✓ Faithfulness: `{faithfulness:.2f}`")
        with col3:
            st.caption(f"🎯 Threat: `{threat_type}`")
        
        # Display threat warning if applicable
        if threat_type != "None" and threat_type != "Error":
            st.warning(f"⚠️ **THREAT DETECTED:** {threat_type} (Severity: {severity})", icon="🚨")
        
        # Display sources if available
        if sources:
            st.info(f"📚 Sources: {', '.join(sources)}")
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
