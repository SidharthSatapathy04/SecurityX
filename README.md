#  SecurityX — Cybersecurity AI Assistant

**A production-ready LangGraph agentic AI system for cybersecurity threat detection and analysis.**

SecurityX is an intelligent agent that helps security analysts, IT administrators, and developers understand cyber threats, attack techniques, and defense strategies. Built with LangGraph, ChromaDB, and Groq's LLM, it combines retrieval-augmented generation (RAG) with active threat detection to provide grounded, actionable security insights.

---

##  Screenshots

### Main Interface
<img width="1804" height="883" alt="Screenshot 2026-04-16 214223" src="https://github.com/user-attachments/assets/999d85a5-7d54-4d84-b92b-afcc85ab6dc7" />

*The SecurityX Streamlit web interface showing the knowledge base topics sidebar and main chat area.*

### Threat Detection in Action
<img width="834" height="699" alt="Screenshot 2026-04-16 214144" src="https://github.com/user-attachments/assets/2e0abcb5-2b51-4b59-a7f5-791f4d9b1fe4" />

*SecurityX identifying a **Brute Force and Credential Attack** from a "multiple login failure" query — complete with severity level, immediate recommended actions, and prevention measures.*

---

##  Features

- ** Retrieval-Augmented Generation (RAG)**
  - 11-document knowledge base covering cybersecurity topics
  - SentenceTransformer embeddings with ChromaDB vector search
  - 3-result context retrieval with source attribution

- ** Active Threat Detection Tool**
  - Analyzes questions for attack pattern signatures
  - Detects: Brute Force, DDoS, and Phishing attacks
  - Assigns severity levels (Low/Medium/High/Critical)
  - Recommends immediate actions

- ** Conversation Memory**
  - Multi-turn conversations with MemorySaver persistence
  - Thread-based conversation tracking
  - Sliding window history (last 6 messages)

- ** Self-Reflection & Quality Gating**
  - Faithfulness evaluation node with 0.7 threshold
  - Automatic retry mechanism (max 2 attempts)
  - Grounded answers constrained to retrieved context

- ** Adversarial Safety**
  - Detects prompt injection attempts
  - Refuses jailbreak patterns gracefully
  - Maintains consistent behavior under attack

- ** Streamlit Web UI**
  - Interactive chat interface
  - Real-time threat detection warnings
  - Faithfulness scoring display
  - Session management with "New Conversation" button

---

##  Knowledge Base

The agent covers **11 cybersecurity domains**:

1. **SQL Injection** — Web application data manipulation attacks
2. **Cross-Site Scripting (XSS)** — Browser-based code injection
3. **Cross-Site Request Forgery (CSRF)** — Session hijacking via forged requests
4. **Distributed Denial of Service (DDoS)** — Bandwidth and protocol flooding
5. **Phishing Attacks** — Social engineering credential harvesting
6. **Malware and Trojans** — Malicious software delivery and detection
7. **Ransomware Attacks** — File encryption extortion and recovery
8. **Zero-Day Vulnerabilities** — Unknown security flaws in the wild
9. **Firewalls and IDS** — Network security monitoring and enforcement
10. **Incident Response Lifecycle** — 6-phase breach management framework
11. **Brute Force & Credential Attacks** — Authentication compromise prevention

---

##  Architecture

```
User Input (question)
    ↓
[memory_node] → Add to history, apply sliding window
    ↓
[router_node] → Keyword matching → Decide route
    ├→ "retrieve" → [retrieval_node] → ChromaDB query
    ├→ "tool" → [tool_node] → Threat detection
    └→ "skip" → [skip_retrieval_node] → Empty context
    ↓
[answer_node] → LLM generates grounded answer + adversarial detection
    ↓
[eval_node] → Faithfulness score → Retry if <0.7 (max 2x)
    ↓
[save_node] → Append to history
    ↓
Return: {answer, threat_type, severity, faithfulness, sources}
```

**8 LangGraph Nodes:**
1. `memory_node` — Conversation history management
2. `router_node` — Deterministic keyword-based routing
3. `retrieval_node` — ChromaDB context search
4. `skip_retrieval_node` — Memory-only fallback
5. `tool_node` — Threat detector tool
6. `answer_node` — Grounded LLM response generation
7. `eval_node` — Faithfulness evaluation & gating
8. `save_node` — History persistence

---

##  Files

```
SecurityX/
├── agent.py                    # agent module 
│  │
├── capstone_streamlit.py       # Streamlit web UI
│  
├── day13_capstone.ipynb        # Complete Jupyter notebook 
│   ├── Part 1: KB setup & retrieval test
│   ├── Part 2: State design
│   ├── Part 3: Node functions & imports
│   ├── Part 4: Graph assembly
│   ├── Part 5: Test suite (10 questions)
│   ├── Part 6: RAGAS baseline evaluation
│   ├── Part 7: Streamlit deployment
│   └── Part 8: Written summary
│
├── screenshots/                # UI screenshots
│   ├── screenshot_ui.png
│   └── screenshot_chat.png
│
├── .env                        # API key configuration (SAMPLE)
├── .gitignore                  # Git exclusions
└── README.md                   # This file
```

---

##  Quick Start

### Prerequisites

- Python 3.9+
- GROQ API key ([Get one free](https://console.groq.com))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SidharthSatapathy04/SecurityX.git
   cd SecurityX
   ```

2. **Create virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install langgraph langchain-groq langchain-core chromadb \
               sentence-transformers streamlit python-dotenv
   ```

4. **Configure GROQ API key**
   ```bash
   # Edit .env file
   GROQ_API_KEY=your_actual_groq_key_here
   ```

### Running the Agent

**Option 1: Python Script**
```python
from agent import ask, build_agent_graph

# Ask a question
result = ask("What is SQL injection and how can I prevent it?", thread_id="session-1")

print(f"Answer: {result['answer']}")
print(f"Threat: {result['threat_type']}")
print(f"Severity: {result['severity']}")
print(f"Faithfulness: {result['faithfulness']:.2f}")
print(f"Sources: {result['sources']}")
```

**Option 2: Streamlit UI**
```bash
streamlit run capstone_streamlit.py
```
Opens at `http://localhost:8501`

**Option 3: Jupyter Notebook**
```bash
jupyter notebook day13_capstone.ipynb
# Kernel → Restart & Run All
```

---
##  Configuration

### Environment Variables
```env
GROQ_API_KEY=gsk_...  # Your Groq API key
```

### Tuning Parameters (in `agent.py`)

```python
# Faithfulness threshold (line ~450)
FAITHFULNESS_THRESHOLD = 0.7

# Maximum eval retries (line ~451)
MAX_EVAL_RETRIES = 2

# Retrieval result count (line ~303)
n_results = 3

# LLM model (line ~105)
model = "llama-3.3-70b-versatile"

# Conversation window size (line ~177)
if len(msgs) > 6:  # Keep last 6 messages
    msgs = msgs[-6:]
```

---

##  Security Considerations

- **API Key Protection:** Never commit `.env` file; use `.gitignore`
- **Adversarial Resilience:** Detects and refuses prompt injection attempts
- **Grounding:** All answers constrained to retrieved context
- **Fallback Safety:** Graceful error handling in all nodes
- **Rate Limiting:** Consider deploying behind reverse proxy for production

---

##  Improvements for Production

1. **Semantic Router:** Replace keyword matching with LLM-based semantic classification
2. **Persistent Memory:** Upgrade MemorySaver to SQL/PostgreSQL database
3. **Multi-Threaded Retrieval:** Parallelize context retrieval for latency reduction
4. **Fine-tuned Evaluation:** Use domain-specific faithfulness metrics
5. **Monitoring:** Add Langsmith integration for agent observability
6. **API Gateway:** Deploy with FastAPI for scalable serving
7. **Rate Limiting:** Implement per-user/per-IP request throttling
8. **Caching:** Add query result caching to reduce LLM calls

---

##  Learning Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [ChromaDB Vector Database](https://www.trychroma.com/)
- [Groq API Docs](https://console.groq.com/docs)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)

---

##  License

This project is provided as-is for educational purposes. See LICENSE file for details.

---

##  Author

**Sidharth Satapathy**
- GitHub: [@SidharthSatapathy04](https://github.com/SidharthSatapathy04)

---

##  Disclaimer

SecurityX is an educational tool for cybersecurity learning. While the information is based on industry best practices, this agent should **not be used as a substitute for professional security advice**. Always consult qualified cybersecurity professionals for production security decisions.

---
