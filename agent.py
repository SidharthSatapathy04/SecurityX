"""
agent.py — SecurityX Cybersecurity Agent
Standalone module containing all agent logic.
Usage: from agent import ask, build_agent_graph
"""

import os
from typing import TypedDict, List, Literal
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

# ============================================================
# PART 1 — KNOWLEDGE BASE (11 DOCUMENTS)
# ============================================================

DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "SQL Injection",
        "text": """SQL Injection is a code injection technique used by attackers to insert malicious SQL statements
into input fields of a web application. When the application executes the query, the injected SQL code is
interpreted as part of the query logic, potentially allowing attackers to view, modify, or delete sensitive data
in the underlying database. SQL injection occurs when user input is not properly sanitized or parameterized
before being included in database queries. Common vulnerable patterns include string concatenation with user
input, lack of input validation, and missing prepared statements. Detection signatures include unusual SQL
keywords in input fields (e.g., 'UNION', 'DROP', 'SELECT'), comment sequences (-- or #), and quote characters
used to break out of string context. Mitigation strategies: (1) Use parameterized queries or prepared statements
which separate SQL code from data, (2) Implement input validation whitelisting only expected characters,
(3) Apply principle of least privilege to database accounts, (4) Use Web Application Firewalls (WAF) to detect
injection patterns, (5) Regular security testing and code reviews, (6) Educate developers on secure coding
practices. Common tools for testing include SQLMap and manual testing with OWASP guidelines."""
    },
    {
        "id": "doc_002",
        "topic": "Cross-Site Scripting (XSS)",
        "text": """Cross-Site Scripting (XSS) is a security vulnerability that allows attackers to inject malicious
scripts into web pages viewed by other users. These scripts execute in the victim's browser within the security
context of the vulnerable website, potentially stealing cookies, session tokens, or sensitive information.
There are three main types of XSS: (1) Stored XSS where malicious code is stored on the server and executed
for all users visiting the page, (2) Reflected XSS where the payload is reflected in the response to a request
sent by the attacker, and (3) DOM-based XSS where the vulnerability exists in client-side code that processes
user input unsafely. Attack vectors include form inputs, URL parameters, search fields, and comment sections.
Common payload markers include <script>, <img onerror=>, <svg onload=>, and event handlers.
Indicators of XSS attacks: JavaScript code in unexpected locations, unusual HTML tags in user input, encoding
bypasses, and polyglot payloads. Prevention measures: (1) Encode all user input before displaying in HTML
context, (2) Use Content Security Policy (CSP) headers to restrict script execution, (3) Implement HTTPOnly
and Secure flags on cookies, (4) Use templating engines with automatic escaping, (5) Validate and sanitize
all user input, (6) Regular security testing including automated scanning and manual penetration testing."""
    },
    {
        "id": "doc_003",
        "topic": "Cross-Site Request Forgery (CSRF)",
        "text": """Cross-Site Request Forgery (CSRF) is an attack where an attacker tricks a user into performing
unwanted actions on a website where they are authenticated. Since the browser automatically includes session
cookies with requests to the target domain, the attacker can forge requests that appear to come from the
legitimate user. CSRF attacks can lead to unauthorized money transfers, password changes, data modifications,
or other actions with the privileges of the victim account. Attack mechanism: (1) Attacker creates a malicious
webpage or email containing a forged request to target site, (2) Victim clicks link while authenticated to
target site, (3) Browser automatically includes valid session cookie with the forged request, (4) Server
processes the request assuming it's legitimate from the authenticated user. Common attack vectors: hidden form
submissions, image tags with URLs, AJAX requests from malicious sites. Detection indicators: requests from
unexpected referrers, suspicious action sequences, unusual IP addresses or user agents for authenticated
sessions. CSRF prevention strategies: (1) Implement CSRF tokens unique per session or per request,
(2) Use SameSite cookie attributes to prevent cross-site cookie inclusion, (3) Verify Origin and Referer
headers, (4) Require re-authentication for sensitive operations, (5) Use POST instead of GET for state-changing
operations, (6) Implement proper CORS policies."""
    },
    {
        "id": "doc_004",
        "topic": "Distributed Denial of Service (DDoS)",
        "text": """Distributed Denial of Service (DDoS) attacks flood target systems with massive traffic volumes
from multiple sources to render services unavailable. Unlike single-source denial of service attacks, DDoS
involves botnets or coordinated networks of compromised machines. Attack types include: (1) Volumetric attacks
(UDP floods, ICMP floods, DNS amplification) consuming bandwidth, (2) Protocol attacks (SYN floods, fragmented
packets, Ping of Death) exploiting protocol weaknesses, (3) Application-layer attacks (HTTP floods, Slowloris,
ReDoS) targeting application logic. Attack detection signatures: sudden traffic spikes, unusual request
patterns, requests from diverse geographic locations, repeated requests to specific resources, traffic from
known botnet IP ranges. Mitigation strategies: (1) Rate limiting and traffic throttling, (2) Blackhole routing
to null-route attack traffic, (3) Traffic scrubbing via DDoS mitigation services, (4) Content Delivery
Networks (CDN) for traffic distribution, (5) Firewalls and routers configured to drop attack traffic,
(6) Anycast network routing spreading traffic across servers, (7) Emergency response procedures and
coordination with ISP. Response time is critical for DDoS incidents, often requiring manual intervention
to activate mitigation measures."""
    },
    {
        "id": "doc_005",
        "topic": "Phishing Attacks",
        "text": """Phishing is a social engineering attack where attackers impersonate trusted entities via email,
messages, or websites to trick users into revealing sensitive information like credentials, payment details,
or personal data. Phishing campaigns often use urgency, fear, or curiosity to motivate victims to act quickly
without careful verification. Common phishing tactics: spoofed email addresses closely mimicking legitimate
domains, urgent language ('Verify your account', 'Immediate action required'), requests for sensitive
information, malicious links or attachments, poor grammar and spelling errors. Variants include spear phishing
targeting specific individuals with customized content, whaling targeting high-level executives, vishing using
voice calls, and smishing using SMS. Detection methods: email filtering and scanning, URL reputation checking,
domain authentication (SPF, DKIM, DMARC), user awareness training, multi-factor authentication. Response
procedures: (1) Do not click links or download attachments from suspicious emails, (2) Verify requests through
alternative channels, (3) Report suspicious emails to security team, (4) Change credentials if compromised,
(5) Monitor account activity for unauthorized access, (6) Enable multi-factor authentication where available."""
    },
    {
        "id": "doc_006",
        "topic": "Malware and Trojans",
        "text": """Malware is malicious software designed to infiltrate, damage, or compromise computer systems
and networks. Malware types include: (1) Viruses that attach to files and replicate when executed, (2) Worms
that self-replicate across networks, (3) Trojans that masquerade as legitimate software but execute malicious
code, (4) Spyware that monitors user activity and steals information, (5) Adware that displays unwanted
advertisements, (6) Rootkits that gain privileged access and hide other malware, (7) Keyloggers capturing
keyboard input. Malware delivery vectors: malicious email attachments, compromised websites, software
vulnerabilities, unpatched systems. System indicators of malware infection: unusual system slowdown, unexpected
network traffic, new processes or services appearing, disabled antivirus software, modified system files or
permissions, pop-up windows. Detection approaches: (1) Antivirus and anti-malware scanning, (2) Behavior
analysis and sandboxing, (3) Network intrusion detection, (4) File integrity monitoring. Removal and
remediation: (1) Disconnect from network to prevent spread, (2) Boot from clean media, (3) Run comprehensive
antimalware scans, (4) Remove quarantined files, (5) Update all software and OS, (6) Change all passwords
from clean system, (7) Monitor for re-infection indicators."""
    },
    {
        "id": "doc_007",
        "topic": "Ransomware Attacks",
        "text": """Ransomware is a type of malware that encrypts victim's files or locks system access, then
demands payment (ransom) for decryption or unlock. Ransomware attacks have evolved from individual attacks
to sophisticated, targeted campaigns against organizations. Attack lifecycle: (1) Initial compromise via
phishing, exploitation, or weak credentials, (2) Lateral movement through network, (3) Persistence mechanisms
to maintain access, (4) Encryption of sensitive files or system volume, (5) Ransom note displaying payment
instructions, (6) Data exfiltration by threat actors in modern attacks. Common variants: CryptoLocker
pioneering file encryption, WannaCry worm-like propagation, Petya/NotPetya affecting Master Boot Record,
REvil targeting large organizations. Detection patterns: unusual file encryption, file extensions changing
to unknown formats, inaccessible files, ransom notes appearing, sudden network slowdown. Mitigation and
prevention: (1) Regular offline backups to enable recovery without paying ransom, (2) Segmentation of
networks to limit spread, (3) Multi-factor authentication reducing initial compromise, (4) EDR tools
detecting malicious behavior, (5) Apply security updates promptly, (6) User awareness training for phishing.
Incident response: (1) Isolate affected systems immediately, (2) Identify attack vector and secure it,
(3) Restore from clean backups if available, (4) Do NOT pay ransom."""
    },
    {
        "id": "doc_008",
        "topic": "Zero-Day Vulnerabilities",
        "text": """Zero-day vulnerabilities are security flaws unknown to vendors and public until actively
exploited in the wild. The 'zero' refers to the zero days vendors have to patch the vulnerability before
attacks occur. Zero-day exploits are highly valuable in cybercriminal markets and nation-state arsenals.
Characteristics: (1) No patch available at time of exploitation, (2) Antivirus signatures do not exist,
(3) System appears fully patched yet remains vulnerable, (4) Exploitation often requires sophisticated
technical knowledge. Vulnerability lifecycle: (1) Code flaw exists in software but unknown to vendor,
(2) Attacker discovers flaw and develops exploit, (3) Attacker uses exploit against targets in wild,
(4) Vendor becomes aware of attacks, (5) Vendor develops and releases patch. Notable zero-days: EternalBlue
(SMB), Heartbleed (OpenSSL), Shellshock (Bash), Log4Shell (Java). Mitigation strategies:
(1) Apply defense-in-depth approach with multiple security layers, (2) Network segmentation limiting blast
radius, (3) EDR solutions detecting unusual behavior rather than signatures, (4) Threat hunting proactively
looking for exploitation indicators, (5) Vulnerability disclosure programs, (6) Bug bounty programs
identifying vulnerabilities before public disclosure, (7) Keep software updated regularly to minimize
exposure window."""
    },
    {
        "id": "doc_009",
        "topic": "Firewalls and Intrusion Detection Systems",
        "text": """Firewalls are security devices that monitor and control incoming and outgoing network traffic
based on predetermined security rules, functioning as barriers between trusted internal networks and untrusted
external networks. Types of firewalls: (1) Packet filtering (stateless) inspecting individual packets against
rules, (2) Stateful firewalls tracking connection states, (3) Application-layer firewalls (WAF) understanding
application protocols, (4) Next-generation firewalls combining traditional firewall with advanced features like
IPS and threat intelligence. Intrusion Detection Systems (IDS) monitor network traffic for signs of malicious
activity or policy violations. Network-based IDS (NIDS) monitors traffic on network segments, while
Host-based IDS (HIDS) monitors individual systems. IDS detection methods: (1) Signature-based matching known
attack patterns, (2) Anomaly-based establishing baseline and detecting deviations, (3) Behavior-based
identifying suspicious activity patterns. IPS (Intrusion Prevention System) similar to IDS but can actively
block malicious traffic. Configuration best practices: (1) Default deny, explicit allow policy, (2) Separate
rules for inbound/outbound traffic, (3) Log rule hits for forensic analysis, (4) Regular rule updates and
threat intelligence integration, (5) Testing rule effectiveness regularly."""
    },
    {
        "id": "doc_010",
        "topic": "Incident Response Lifecycle",
        "text": """Incident response is a coordinated approach to managing security incidents and breaches.
A structured incident response program minimizes damage and recovery time. The incident response lifecycle
consists of six phases: (1) PREPARATION - establishing IR team, tools, communication channels, and incident
classification criteria, (2) DETECTION AND ANALYSIS - identifying security events, determining if incident
occurred, and collecting initial evidence, (3) CONTAINMENT - stopping the attack progression, preventing
further compromise using short-term and long-term containment, (4) ERADICATION - removing attacker tools,
malware, and closing exploited vulnerabilities, (5) RECOVERY - restoring systems to normal operations and
monitoring for re-infection, (6) POST-INCIDENT ACTIVITY - conducting root cause analysis, lessons learned,
updating IR procedures, and sharing threat intelligence. Key roles: IR Coordinator, SOC Analysts, Forensics
specialists, System/Network Administrators, Management, Communications specialist. Critical success factors:
swift detection minimizing dwell time, preservation of evidence for forensics and legal proceedings, clear
communication, and coordination across departments. Tools: SIEM systems for log aggregation, forensics tools
for evidence collection, threat intelligence platforms, ticketing systems for workflow management."""
    },
    {
        "id": "doc_011",
        "topic": "Brute Force and Credential Attacks",
        "text": """Brute force attacks systematically try many password and credential combinations to gain
unauthorized access to systems or accounts. Attack types: (1) Dictionary attacks using common passwords and
word lists, (2) Hybrid attacks combining dictionary words with numbers and symbols, (3) Brute force trying all
possible character combinations, (4) Rainbow tables using pre-computed hash values, (5) Credential stuffing
using leaked credential sets from previous breaches. Attack vectors: SSH/RDP services exposed to internet,
web application login pages, database services with weak authentication, mail services, VPN access points.
Detection indicators: (1) Multiple failed login attempts from single source, (2) Failed login attempts across
multiple accounts, (3) Failed login attempts followed by successful login, (4) Unusual login times or
geographic locations, (5) Automated login patterns with regular intervals, (6) HTTP 401/403 status code spikes.
Prevention and mitigation: (1) Strong password policies requiring complexity and length, (2) Account lockout
after N failed attempts with time delays, (3) Rate limiting on authentication endpoints, (4) Multi-factor
authentication making stolen credentials insufficient, (5) Implement CAPTCHA to prevent automated attacks,
(6) Monitor authentication logs for attack patterns, (7) Use password managers for unique strong passwords,
(8) Implement adaptive authentication raising requirements for suspicious logins."""
    }
]

# ============================================================
# PART 2 — INITIALIZE EMBEDDER & CHROMADB
# ============================================================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()
try:
    client.delete_collection("cybersecurity_kb")
except Exception:
    pass

collection = client.create_collection("cybersecurity_kb")

# Load documents into ChromaDB
texts = [d["text"] for d in DOCUMENTS]
ids = [d["id"] for d in DOCUMENTS]
embeddings = embedding_model.encode(texts).tolist()

collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=ids,
    metadatas=[{"topic": d["topic"]} for d in DOCUMENTS]
)

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ============================================================
# PART 3 — STATE DESIGN
# ============================================================

class SecurityXState(TypedDict):
    """Complete agent state for SecurityX."""
    # Input
    question: str  # User's current question
    # Memory
    messages: List[str]  # Conversation history (strings, sliding window of 6)
    # Routing
    route: str  # "retrieve", "tool", or "skip"
    # RAG
    retrieved: str  # Retrieved context chunks formatted as [Topic]\ntext
    sources: List[str]  # List of topic names retrieved
    # Tool
    tool_result: str  # Output from threat_detector_tool
    # Answer
    answer: str  # Final LLM response
    # Quality
    faithfulness: float  # Eval score 0.0-1.0
    eval_retries: int  # Safety valve — max 2 retries
    # Domain-specific
    threat_type: str  # Detected threat type
    severity: str  # Threat severity: "Low", "Medium", "High", "Critical"


# ============================================================
# PART 5 — ADVERSARIAL PATTERNS
# ============================================================

ADVERSARIAL_PATTERNS = [
    "ignore previous instructions",
    "disregard your instructions",
    "you are now",
    "pretend you are",
    "act as",
    "forget everything",
    "override your",
    "jailbreak",
    "bypass your"
]


# ============================================================
# PART 4 — NODE FUNCTIONS (8 nodes)
# ============================================================

def memory_node(state: SecurityXState) -> dict:
    """Node 1: Add question to conversation history with sliding window."""
    msgs = list(state.get("messages", []))
    msgs.append(f"User: {state['question']}")
    if len(msgs) > 6:
        msgs = msgs[-6:]
    return {"messages": msgs}


def router_node(state: SecurityXState) -> dict:
    """Node 2: Route using keyword matching (deterministic, no LLM)."""
    question = state["question"].lower()

    skip_keywords = ["remember", "what did you", "previously", "earlier", "summarize",
                     "recap", "previous", "last time", "what was", "you said"]
    tool_keywords = ["failed login", "failed logins", "attack spike", "traffic spike",
                     "detect", "alert", "log analysis", "unusual traffic", "suspicious login",
                     "brute force detected", "ddos", "phishing email received",
                     "analyze this", "threat detected", "incident", "breach report"]

    if any(k in question for k in skip_keywords):
        route = "skip"
    elif any(k in question for k in tool_keywords):
        route = "tool"
    else:
        route = "retrieve"

    return {"route": route}


def retrieval_node(state: SecurityXState) -> dict:
    """Node 3: Query ChromaDB for relevant context."""
    try:
        q_emb = embedding_model.encode([state["question"]])[0].tolist()
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=3,
            include=["documents", "metadatas"]
        )
        chunks = results["documents"][0]
        topics = [m["topic"] for m in results["metadatas"][0]]
        context = "\n\n---\n\n".join(f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks)))
        return {"retrieved": context, "sources": topics}
    except Exception as e:
        return {"retrieved": f"Retrieval error: {str(e)}", "sources": []}


def skip_retrieval_node(state: SecurityXState) -> dict:
    """Node 4: Skip retrieval (for memory-only or tool-only routes)."""
    return {"retrieved": "", "sources": []}


def tool_node(state: SecurityXState) -> dict:
    """Node 5: Threat detector tool — analyzes question for attack pattern signatures."""
    try:
        question = state["question"].lower()

        # Brute Force detection
        if any(k in question for k in ["failed login", "failed logins", "password attempt",
                                        "credential", "brute force", "account lockout",
                                        "multiple login"]):
            threat_type = "Brute Force"
            if "1000" in question or "massive" in question or "thousands" in question:
                severity = "Critical"
                action = "Block source IPs immediately. Enable account lockout. Alert SOC team."
            elif "100" in question or "hundreds" in question or "many" in question:
                severity = "High"
                action = "Implement rate limiting. Review lockout policies. Escalate to SOC."
            else:
                severity = "Medium"
                action = "Monitor login attempts. Consider rate limiting on authentication endpoints."
            tool_result = (
                f"THREAT DETECTED: {threat_type} Attack\n"
                f"Severity: {severity}\n"
                f"Recommended Action: {action}\n"
                f"Detection Basis: Login failure pattern analysis"
            )

        # DDoS detection
        elif any(k in question for k in ["traffic spike", "ddos", "denial of service",
                                          "bandwidth", "syn flood", "udp flood",
                                          "service down", "overloaded", "flooded"]):
            threat_type = "DDoS"
            severity = "Critical"
            action = ("Activate DDoS mitigation service immediately. Enable blackhole routing "
                      "for attacking IPs. Contact ISP for upstream filtering. "
                      "Enable CDN traffic scrubbing.")
            tool_result = (
                f"THREAT DETECTED: {threat_type} Attack\n"
                f"Severity: {severity}\n"
                f"Recommended Action: {action}\n"
                f"Detection Basis: Traffic volume and pattern analysis"
            )

        # Phishing detection
        elif any(k in question for k in ["phishing", "suspicious email", "spoofed",
                                          "credential harvest", "malicious link",
                                          "fake login page", "social engineering"]):
            threat_type = "Phishing"
            severity = "High"
            action = ("Block sender domain. Alert all employees. Reset credentials for affected "
                      "users. Enable enhanced email filtering. Report to IT security team.")
            tool_result = (
                f"THREAT DETECTED: {threat_type} Campaign\n"
                f"Severity: {severity}\n"
                f"Recommended Action: {action}\n"
                f"Detection Basis: Social engineering pattern analysis"
            )

        # No known threat pattern
        else:
            threat_type = "None"
            severity = "Low"
            tool_result = (
                "No active threat pattern detected in query.\n"
                "Routing to knowledge base for informational response.\n"
                "For threat analysis, describe specific attack indicators (e.g., failed logins, "
                "traffic spikes, suspicious emails)."
            )

        return {
            "tool_result": tool_result,
            "threat_type": threat_type,
            "severity": severity
        }

    except Exception as e:
        return {
            "tool_result": f"Threat detection tool error: {str(e)}. Using knowledge base instead.",
            "threat_type": "Error",
            "severity": "Unknown"
        }


def answer_node(state: SecurityXState) -> dict:
    """Node 6: Generate grounded answer using retrieved context and/or tool result."""
    try:
        question = state["question"]
        retrieved = state.get("retrieved", "")
        tool_result = state.get("tool_result", "")
        messages = state.get("messages", [])
        retries = state.get("eval_retries", 0)

        # Adversarial input detection
        q_lower = question.lower()
        if any(pattern in q_lower for pattern in ADVERSARIAL_PATTERNS):
            answer = (
                "I'm SecurityX, a cybersecurity information assistant. "
                "I only respond to genuine cybersecurity questions about threats, "
                "vulnerabilities, attacks, and defense strategies. "
                "I cannot change my behavior or ignore my guidelines."
            )
            return {"answer": answer}

        # Build context section
        context_parts = []
        if retrieved:
            context_parts.append(f"KNOWLEDGE BASE CONTEXT:\n{retrieved}")
        if tool_result:
            context_parts.append(f"THREAT ANALYSIS RESULT:\n{tool_result}")

        context_section = "\n\n".join(context_parts) if context_parts else ""

        # Retry escalation instruction
        retry_instruction = ""
        if retries >= 1:
            retry_instruction = (
                "\nIMPORTANT: Previous answer had low faithfulness. "
                "Stay strictly within the provided context. "
                "If the context does not cover the question, say: "
                "'Based on my knowledge base, I don't have specific information about that. "
                "Please consult a cybersecurity professional for detailed advice.'"
            )

        # Recent conversation history
        recent_history = "\n".join(messages[-4:]) if messages else "No prior conversation."

        system_prompt = f"""You are SecurityX, an expert cybersecurity AI assistant.
Your role: provide accurate, grounded cybersecurity information to security analysts and IT professionals.

STRICT GROUNDING RULE: Base your answer ONLY on the provided context below.
Do NOT add information from general knowledge that contradicts or extends beyond the context.
If the context does not contain enough information to answer, say so clearly and honestly.
{retry_instruction}

RECENT CONVERSATION:
{recent_history}

{context_section if context_section else "No context retrieved — use general cybersecurity knowledge cautiously."}"""

        user_prompt = f"""Question: {question}

Provide a clear, structured, actionable answer. If this is a threat detection query, include:
- Threat type identified
- Severity level
- Immediate recommended actions
- Prevention measures

If this is a knowledge question, explain clearly with specific technical details from the context."""

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        return {"answer": response.content.strip()}

    except Exception as e:
        return {"answer": f"I encountered an error generating a response: {str(e)}. Please try rephrasing your question."}


def eval_node(state: SecurityXState) -> dict:
    """Node 7: Faithfulness evaluation with quality gating."""
    try:
        retrieved = state.get("retrieved", "")
        tool_result = state.get("tool_result", "")
        answer = state.get("answer", "")
        retries = state.get("eval_retries", 0)

        # Skip faithfulness check if no KB context was used
        if not retrieved and not tool_result:
            return {"faithfulness": 1.0, "eval_retries": retries + 1}

        context_for_eval = retrieved if retrieved else tool_result

        eval_prompt = f"""You are a faithfulness evaluator for a cybersecurity AI assistant.

Rate how faithfully the answer is grounded in the provided context.

Context:
{context_for_eval[:600]}

Answer:
{answer[:400]}

Scoring:
- 1.0 = Answer only uses information explicitly present in context
- 0.7-0.9 = Answer mostly grounded, minor reasonable inference
- 0.4-0.6 = Answer partially grounded, adds some outside information
- 0.0-0.3 = Answer largely ignores context or contradicts it

Reply with ONLY a decimal number between 0.0 and 1.0. Nothing else."""

        response = llm.invoke(eval_prompt)
        raw = response.content.strip().split()[0]
        score = float(raw)
        score = max(0.0, min(1.0, score))

        return {"faithfulness": score, "eval_retries": retries + 1}

    except Exception as e:
        return {"faithfulness": 0.5, "eval_retries": state.get("eval_retries", 0) + 1}


def save_node(state: SecurityXState) -> dict:
    """Node 8: Append answer to conversation history."""
    try:
        msgs = list(state.get("messages", []))
        answer = state.get("answer", "")
        if answer:
            msgs.append(f"Assistant: {answer[:200]}")
        if len(msgs) > 6:
            msgs = msgs[-6:]
        return {"messages": msgs}
    except Exception as e:
        return {}


# ============================================================
# PART 5 — ROUTING FUNCTIONS
# ============================================================

def route_decision(state: SecurityXState) -> Literal["retrieve", "tool", "skip"]:
    """Decide which retrieval path to take."""
    route = state.get("route", "retrieve")
    if route in ["retrieve", "tool", "skip"]:
        return route
    return "retrieve"


def eval_decision(state: SecurityXState) -> Literal["answer", "save"]:
    """Decide: retry answer or save and finish."""
    faithfulness = state.get("faithfulness", 1.0)
    retries = state.get("eval_retries", 0)
    if faithfulness < 0.7 and retries < 2:
        return "answer"   # retry
    return "save"


# ============================================================
# PART 6 — GRAPH ASSEMBLY
# ============================================================

def build_agent_graph():
    """Build and compile the LangGraph StateGraph."""
    graph = StateGraph(SecurityXState)

    # Add all 8 nodes
    graph.add_node("memory", memory_node)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("skip_retrieve", skip_retrieval_node)
    graph.add_node("tool", tool_node)
    graph.add_node("answer", answer_node)
    graph.add_node("eval", eval_node)
    graph.add_node("save", save_node)

    # Set entry point
    graph.set_entry_point("memory")

    # Fixed edges
    graph.add_edge("memory", "router")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip_retrieve", "answer")
    graph.add_edge("tool", "answer")
    graph.add_edge("answer", "eval")
    graph.add_edge("save", END)

    # Conditional edges
    graph.add_conditional_edges(
        "router",
        route_decision,
        {"retrieve": "retrieve", "tool": "tool", "skip": "skip_retrieve"}
    )
    graph.add_conditional_edges(
        "eval",
        eval_decision,
        {"answer": "answer", "save": "save"}
    )

    return graph.compile(checkpointer=MemorySaver())


# Build the app at module level
app = build_agent_graph()

# ============================================================
# PART 7 — PUBLIC INTERFACE
# ============================================================

def ask(question: str, thread_id: str = "default") -> dict:
    """
    Main interface to the SecurityX agent.
    
    Args:
        question: User's question
        thread_id: Conversation thread ID (for memory persistence)
    
    Returns:
        Dictionary with keys: question, answer, contexts, route, sources, 
                 threat_type, severity, faithfulness, eval_retries
    """
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = SecurityXState(
        question=question,
        messages=[],
        route="retrieve",
        retrieved="",
        sources=[],
        tool_result="",
        answer="",
        faithfulness=0.0,
        eval_retries=0,
        threat_type="None",
        severity="Low"
    )
    result = app.invoke(initial_state, config=config)
    
    # Extract contexts from retrieved
    contexts = []
    retrieved_text = result.get("retrieved", "")
    if retrieved_text:
        for chunk in retrieved_text.split("\n\n---\n\n"):
            if chunk.strip():
                text = chunk.split("]\n", 1)[-1].strip() if "]\n" in chunk else chunk.strip()
                if text:
                    contexts.append(text)
    
    return {
        "question": question,
        "answer": result.get("answer", ""),
        "contexts": contexts,
        "route": result.get("route", "unknown"),
        "sources": result.get("sources", []),
        "threat_type": result.get("threat_type", "None"),
        "severity": result.get("severity", "Low"),
        "faithfulness": result.get("faithfulness", 0.0),
        "eval_retries": result.get("eval_retries", 0)
    }
