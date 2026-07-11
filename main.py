from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from sqlalchemy.orm import Session
from sqlalchemy import func
from passlib.context import CryptContext
import re
import time
from typing import List, Optional

from database import Base, SessionLocal, engine
import models
from vector_db import search_vector_db
from llm.rag_generator import generate_answer

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Chat Assistant API",
    version="6.0.0",
    description="Offline RAG — BM25+TF-IDF+FAISS hybrid + Phi-3 Mini LLM"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ══════════════════════════════════════════════════════════════
# CVE → MITRE LINKING
# ══════════════════════════════════════════════════════════════
CWE_TO_MITRE = {
    "buffer overflow":          [("T1203", "Exploitation for Client Execution")],
    "stack overflow":           [("T1203", "Exploitation for Client Execution")],
    "heap overflow":            [("T1203", "Exploitation for Client Execution")],
    "use after free":           [("T1203", "Exploitation for Client Execution")],
    "out-of-bounds":            [("T1203", "Exploitation for Client Execution")],
    "memory corruption":        [("T1203", "Exploitation for Client Execution")],
    "sql injection":            [("T1190", "Exploit Public-Facing Application")],
    "command injection":        [("T1059", "Command and Scripting Interpreter")],
    "code injection":           [("T1059", "Command and Scripting Interpreter")],
    "remote code execution":    [("T1203", "Exploitation for Client Execution"),
                                 ("T1190", "Exploit Public-Facing Application")],
    "arbitrary code":           [("T1203", "Exploitation for Client Execution")],
    "cross-site scripting":     [("T1189", "Drive-by Compromise")],
    "xss":                      [("T1189", "Drive-by Compromise")],
    "csrf":                     [("T1185", "Browser Session Hijacking")],
    "path traversal":           [("T1083", "File and Directory Discovery")],
    "directory traversal":      [("T1083", "File and Directory Discovery")],
    "file inclusion":           [("T1190", "Exploit Public-Facing Application")],
    "ssrf":                     [("T1090", "Proxy")],
    "open redirect":            [("T1566", "Phishing")],
    "authentication bypass":    [("T1078", "Valid Accounts")],
    "improper authentication":  [("T1078", "Valid Accounts")],
    "privilege escalation":     [("T1068", "Exploitation for Privilege Escalation")],
    "improper authorization":   [("T1548", "Abuse Elevation Control Mechanism")],
    "information disclosure":   [("T1082", "System Information Discovery")],
    "sensitive information":    [("T1552", "Unsecured Credentials")],
    "hardcoded credential":     [("T1552", "Unsecured Credentials")],
    "cleartext":                [("T1040", "Network Sniffing")],
    "denial of service":        [("T1499", "Endpoint Denial of Service")],
    "resource exhaustion":      [("T1499", "Endpoint Denial of Service")],
    "deserialization":          [("T1203", "Exploitation for Client Execution")],
    "xml external entity":      [("T1190", "Exploit Public-Facing Application")],
    "template injection":       [("T1190", "Exploit Public-Facing Application")],
    "man-in-the-middle":        [("T1557", "Adversary-in-the-Middle")],
    "dns spoofing":             [("T1557", "Adversary-in-the-Middle")],
    "arp spoofing":             [("T1557", "Adversary-in-the-Middle")],
}

def link_cve_to_mitre(cve_text: str) -> list:
    text_lower = cve_text.lower()
    found = {}
    for keyword, techniques in CWE_TO_MITRE.items():
        if keyword in text_lower:
            for tech_id, tech_name in techniques:
                if tech_id not in found:
                    found[tech_id] = {
                        "tech_id":         tech_id,
                        "tech_name":       tech_name,
                        "matched_keyword": keyword,
                    }
    return list(found.values())


# ── REQUEST MODELS ────────────────────────────────────────────
class RegisterRequest(BaseModel):
    username: str
    password: str

    @validator("username")
    def username_valid(cls, v):
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters")
        if not re.match(r"^[a-zA-Z0-9_]+$", v):
            raise ValueError("Username can only contain letters, numbers, underscores")
        return v

    @validator("password")
    def password_valid(cls, v):
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v


class LoginRequest(BaseModel):
    username: str
    password: str


class ContextMessage(BaseModel):
    user: str
    bot:  str


class ChatRequest(BaseModel):
    username: str
    password: str
    message:  str
    db_type:  str = "hybrid"
    context:  Optional[List[ContextMessage]] = []

    @validator("message")
    def message_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Message cannot be empty")
        if len(v) > 1000:
            raise ValueError("Message too long, max 1000 characters")
        return v.strip()

    @validator("db_type")
    def db_type_valid(cls, v):
        return "hybrid"


class ForgotPasswordRequest(BaseModel):
    username: str
    new_password: str

    @validator("new_password")
    def password_valid(cls, v):
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v


# ── ROUTES ────────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "message":  "AI Chat Assistant API is running",
        "version":  "6.0.0",
        "search":   "BM25 + TF-IDF + FAISS hybrid + Phi-3 Mini LLM",
        "datasets": ["Conversation", "Cybersecurity Q&A", "AI/ML Q&A",
                     "CVE", "MITRE ATT&CK"],
        "features": ["conversation memory", "context-aware search",
                     "CVE lookup", "ATT&CK technique lookup",
                     "CVE→MITRE linking", "Phi-3 answer generation"]
    }


@app.get("/health")
def health():
    return {"status": "ok", "search": "hybrid-bm25-tfidf-faiss", "llm": "phi3-gguf"}


@app.post("/signup", status_code=status.HTTP_201_CREATED)
def signup(data: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(models.User).filter(
        models.User.username == data.username).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="Username already exists")
    new_user = models.User(
        username=data.username,
        password=hash_password(data.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "Signup successful", "user_id": new_user.id}


@app.post("/login")
def login(data: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(
        models.User.username == data.username).first()
    if not user or not verify_password(data.password, user.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid username or password")
    return {"message": "Login successful"}


@app.post("/forgot-password")
def forgot_password(data: ForgotPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(
        models.User.username == data.username).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="User not found")
    user.password = hash_password(data.new_password)
    db.commit()
    return {"message": "Password updated successfully"}


@app.post("/chat")
def chat(data: ChatRequest, db: Session = Depends(get_db)):
    # ── Auth ──────────────────────────────────────────────────
    user = db.query(models.User).filter(
        models.User.username == data.username).first()
    if not user or not verify_password(data.password, user.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid login credentials")

    context = [{"user": c.user, "bot": c.bot} for c in (data.context or [])][-3:]

    start = time.time()

    # ── Step 1: Retrieve best answer from vector DB ────────────
    result     = search_vector_db(data.message, db_type="hybrid", context=context)
    raw_answer = result["answer"]
    source     = result.get("source", "hybrid")
    score      = result.get("score", 0.0)

    # ── Step 2: CVE → MITRE linking ───────────────────────────
    mitre_techniques = []
    if source == "cve":
        mitre_techniques = link_cve_to_mitre(raw_answer)

    # ── Step 3: Build contexts list for Phi-3 ─────────────────
    contexts = [raw_answer]
    if mitre_techniques:  
        mitre_text = "Related MITRE ATT&CK techniques:\n" + "\n".join(
            f"  - {t['tech_id']}: {t['tech_name']}"
            for t in mitre_techniques
        )
        contexts.append(mitre_text)
    
    # ── Step 4: Generate fluent answer via Phi-3 ──────────────
    bot_reply = generate_answer(data.message, contexts, source=source)

    # If llama-server not running, fall back to raw answer
    if bot_reply.startswith("LLM Error:"):
        print(f"[Chat] Phi-3 unavailable ({bot_reply}) — using raw answer")
        bot_reply = raw_answer

    elapsed = round(time.time() - start, 3)

    # ── Step 5: Save to DB ────────────────────────────────────
    new_message = models.Message(
        username      = data.username,
        user_message  = data.message,
        bot_response  = bot_reply,
        score         = str(round(score, 4)),
        db_used       = source,
        response_time = str(elapsed)
    )
    db.add(new_message)
    db.commit()

    return {
        "reply":            bot_reply,
        "score":            str(round(score, 4)),
        "db_used":          source,
        "mitre_techniques": mitre_techniques,
        "response_time":    elapsed
    }


@app.get("/messages/{username}")
def get_messages(username: str, db: Session = Depends(get_db)):
    messages = (
        db.query(models.Message)
        .filter(models.Message.username == username)
        .order_by(models.Message.created_at.asc())
        .all()
    )
    return messages


@app.get("/admin/stats")
def admin_stats(db: Session = Depends(get_db)):
    total_users    = db.query(models.User).count()
    total_messages = db.query(models.Message).count()

    all_messages = db.query(models.Message).all()
    source_counts = {}
    for m in all_messages:
        src = m.db_used or "hybrid"
        source_counts[src] = source_counts.get(src, 0) + 1

    faiss_count = (
        source_counts.get("faiss",  0) + source_counts.get("hybrid", 0) +
        source_counts.get("cyber",  0) + source_counts.get("conv",   0) +
        source_counts.get("ai",     0) + source_counts.get("cve",    0) +
        source_counts.get("attck",  0)
    )
    chroma_count = source_counts.get("chroma", 0)
    milvus_count = source_counts.get("milvus", 0)
    qdrant_count = source_counts.get("qdrant", 0)

    top_questions = (
        db.query(
            models.Message.user_message,
            func.count(models.Message.user_message).label("count")
        )
        .group_by(models.Message.user_message)
        .order_by(func.count(models.Message.user_message).desc())
        .limit(5)
        .all()
    )

    def avg_time(db_name):
        times = [
            float(m.response_time)
            for m in all_messages
            if m.db_used == db_name and m.response_time
        ]
        return round(sum(times) / len(times), 3) if times else 0

    return {
        "total_users":    total_users,
        "total_messages": total_messages,
        "db_usage": {
            "faiss":  faiss_count,
            "chroma": chroma_count,
            "milvus": milvus_count,
            "qdrant": qdrant_count,
        },
        "avg_response_time": {
            "faiss":  avg_time("faiss"),
            "chroma": avg_time("chroma"),
            "milvus": avg_time("milvus"),
            "qdrant": avg_time("qdrant"),
        },
        "top_questions": [
            {"question": q.user_message, "count": q.count}
            for q in top_questions
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)