"""
llm/prompt_builder.py
=====================
Builds Phi-3 prompts.
"""

import re

# ── Greeting detection ────────────────────────────────────────
_GREETING_PAT = re.compile(
    r"^(hi+|hey+|hello+|howdy|sup|what'?s up|good (morning|evening|afternoon)|"
    r"how are you|how r u|hru|yo+|greetings|namaste|hola|"
    r"who are you|what are you|what can you do|help me|help|"
    r"thanks|thank you|bye|goodbye|ok|okay|sure|nothing|nm)[\?!\.\s]*$",
    re.I,
)

# ── Low-quality context patterns ──────────────────────────────
_LOW_QUALITY_PAT = re.compile(
    r"(github\.com|this repo|this dataset|kaggle|stackover|pastebin|"
    r"click here|see link|check out|http[s]?://|www\.|\.com/|\.org/)",
    re.I,
)

# ── Per-source truncation limits ──────────────────────────────
_TRUNCATE_LIMIT = {
    "cve":       600,
    "attck":     1400,
    "cyber":     900,
    "vuln_def":  1000,
    "vuln_chat": 99999,
    "custom":    900,
    "conv":      400,
    "default":   900,
}

# ── Per-source system prompts ─────────────────────────────────
_SYSTEM = {
    "cve": (
        "You are CyberShield AI, a CVE vulnerability expert.\n"
        "Answer ONLY using the provided Context.\n"
        "Include: CVE ID, affected product, severity, and remediation if present.\n"
        "If the Context lacks details, say: "
        "'My database has limited info on this CVE. Check https://nvd.nist.gov for full details.'\n"
        "Do NOT invent CVSS scores, dates, or vendor names. Be concise (3-5 sentences)."
    ),
    "attck": (
        "You are CyberShield AI, a MITRE ATT&CK expert.\n"
        "Answer ONLY using the provided Context. Do NOT add any information not in the Context.\n"
        "Write your answer in plain prose — do NOT use labels like 'What it is:' or 'Tactics:'.\n"
        "Structure:\n"
        "  - First sentence: what the technique or tactic is.\n"
        "  - Second sentence: which tactic(s) it belongs to (only if Tactics field is in Context).\n"
        "  - Third sentence: platforms affected (only if Platforms field is in Context).\n"
        "  - Final sentence: how to detect it (only if Detection field is in Context).\n"
        "STRICT RULES:\n"
        "  - If a field is absent from the Context, SKIP IT ENTIRELY.\n"
        "  - Do NOT write 'Not specified', 'Not mentioned', 'Not available', or 'N/A'.\n"
        "  - Do NOT invent technique IDs, tool names, or procedure examples.\n"
        "  - Do NOT start your answer with 'Based on the provided context'.\n"
        "  - Do NOT suggest consulting other resources or experts."
    ),
    "cyber": (
        "You are CyberShield AI, a cybersecurity expert.\n"
        "Answer ONLY using the provided Context.\n"
        "Explain clearly: what it is, how it works, and how to defend against it.\n"
        "If the Context is insufficient, say: "
        "'I don't have enough detail on that topic. Try a more specific question.'\n"
        "Do NOT make up attack tools, statistics, or CVE references.\n"
        "Do NOT suggest consulting external experts or other resources.\n"
        "Be concise (3-5 sentences)."
    ),
    "vuln_def": (
        "You are CyberShield AI, a vulnerability definitions expert.\n"
        "Answer ONLY using the provided Context.\n"
        "Explain the vulnerability clearly: what it is, its impact, and how to mitigate it.\n"
        "If the Context is insufficient, say: "
        "'I don't have enough detail on that vulnerability type.'\n"
        "Do NOT invent examples or CVE IDs.\n"
        "Do NOT suggest consulting external experts or other resources.\n"
        "Be concise (3-5 sentences)."
    ),
    "vuln_chat": (
        "You are CyberShield AI, a vulnerability mitigation expert.\n"
        "The Context below already contains the complete answer.\n"
        "Copy it faithfully — do NOT summarize, shorten, paraphrase, or omit any steps.\n"
        "Format rules:\n"
        "  - Each numbered mitigation step MUST be on its own line.\n"
        "  - Do NOT merge list items into a paragraph.\n"
        "  - Preserve all numbering exactly as in the Context.\n"
        "  - Add a one-sentence intro before the list if helpful.\n"
        "Do NOT add 'consult an expert' or any external resource suggestions.\n"
        "Do NOT invent technical details not present in the Context."
    ),
    "custom": (
        "You are CyberShield AI, a cybersecurity expert.\n"
        "Answer ONLY using the provided Context.\n"
        "If the Context is insufficient, say: "
        "'I don't have enough detail on that topic.'\n"
        "Do NOT invent facts.\n"
        "Do NOT suggest consulting external experts.\n"
        "Be concise (3-5 sentences)."
    ),
    "conv": (
        "You are CyberShield AI, a friendly cybersecurity assistant.\n"
        "Respond naturally and conversationally.\n"
        "Keep it short, friendly, and helpful."
    ),
    "fallback": (
        "You are CyberShield AI, a cybersecurity assistant.\n"
        "The user asked a question but no relevant context was found in the database.\n"
        "Politely tell the user you don't have information on that topic and suggest "
        "they try rephrasing or ask about CVEs, MITRE ATT&CK, or cybersecurity concepts."
    ),
    "default": (
        "You are CyberShield AI, a cybersecurity assistant.\n"
        "Return the answer from the Context exactly as written.\n"
        "Do not summarize. Do not shorten.\n"
        "If the Context does not contain enough information, say: "
        "'I don't have enough information about that in my database.'\n"
        "Do NOT invent CVE IDs, technique IDs, tools, or statistics.\n"
        "Be concise: 2-4 sentences maximum."
    ),
}

WEAK_THRESHOLD = 0.25


def _format_context(text: str) -> str:
    """Convert inline numbered lists to multiline."""
    if not text:
        return text
    text = re.sub(r'(?<!\n)\s+(\d{1,2})\.\s+', r'\n\1. ', text)
    return text.strip()


def build_prompt(question: str, contexts: list, source: str = "default") -> str:
    # ── Greeting shortcut ─────────────────────────────────────
    if _GREETING_PAT.match(question.strip()):
        return (
            f"<|system|>\n{_SYSTEM['conv']}<|end|>\n"
            f"<|user|>\n{question.strip()}<|end|>\n"
            f"<|assistant|>"
        )

    system        = _SYSTEM.get(source, _SYSTEM["default"])
    context_block = _build_context_block(contexts, source)

    # ── Per-source instruction suffix ─────────────────────────
    if source == "vuln_chat":
        instruction = (
            "Return the answer from the Context exactly as written. "
            "Do not summarize. Do not shorten. Do not omit any numbered items. "
            "Put each numbered step on its own line."
        )
    elif source == "attck":
        instruction = (
            "Write 2-4 plain sentences using only what is in the Context. "
            "Skip any field not present in the Context — do not mention it at all. "
            "Do not use section labels. Do not start with 'Based on the provided context'."
        )
    elif source == "cve":
        instruction = (
            "Answer using only the CVE details in the Context. "
            "Do not invent scores, dates, or remediation steps not mentioned."
        )
    else:
        instruction = (
            "Answer based strictly on the Context above. "
            "Do not suggest consulting other sources or experts."
        )

    return (
        f"<|system|>\n{system}<|end|>\n"
        f"<|user|>\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question.strip()}\n\n"
        f"{instruction}<|end|>\n"
        f"<|assistant|>"
    )


def _build_context_block(contexts: list, source: str = "default") -> str:
    if not contexts:
        return "[No context retrieved — state you cannot answer.]"

    truncate_limit = _TRUNCATE_LIMIT.get(source, 900)
    chunks = []

    for item in contexts:
        if isinstance(item, str):
            text, score = item.strip(), 1.0
        elif isinstance(item, dict):
            text  = item.get("answer", "").strip()
            score = float(item.get("score", 1.0))
        else:
            continue

        if not text or len(text) < 15:
            continue

        if score < WEAK_THRESHOLD:
            print(f"[PromptBuilder] Dropped low-score chunk (score={score:.3f})")
            continue

        if _LOW_QUALITY_PAT.search(text[:300]):
            print(f"[PromptBuilder] Dropped low-quality chunk (link/repo reference)")
            continue

        if len(text) > truncate_limit:
            text = text[:truncate_limit] + "...[truncated]"
            print(f"[PromptBuilder] Truncated [{source}] context to {truncate_limit} chars")

        print(f"[PromptBuilder] Context length: {len(text)} chars (source={source})")

        text = _format_context(text)
        chunks.append(text)

        if len(chunks) >= 3:
            break

    if not chunks:
        return "[Context quality too low — tell the user you cannot answer confidently on this topic.]"

    if len(chunks) == 1:
        return chunks[0]

    return "\n\n---\n\n".join(f"[Source {i+1}]\n{c}" for i, c in enumerate(chunks))