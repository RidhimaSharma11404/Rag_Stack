"""
vector_db.py — Hybrid BM25 + TF-IDF + FAISS Semantic Search (v6)
=================================================================
Architecture
------------
Four separate FAISS indexes (one per source group) share a single
MiniLM model instance loaded exactly once.

Search pipeline (every query goes through ALL layers):
  Layer 0 : Query normalisation  – lower-case, strip noise chars
  Layer 1 : Alias / abbreviation expansion  – "cyber" → "cybersecurity …"
  Layer 2 : Fuzzy spell-correct  – "phising" → "phishing"  (difflib)
  Layer 3 : Synonym rewrite      – controlled vocab mapping
  Layer 4 : BM25 + TF-IDF hybrid keyword search  (whole corpus)
  Layer 5 : Per-source FAISS semantic search  (always runs, not rescue-only)
  Layer 6 : Score fusion         – merge + re-rank both result lists
  Layer 7 : Answer validation    – relevance check before returning
  Layer 8 : Graceful degradation – informative fallback (never vague)
  Layer 9 : Phi-3 Mini LLM       – rephrase retrieved answer fluently (optional)

Source groups
-------------
  cve   → CVE records
  attck → MITRE ATT&CK
  cyber → Cybersecurity Q&A + Custom + Vuln Definitions + Vuln Chatbot
  other → AI/ML Q&A + Conversation
"""

import os, re, math, pickle, hashlib, time, warnings
from difflib import get_close_matches
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cve_loader  import load_cve_dataset
from stix_loader import load_stix_dataset

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# PHI-3 INTEGRATION — Layer 9
# ══════════════════════════════════════════════════════════════
USE_PHI3 = True
_phi3 = None

def _get_phi3():
    global _phi3
    if _phi3 is None and USE_PHI3:
        try:
            from phi3_generator import get_phi3
            _phi3 = get_phi3()
            print("[VectorDB] Phi-3 generator loaded ✓")
        except FileNotFoundError as e:
            print(f"[VectorDB] WARNING: {e}")
            print("[VectorDB] Phi-3 disabled — run download_phi3_for_offline.py first.")
        except Exception as e:
            print(f"[VectorDB] WARNING: Phi-3 failed to load ({e}). Using raw answers.")
    return _phi3

# ══════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
AI_CSV           = os.path.join(BASE_DIR, "qa_dataset.csv")
CONV_CSV         = os.path.join(BASE_DIR, "Conversation.csv")
CYBER_CSV        = os.path.join(BASE_DIR, "cybersecurity-qa-v1.csv")
CUSTOM_CSV       = os.path.join(BASE_DIR, "custom_cyber.csv")

VULN_CHAT_DATASET_CSV   = os.path.join(BASE_DIR, "vulnerabilities_chatbot_dataset_Chatbot_Dataset.csv")
VULN_CHAT_INDEX_CSV     = os.path.join(BASE_DIR, "vulnerabilities_chatbot_dataset_Vulnerability_Index.csv")
VULN_DEF_QA_CSV         = os.path.join(BASE_DIR, "vulnerability_definitions_Chatbot_QA_-_Definitions.csv")
# ✅ FIX 1: Corrected filename typo ("Vulnerations" → "Definitions")
VULN_DEF_DEFINITIONS_CSV = os.path.join(BASE_DIR, "vulnerability_definitions_Vulnerability_Definitions.csv")

CVE_REPO         = os.path.join(BASE_DIR, "cvelistV5-main", "cvelistV5-main")
STIX_REPO        = os.path.join(BASE_DIR, "attack-stix-data-master", "attack-stix-data-master")
CACHE_DIR        = os.path.join(BASE_DIR, "search_cache")
CACHE_FILE       = os.path.join(CACHE_DIR, "hybrid_v6.pkl")
HASH_FILE        = os.path.join(CACHE_DIR, "hash_v6.txt")
FAISS_CVE_FILE   = os.path.join(CACHE_DIR, "faiss_cve_v6.pkl")
FAISS_ATTCK_FILE = os.path.join(CACHE_DIR, "faiss_attck_v6.pkl")
FAISS_CYBER_FILE = os.path.join(CACHE_DIR, "faiss_cyber_v6.pkl")
FAISS_OTHER_FILE = os.path.join(CACHE_DIR, "faiss_other_v6.pkl")
os.makedirs(CACHE_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# TUNING CONSTANTS
# ══════════════════════════════════════════════════════════════
BM25_K1, BM25_B = 1.5, 0.75
BM25_WEIGHT     = 0.30
TFIDF_WEIGHT    = 0.20
FAISS_WEIGHT    = 0.50

FAISS_MAX  = {"cve": 60_000, "attck": 5_000, "cyber": 15_000, "other": 30_000}
FAISS_BATCH = 512

SOURCE_GROUP = {
    "cve": "cve", "attck": "attck",
    "cyber": "cyber", "custom": "cyber",
    "vuln_def": "cyber", "vuln_chat": "cyber",
    "ai": "other", "conv": "other",
}

SOURCE_BOOST = {
    "custom":    1.35, "vuln_def":  1.40, "vuln_chat": 1.40,
    "cve":       1.20, "attck":     1.25, "cyber":     1.20,
    "ai":        0.90, "conv":      1.00,
}

MIN_SCORE = {
    "custom":    0.07, "vuln_def":  0.06, "vuln_chat": 0.06,
    "cve":       0.08, "attck":     0.08, "cyber":     0.09,
    "ai":        0.10, "conv":      0.06,
}

FAISS_SIM_FLOOR = 0.15
CYBER_MARGIN    = 0.18

# ── Validation: answer must contain ≥1 of these per-source signals ──
# ✅ FIX 3 & 4: Expanded vuln_chat and vuln_def signals to cover
#    mitigation-specific vocabulary (sanitize, parameterize, etc.)
_ANSWER_SIGNALS = {
    "cve":       re.compile(
        r"\b(cve|vulnerability|cvss|affected|patch|severity|"
        r"exploit|disclosure|vendor|product)\b", re.I),
    "attck":     re.compile(
        r"\b(technique|tactic|mitre|att&?ck|procedure|adversary|"
        r"detection|mitigation|sub-technique|group|software)\b", re.I),
    "cyber":     re.compile(
        r"\b(security|attack|malware|threat|encrypt|firewall|"
        r"intrusion|phish|ransomware|vulnerability|exploit|"
        r"authentication|authorization|network|protocol)\b", re.I),
    "custom":    re.compile(
        r"\b(security|attack|malware|threat|encrypt|firewall|"
        r"intrusion|phish|ransomware|vulnerability|exploit)\b", re.I),
    # ✅ Expanded: added sanitize, parameterize, prepared, validate, input,
    #    injection, mitigation, prevent, remediat, update, patch, configure
    "vuln_def":  re.compile(
        r"\b(vulnerability|security|attack|authentication|application|"
        r"method|access|inject|overflow|exposure|weakness|flaw|"
        r"sanitize|parameteriz|prepared|statement|validate|input|"
        r"mitigation|prevent|remediat|patch|update|configure|"
        r"encrypt|firewall|control|threat|risk|defense|protect)\b", re.I),
    # ✅ Expanded: same broad mitigation vocabulary
    "vuln_chat": re.compile(
        r"\b(vulnerability|security|attack|mitigation|implement|"
        r"authentication|protect|prevent|weak|exploit|"
        r"sanitize|parameteriz|prepared|statement|validate|input|"
        r"remediat|patch|update|configure|encrypt|firewall|"
        r"control|threat|risk|defense|injection|overflow|"
        r"best practice|recommend|should|must|ensure|avoid)\b", re.I),
    "ai":        re.compile(
        r"\b(model|train|neural|learning|data|algorithm|"
        r"inference|accuracy|parameter|gradient|loss)\b", re.I),
    "conv":      re.compile(r".{20,}", re.S),
}

# ══════════════════════════════════════════════════════════════
# LAYER 0 — QUERY NORMALISATION
# ══════════════════════════════════════════════════════════════
_NOISE  = re.compile(r"[^\w\s\-\.&/]")
_MULTI  = re.compile(r"\s{2,}")
_REPEAT = re.compile(r"(.)\1{2,}")

def _normalise(text: str) -> str:
    text = text.lower().strip()
    text = _NOISE.sub(" ", text)
    text = _MULTI.sub(" ", text)
    text = _REPEAT.sub(r"\1\1", text)
    return text

# ══════════════════════════════════════════════════════════════
# LAYER 1 — ALIAS / ABBREVIATION EXPANSION
# ══════════════════════════════════════════════════════════════
_ALIAS: dict[str, str] = {
    "cyber":        "cybersecurity security vulnerability attack defense",
    "infosec":      "information security cybersecurity",
    "appsec":       "application security web vulnerability owasp",
    "netsec":       "network security firewall intrusion packet",
    "cloudsec":     "cloud security aws azure gcp misconfiguration",
    "devsecops":    "devops security pipeline secure development",
    "secops":       "security operations center monitoring incident",
    "redteam":      "red team offensive penetration test adversary simulation",
    "blueteam":     "blue team defensive detection incident response soc",
    "purpleteam":   "purple team red blue collaboration detection",
    "vuln":         "vulnerability cve exploit security flaw weakness",
    "vulns":        "vulnerabilities cve exploits security flaws",
    "cves":         "cve common vulnerabilities exposures list",
    "nvd":          "national vulnerability database nist cve",
    "cvss":         "common vulnerability scoring system severity score",
    "0day":         "zero day vulnerability exploit unpatched",
    "zeroday":      "zero day vulnerability exploit unpatched",
    "attck":        "mitre att&ck technique tactic adversary procedure",
    "mitre":        "mitre att&ck technique tactic framework",
    "ttp":          "tactics techniques procedures mitre att&ck adversary",
    "ioc":          "indicator of compromise threat detection hash ip domain",
    "iocs":         "indicators of compromise threat detection",
    "apt":          "advanced persistent threat nation state espionage",
    "ta":           "threat actor group nation state criminal",
    "stix":         "structured threat information expression mitre att&ck",
    "taxii":        "trusted automated exchange indicator information",
    "sqli":         "sql injection database attack query manipulation",
    "xss":          "cross site scripting web attack javascript injection",
    "csrf":         "cross site request forgery web attack session",
    "ssrf":         "server side request forgery cloud internal attack",
    "lfi":          "local file inclusion path traversal directory",
    "rfi":          "remote file inclusion web vulnerability code execution",
    "rce":          "remote code execution arbitrary command vulnerability",
    "idor":         "insecure direct object reference access control flaw",
    "xxe":          "xml external entity injection vulnerability",
    "ssti":         "server side template injection web vulnerability",
    "deserialization": "insecure deserialization object injection vulnerability",
    "clickjack":    "clickjacking ui redress attack iframe",
    "openredirect": "open redirect phishing url manipulation",
    "bof":          "buffer overflow stack heap memory corruption exploit",
    "bufover":      "buffer overflow memory corruption exploit",
    "heapspray":    "heap spray memory exploit browser",
    "uaf":          "use after free memory corruption heap exploit",
    "rop":          "return oriented programming exploit bypass",
    "privesc":      "privilege escalation local exploit elevation admin",
    "privesec":     "privilege escalation local exploit elevation",
    "lpe":          "local privilege escalation kernel exploit",
    "ddos":         "distributed denial of service attack flood amplification",
    "dos":          "denial of service attack resource exhaustion",
    "mitm":         "man in the middle attack eavesdrop intercept",
    "arp":          "arp spoofing poisoning man in the middle network",
    "dns":          "dns spoofing hijacking poisoning domain resolution",
    "sniff":        "packet sniffing network capture wireshark eavesdrop",
    "recon":        "reconnaissance footprinting information gathering osint",
    "osint":        "open source intelligence gathering reconnaissance",
    "pivoting":     "network pivoting lateral movement tunneling",
    "c2":           "command and control malware botnet infrastructure",
    "c&c":          "command and control malware botnet infrastructure",
    "rat":          "remote access trojan malware backdoor remote control",
    "rootkit":      "rootkit stealth malware kernel persistence",
    "keylogger":    "keylogger credential theft input capture",
    "dropper":      "dropper malware loader stager payload",
    "packer":       "packer obfuscation malware evasion",
    "wiper":        "wiper destructive malware disk overwrite",
    "cryptojack":   "cryptojacking mining malware cpu resource",
    "spyware":      "spyware surveillance monitoring credential theft",
    "soc":          "security operations center analyst monitoring alert",
    "siem":         "security information event management log analysis alert",
    "edr":          "endpoint detection response malware behavioral",
    "xdr":          "extended detection response cross layer visibility",
    "waf":          "web application firewall rule block injection",
    "ids":          "intrusion detection system signature anomaly alert",
    "ips":          "intrusion prevention system inline block attack",
    "dlp":          "data loss prevention exfiltration sensitive data",
    "ueba":         "user entity behavior analytics anomaly insider",
    "soar":         "security orchestration automation response playbook",
    "deception":    "deception technology honeypot canary token trap",
    "mfa":          "multi factor authentication two factor otp token",
    "2fa":          "two factor authentication multi factor otp",
    "iam":          "identity access management user permission role",
    "rbac":         "role based access control permission authorization",
    "abac":         "attribute based access control policy authorization",
    "pam":          "privileged access management admin credential vault",
    "sso":          "single sign on federation saml oauth identity",
    "saml":         "security assertion markup language sso federation",
    "oauth":        "open authorization token delegation api access",
    "jwt":          "json web token authentication authorization bearer",
    "ldap":         "lightweight directory access protocol active directory",
    "ad":           "active directory windows domain authentication ldap",
    "pki":          "public key infrastructure certificate authority tls ssl",
    "tls":          "transport layer security ssl encryption protocol https",
    "ssl":          "secure sockets layer tls encryption certificate https",
    "aes":          "advanced encryption standard symmetric block cipher",
    "rsa":          "rsa asymmetric public key encryption signature",
    "sha":          "sha hash algorithm digest integrity",
    "md5":          "md5 hash collision weak digest",
    "hmac":         "hash based message authentication code integrity",
    "ec":           "elliptic curve cryptography ecc key exchange",
    "vpn":          "virtual private network tunnel encryption remote access",
    "tor":          "tor anonymity onion routing privacy dark web",
    "smb":          "server message block windows file share lateral movement",
    "rdp":          "remote desktop protocol windows lateral movement brute",
    "ssh":          "secure shell remote access tunneling command",
    "ftp":          "file transfer protocol cleartext credential",
    "http":         "hypertext transfer protocol web request response",
    "https":        "http secure tls web encrypted traffic",
    "icmp":         "internet control message protocol ping network",
    "tcp":          "transmission control protocol connection network",
    "pentest":      "penetration testing offensive security assessment",
    "metasploit":   "metasploit framework exploit payload post exploitation",
    "burp":         "burp suite web proxy intercept vulnerability scanner",
    "nmap":         "nmap network scanner port service reconnaissance",
    "nessus":       "nessus vulnerability scanner assessment",
    "cobalt":       "cobalt strike red team c2 lateral movement beacon",
    "bloodhound":   "bloodhound active directory privilege escalation path",
    "mimikatz":     "mimikatz credential dumping lsass windows password",
    "hashcat":      "hashcat password cracking hash brute force",
    "ml":           "machine learning model training inference prediction",
    "dl":           "deep learning neural network architecture training",
    "llm":          "large language model gpt transformer generative ai",
    "nlp":          "natural language processing text classification sentiment",
    "cv":           "computer vision image recognition cnn object detection",
    "rl":           "reinforcement learning reward agent policy optimization",
    "gan":          "generative adversarial network image synthesis deepfake",
    "bert":         "bert transformer pretrained language model nlp",
    "gpt":          "gpt generative pretrained transformer language model",
    "cnn":          "convolutional neural network image classification vision",
    "rnn":          "recurrent neural network sequence time series lstm",
    "lstm":         "long short term memory recurrent sequence model",
    "xai":          "explainable artificial intelligence interpretability model",
    "mlops":        "machine learning operations deployment pipeline",
    "fedlearn":     "federated learning privacy distributed model training",
    "cybersecuirty":    "cybersecurity security vulnerability attack",
    "cyberscurity":     "cybersecurity security vulnerability attack",
    "cybersecurity":    "cybersecurity security vulnerability attack defense",
    "malwre":           "malware virus trojan ransomware",
    "randsomware":      "ransomware encryption extortion malware",
    "ransomwhere":      "ransomware encryption extortion malware",
    "phising":          "phishing email social engineering credential",
    "phsihing":         "phishing email social engineering credential",
    "firwall":          "firewall network security packet filter rule",
    "encrytion":        "encryption cryptography data protection cipher",
    "authetication":    "authentication login password credential verify",
    "vulnerabilty":     "vulnerability cve security flaw weakness exploit",
    "priviledge":       "privilege escalation access control elevation",
    "authorizaton":     "authorization access control permission role",
    "penetartion":      "penetration testing offensive security assessment",
    "pentesting":       "penetration testing offensive security assessment",
}

_ALIAS_RE: list[tuple[re.Pattern, str]] = [
    (re.compile(rf"\b{re.escape(k)}\b", re.I), v)
    for k, v in _ALIAS.items()
]

def _alias_expand(text: str) -> str:
    additions: list[str] = []
    for pat, expansion in _ALIAS_RE:
        if pat.search(text):
            additions.append(expansion)
    return f"{text} {' '.join(additions)}" if additions else text

# ══════════════════════════════════════════════════════════════
# LAYER 2 — FUZZY SPELL CORRECTION
# ══════════════════════════════════════════════════════════════
_SPELL_VOCAB: list[str] = sorted(set(
    list(_ALIAS.keys()) + [
        "cybersecurity","vulnerability","authentication","authorization",
        "encryption","firewall","malware","ransomware","phishing","exploit",
        "intrusion","penetration","reconnaissance","lateral","escalation",
        "exfiltration","persistence","detection","mitigation","remediation",
        "incident","forensics","threat","adversary","payload","backdoor",
        "trojan","rootkit","botnet","spyware","adware","worm","virus",
        "credential","certificate","signature","anomaly","baseline","audit",
        "compliance","regulation","policy","governance","risk","assessment",
    ]
))

def _spell_correct(text: str) -> str:
    tokens    = text.split()
    corrected = []
    for tok in tokens:
        if len(tok) < 5 or tok in _SPELL_VOCAB:
            corrected.append(tok)
            continue
        matches = get_close_matches(tok, _SPELL_VOCAB, n=1, cutoff=0.82)
        if matches and matches[0] != tok:
            corrected.append(matches[0])
            print(f"  [SpellCorrect] '{tok}' → '{matches[0]}'")
        else:
            corrected.append(tok)
    return " ".join(corrected)

# ══════════════════════════════════════════════════════════════
# LAYER 3 — SYNONYM REWRITE
# ══════════════════════════════════════════════════════════════
_SYNONYM: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bhow (do i|can i|to)\b", re.I),      "how to"),
    (re.compile(r"\bwhat is (a |an |the )?\b", re.I),   "what is "),
    (re.compile(r"\bexplain (me )?\b", re.I),            "explain "),
    (re.compile(r"\btell me about\b", re.I),             "explain "),
    (re.compile(r"\bdefine\b", re.I),                    "what is "),
    (re.compile(r"\bdifference between\b", re.I),        "compare "),
    (re.compile(r"\bvs\.?\b", re.I),                     "versus "),
    (re.compile(r"\bhow does .+ work\b", re.I),          lambda m: m.group()),
    (re.compile(r"\bcode exec(ution)?\b", re.I),         "remote code execution"),
    (re.compile(r"\bpass[- ]the[- ]hash\b", re.I),       "pass the hash credential lateral"),
    (re.compile(r"\bpass[- ]the[- ]ticket\b", re.I),     "kerberos ticket lateral movement"),
    (re.compile(r"\bgolden ticket\b", re.I),              "kerberos golden ticket forging"),
    (re.compile(r"\bsilver ticket\b", re.I),              "kerberos silver ticket forging"),
    (re.compile(r"\bdcsync\b", re.I),                    "dcsync credential dumping active directory"),
    (re.compile(r"\blsass\b", re.I),                     "lsass credential dumping mimikatz windows"),
    (re.compile(r"\bsupply chain\b", re.I),               "supply chain attack software compromise"),
    (re.compile(r"\bsocial eng(ineer)?(ing)?\b", re.I),  "social engineering phishing pretexting"),
    (re.compile(r"\bwater(ing)?[- ]hole\b", re.I),        "watering hole attack web compromise"),
    (re.compile(r"\bspear[- ]?phish\b", re.I),            "spearphishing targeted email attack"),
    (re.compile(r"\bvishing\b", re.I),                    "voice phishing phone social engineering"),
    (re.compile(r"\bsmishing\b", re.I),                   "sms phishing text message social engineering"),
    (re.compile(r"\bbrute[- ]?force\b", re.I),            "brute force password cracking credential"),
    (re.compile(r"\bcred(ential)?[- ]?stuff\b", re.I),   "credential stuffing password spray attack"),
    (re.compile(r"\bpwd?\b", re.I),                       "password credential"),
    (re.compile(r"\bauth\b", re.I),                       "authentication"),
    (re.compile(r"\bperms?\b", re.I),                     "permissions access control"),
    (re.compile(r"\badmin\b", re.I),                      "administrator privilege"),
]

def _synonym_rewrite(text: str) -> str:
    for pat, repl in _SYNONYM:
        if callable(repl):
            text = pat.sub(repl, text)
        else:
            text = pat.sub(repl, text)
    return text

# ══════════════════════════════════════════════════════════════
# FULL QUERY PRE-PROCESSING  (Layers 0-3 combined)
# ══════════════════════════════════════════════════════════════
_FOLLOWUP = re.compile(
    r"^(tell me more|explain (it|that|this|more)|what about (it|that)?|"
    r"how does (it|that|this) work|more (details?|info)|elaborate|"
    r"give me (more|an example)|can you (elaborate|expand)|what else|"
    r"and\??|so\??|why\??|how\??)\.?$", re.I,
)
_CYBER_KW = re.compile(
    r"\b(cve-?\d{4}|t\d{4}|exploit|malware|ransomware|phishing|"
    r"sql.?inject|xss|csrf|overflow|privilege|lateral|credential|"
    r"firewall|encryption|vulnerability|threat|attack|backdoor|"
    r"rootkit|trojan|botnet|apt|pentest|siem|soc|ioc|ttp)\b", re.I,
)

_CVE_ID_PAT       = re.compile(r"cve-\d{4}-\d+",         re.I)
_TECHNIQUE_ID_PAT = re.compile(r"\bt\d{4}(\.\d{3})?\b",  re.I)

def _preprocess(query: str, context: list = []) -> tuple[str, str]:
    q = _normalise(query)
    q = _spell_correct(q)
    sem_query = q

    _q_stripped = sem_query.strip()
    _wc = len(_q_stripped.split())
    _already_question = bool(re.match(
        r"^(what|how|why|when|where|who|explain|define|describe|tell)", _q_stripped, re.I
    ))
    if (_wc <= 3
            and not _already_question
            and not _CVE_ID_PAT.search(_q_stripped)
            and not _TECHNIQUE_ID_PAT.search(_q_stripped)):
        sem_query = f"what is {_q_stripped} and how does it work"
        print(f"  [SemReframe] '{_q_stripped}' → '{sem_query}'")

    q = _alias_expand(q)
    q = _synonym_rewrite(q)

    if context and len(query.split()) <= 10 and _FOLLOWUP.match(query.strip()):
        prev_q   = context[-1].get("user", "")
        prev_bot = context[-1].get("bot",  "")
        kw = " ".join(dict.fromkeys(_CYBER_KW.findall(prev_bot[:500])[:5]))
        base = prev_q or ""
        q = f"{base} {q} {kw}".strip() if (base or kw) else q
        sem_query = f"{base} {sem_query}".strip() if base else sem_query

    return q, sem_query

# ══════════════════════════════════════════════════════════════
# TOKENIZER
# ══════════════════════════════════════════════════════════════
_TOK = re.compile(r"[a-z0-9]+(?:[-\.][a-z0-9]+)*")
def _tok(text: str) -> list[str]:
    return _TOK.findall(text.lower()) if text else []

# ══════════════════════════════════════════════════════════════
# DATASET LOADERS
# ══════════════════════════════════════════════════════════════
def _load_conversation():
    if not os.path.exists(CONV_CSV):
        print("[VectorDB] Conversation          : NOT FOUND"); return [], []
    df = pd.read_csv(CONV_CSV, dtype=str)
    df["question"] = df["question"].fillna("").str.strip()
    df["answer"]   = df["answer"].fillna("").str.strip()
    df = df[(df["question"].str.len() >= 2) & (df["answer"].str.len() >= 2)]
    print(f"[VectorDB] Conversation          : {len(df):>7,} rows")
    return df["question"].tolist(), df["answer"].tolist()

def _load_ai_qa():
    if not os.path.exists(AI_CSV):
        print("[VectorDB] AI Q&A                : NOT FOUND"); return [], []
    df = pd.read_csv(AI_CSV, dtype=str)
    df["input"]  = df["input"].fillna("").str.strip()
    df["target"] = df["target"].fillna("").str.strip()
    mask = (
        (df["input"].str.len()   >= 8) &
        (~df["input"].str.startswith('"')) &
        (df["target"].str.len()  >= 60) &
        (~df["target"].isin(["", "nan", "NaN"]))
    )
    df = df[mask]
    print(f"[VectorDB] AI/ML Q&A             : {len(df):>7,} rows")
    return df["input"].tolist(), df["target"].tolist()

def _load_cybersecurity():
    if not os.path.exists(CYBER_CSV):
        print("[VectorDB] Cybersecurity         : NOT FOUND"); return [], [], []
    df = pd.read_csv(CYBER_CSV, dtype=str)
    df["question"] = df["question"].fillna("")
    df["answer"]   = df["answer"].fillna("").str.strip()
    titles  = df["question"].str.split("\n").str[0].str.strip()
    bodies  = df["question"].str.split("\n").str[1:].str.join(" ").str.strip()
    mask    = (titles.str.len() >= 8) & (df["answer"].str.len() >= 20)
    titles  = titles[mask].reset_index(drop=True)
    bodies  = bodies[mask].reset_index(drop=True)
    answers = df["answer"][mask].reset_index(drop=True)
    docs    = (titles + " " + bodies.str[:400] + " " + answers.str[:400]).tolist()
    print(f"[VectorDB] Cybersecurity Q&A     : {len(titles):>7,} rows")
    return titles.tolist(), answers.tolist(), docs

def _load_cve():
    for path in [CVE_REPO, os.path.join(BASE_DIR, "cvelistV5-main")]:
        if os.path.isdir(path):
            try:
                df = load_cve_dataset(path, max_records=60000)
                if not df.empty:
                    print(f"[VectorDB] CVE Records           : {len(df):>7,} rows")
                    return df["question"].tolist(), df["answer"].tolist()
            except Exception as e:
                print(f"[VectorDB] CVE error: {e}")
    print("[VectorDB] CVE Records           :       0 rows (folder not found)")
    return [], []

def _load_attck():
    for path in [STIX_REPO, os.path.join(BASE_DIR, "attack-stix-data-master")]:
        if os.path.isdir(path):
            try:
                df = load_stix_dataset(path)
                if not df.empty:
                    print(f"[VectorDB] MITRE ATT&CK          : {len(df):>7,} rows")
                    return df["question"].tolist(), df["answer"].tolist()
            except Exception as e:
                print(f"[VectorDB] STIX error: {e}")
    print("[VectorDB] MITRE ATT&CK          :       0 rows (folder not found)")
    return [], []

def _load_custom():
    if not os.path.exists(CUSTOM_CSV):
        return [], []
    try:
        df = pd.read_csv(CUSTOM_CSV, dtype=str)
        df["question"] = df["question"].fillna("").str.strip()
        df["answer"]   = df["answer"].fillna("").str.strip()
        df = df[(df["question"].str.len() >= 5) & (df["answer"].str.len() >= 10)]
        print(f"[VectorDB] Custom Q&A            : {len(df):>7,} rows")
        return df["question"].tolist(), df["answer"].tolist()
    except Exception as e:
        print(f"[VectorDB] custom_cyber.csv error: {e}"); return [], []

# ══════════════════════════════════════════════════════════════
# NEW LOADER 1 — vulnerability_definitions_Vulnerability_Definitions.csv
# Columns: Row_ID, Group_ID, Group_Name, Vulnerability, Definition
# ══════════════════════════════════════════════════════════════
def _load_vuln_definitions():
    if not os.path.exists(VULN_DEF_DEFINITIONS_CSV):
        print("[VectorDB] Vuln Definitions      : NOT FOUND")
        return [], []

    try:
        df = pd.read_csv(VULN_DEF_DEFINITIONS_CSV, dtype=str)

        df["Group_Name"]    = df["Group_Name"].fillna("").str.strip()
        df["Vulnerability"] = df["Vulnerability"].fillna("").str.strip()
        df["Definition"]    = df["Definition"].fillna("").str.strip()

        df = df[
            (df["Vulnerability"].str.len() >= 3)
            & (df["Definition"].str.len() >= 20)
        ]

        questions = (df["Group_Name"] + " - " + df["Vulnerability"]).tolist()
        answers   = df["Definition"].tolist()

        print(f"[VectorDB] Vuln Definitions      : {len(df):>7,} rows")
        return questions, answers

    except Exception as e:
        print(f"[VectorDB] Vuln Definitions error: {e}")
        return [], []

# ══════════════════════════════════════════════════════════════
# NEW LOADER 2 — vulnerabilities_chatbot_dataset_Chatbot_Dataset.csv
# Columns: Row_ID, Group_ID, Group_Name, Specific_Vulnerability,
#          Mitigation_Techniques, User_Query, Bot_Response
# ══════════════════════════════════════════════════════════════
def _load_vuln_chatbot():
    if not os.path.exists(VULN_CHAT_DATASET_CSV):
        print("[VectorDB] Vuln Chatbot Dataset  : NOT FOUND")
        return [], []

    try:
        df = pd.read_csv(VULN_CHAT_DATASET_CSV, dtype=str)

        df["User_Query"]   = df["User_Query"].fillna("").str.strip()
        df["Bot_Response"] = df["Bot_Response"].fillna("").str.strip()

        df = df[
            (df["User_Query"].str.len() >= 5)
            & (df["Bot_Response"].str.len() >= 20)
        ]

        print(f"[VectorDB] Vuln Chatbot Dataset  : {len(df):>7,} rows")
        return df["User_Query"].tolist(), df["Bot_Response"].tolist()

    except Exception as e:
        print(f"[VectorDB] Vuln Chatbot Dataset error: {e}")
        return [], []

def _load_vuln_definition_qa():
    if not os.path.exists(VULN_DEF_QA_CSV):
        return [], []

    try:
        df = pd.read_csv(VULN_DEF_QA_CSV, dtype=str)

        df["User_Query"]   = df["User_Query"].fillna("").str.strip()
        df["Bot_Response"] = df["Bot_Response"].fillna("").str.strip()

        df = df[
            (df["User_Query"].str.len() >= 5)
            & (df["Bot_Response"].str.len() >= 20)
        ]

        print(f"[VectorDB] Vuln Definition QA   : {len(df):>7,} rows")
        return df["User_Query"].tolist(), df["Bot_Response"].tolist()

    except Exception as e:
        print(f"[VectorDB] Vuln Definition QA error: {e}")
        return [], []

# ══════════════════════════════════════════════════════════════
# LOAD ALL  ← ✅ FIX 2: Deduplication now uses source-priority order
#              so vuln_def/vuln_chat entries are NOT overwritten by
#              earlier generic sources with the same question string.
# ══════════════════════════════════════════════════════════════

# Source priority: higher index = higher priority wins dedup
_SOURCE_PRIORITY = {
    "conv": 0, "ai": 1, "cyber": 2, "attck": 3,
    "cve": 4, "custom": 5, "vuln_def": 6, "vuln_chat": 7,
}

def _load_all():
    print("[VectorDB] ══════════════════════════════════════════")
    print("[VectorDB] Loading all datasets...")
    print("[VectorDB] ══════════════════════════════════════════")
    all_q, all_a, all_d, all_s = [], [], [], []

    def add(qs, as_, src, docs=None):
        all_q.extend(qs); all_a.extend(as_)
        all_d.extend(docs or [f"{q} {a[:400]}" for q, a in zip(qs, as_)])
        all_s.extend([src] * len(qs))

   
    
    q, a       = _load_cve();            add(q, a, "cve")
    q, a       = _load_attck();          add(q, a, "attck")
    q, a       = _load_vuln_definitions();   add(q, a, "vuln_def")
    q, a       = _load_vuln_definition_qa(); add(q, a, "vuln_def")
    q, a       = _load_vuln_chatbot();       add(q, a, "vuln_chat")
    q, a, docs = _load_cybersecurity();  add(q, a, "cyber", docs)
    q, a       = _load_custom();         add(q, a, "custom")
    q, a       = _load_conversation();   add(q, a, "conv")
    q, a       = _load_ai_qa();          add(q, a, "ai")

    # ✅ FIX 2: Priority-based deduplication — highest priority source wins
    # vuln_def (6) and vuln_chat (7) beat cve (4), cyber (2), etc.
    seen: dict[str, tuple] = {}
    for q, a, d, s in zip(all_q, all_a, all_d, all_s):
        key = q.strip().lower()
        if not key:
            continue
        if key not in seen:
            seen[key] = (q, a, d, s)
        else:
            existing_priority = _SOURCE_PRIORITY.get(seen[key][3], 0)
            new_priority      = _SOURCE_PRIORITY.get(s, 0)
            if new_priority > existing_priority:
                seen[key] = (q, a, d, s)  # higher-priority source wins

    if not seen:
        return [], [], [], []

    records = list(seen.values())
    q_out = [r[0] for r in records]
    a_out = [r[1] for r in records]
    d_out = [r[2] for r in records]
    s_out = [r[3] for r in records]

    # Count per source for logging
    src_counts: dict[str, int] = {}
    for s in s_out:
        src_counts[s] = src_counts.get(s, 0) + 1
    for src, cnt in sorted(src_counts.items()):
        print(f"[VectorDB]   {src:<12}: {cnt:>7,} (post-dedup)")

    print(f"[VectorDB] ─────────────────────────────────────────")
    print(f"[VectorDB] TOTAL (deduped)       : {len(q_out):>7,} rows")
    print(f"[VectorDB] ══════════════════════════════════════════")
    return q_out, a_out, d_out, s_out

# ══════════════════════════════════════════════════════════════
# LAYER 4 — BM25
# ══════════════════════════════════════════════════════════════
class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N     = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(self.N, 1)
        self.dfc   = Counter()
        for doc in corpus: self.dfc.update(set(doc))
        self.tf    = [Counter(d) for d in corpus]
        self.dl    = [len(d) for d in corpus]

    def scores(self, qtoks: list) -> np.ndarray:
        out = np.zeros(self.N, dtype=np.float32)
        for qt in set(qtoks):
            if qt not in self.dfc: continue
            idf = math.log(
                (self.N - self.dfc[qt] + 0.5) / (self.dfc[qt] + 0.5) + 1.0
            )
            for i in range(self.N):
                tf = self.tf[i].get(qt, 0)
                if not tf: continue
                denom = tf + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                out[i] += idf * tf * (self.k1 + 1) / denom
        return out

# ══════════════════════════════════════════════════════════════
# LAYER 5 — SENTENCE-TRANSFORMER MODEL (singleton)
# ══════════════════════════════════════════════════════════════
class _ModelLoader:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls._load()
        return cls._instance

    @staticmethod
    def _load():
        os.environ["TRANSFORMERS_OFFLINE"]       = "1"
        os.environ["HF_DATASETS_OFFLINE"]        = "1"
        os.environ["HF_HUB_OFFLINE"]             = "1"
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(BASE_DIR, "models")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("[ModelLoader] sentence-transformers not installed — FAISS disabled.")
            return None

        local_model = os.path.join(BASE_DIR, "models", "all-MiniLM-L6-v2")
        if os.path.isdir(local_model):
            print(f"[ModelLoader] Loading MiniLM from: {local_model}")
            model = SentenceTransformer(local_model)
            print("[ModelLoader] Model pre-warmed ✓")
            return model

        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        os.environ["HF_HUB_OFFLINE"]       = "0"
        print("[ModelLoader] Downloading all-MiniLM-L6-v2 from HuggingFace…")
        model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            cache_folder=os.path.join(BASE_DIR, "models"),
        )
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"]       = "1"
        print("[ModelLoader] Download complete. Offline mode re-enabled.")
        return model

# ══════════════════════════════════════════════════════════════
# LAYER 5 — PER-SOURCE FAISS INDEX
# ══════════════════════════════════════════════════════════════
class FAISSIndex:
    def __init__(self, group: str, max_docs: int):
        self.group    = group
        self.max_docs = max_docs
        self.index    = None
        self.doc_ids  = []
        self.ready    = False

    def build(self, docs: list, global_ids: list):
        model = _ModelLoader.get()
        if model is None:
            return
        try:
            import faiss
        except ImportError:
            print(f"[FAISS-{self.group}] faiss-cpu not installed — skipping.")
            return

        use_docs = docs[: self.max_docs]
        use_ids  = global_ids[: self.max_docs]
        print(f"[FAISS-{self.group}] Encoding {len(use_docs):,} docs…")
        t0 = time.time()
        embeddings = model.encode(
            use_docs, batch_size=FAISS_BATCH,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        print(f"[FAISS-{self.group}] Encoded in {time.time()-t0:.1f}s — {embeddings.shape}")
        dim        = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        self.doc_ids = use_ids
        self.ready   = True
        print(f"[FAISS-{self.group}] {self.index.ntotal:,} vectors, dim={dim}")

    def search(self, query: str, top_k: int = 10) -> list:
        if not self.ready:
            return []
        model = _ModelLoader.get()
        if model is None:
            return []
        try:
            qvec = model.encode(
                [query], normalize_embeddings=True, convert_to_numpy=True
            ).astype(np.float32)
            scores, positions = self.index.search(qvec, top_k)
            n = len(self.doc_ids)
            return [
                (self.doc_ids[pos], float(score))
                for pos, score in zip(positions[0], scores[0])
                if 0 <= pos < n and float(score) >= FAISS_SIM_FLOOR
            ]
        except Exception as e:
            print(f"[FAISS-{self.group}] Search error: {e}")
            return []

    def save(self, path: str):
        if not self.ready: return
        try:
            with open(path, "wb") as f:
                pickle.dump({"index": self.index, "doc_ids": self.doc_ids}, f, protocol=4)
            print(f"[FAISS-{self.group}] Saved → {os.path.basename(path)}")
        except Exception as e:
            print(f"[FAISS-{self.group}] Save failed: {e}")

    def load(self, path: str) -> bool:
        try:
            with open(path, "rb") as f:
                d = pickle.load(f)
            self.index   = d["index"]
            self.doc_ids = d["doc_ids"]
            self.ready   = True
            print(f"[FAISS-{self.group}] Loaded — {self.index.ntotal:,} vectors ✓")
            return True
        except Exception as e:
            print(f"[FAISS-{self.group}] Load failed ({e}), will rebuild.")
            return False

_FAISS_CACHE = {
    "cve":   FAISS_CVE_FILE,
    "attck": FAISS_ATTCK_FILE,
    "cyber": FAISS_CYBER_FILE,
    "other": FAISS_OTHER_FILE,
}

# ══════════════════════════════════════════════════════════════
# BUILD + CACHE
# ══════════════════════════════════════════════════════════════
def _hash(qs: list) -> str:
    return hashlib.md5("|".join(qs[:300]).encode()).hexdigest()

def _build_bm25_tfidf(docs: list):
    print("[VectorDB] Building BM25 + TF-IDF index…")
    t0     = time.time()
    corpus = [_tok(d) for d in docs]
    bm25   = BM25(corpus, BM25_K1, BM25_B)
    tfidf  = TfidfVectorizer(
        ngram_range=(1, 2), max_features=150_000,
        sublinear_tf=True, min_df=1,
    )
    mat = tfidf.fit_transform(docs)
    print(f"[VectorDB] Built in {time.time()-t0:.1f}s — {len(docs):,} docs, vocab {mat.shape[1]:,}")
    return bm25, tfidf, mat

# ── GLOBALS ──────────────────────────────────────────────────
questions: list = []
answers:   list = []
sources:   list = []
_bm25  = None
_tfidf = None
_mat   = None
_faiss: dict[str, FAISSIndex] = {
    "cve":   FAISSIndex("cve",   FAISS_MAX["cve"]),
    "attck": FAISSIndex("attck", FAISS_MAX["attck"]),
    "cyber": FAISSIndex("cyber", FAISS_MAX["cyber"]),
    "other": FAISSIndex("other", FAISS_MAX["other"]),
}

def _init():
    global questions, answers, sources, _bm25, _tfidf, _mat

    questions, answers, search_docs, sources = _load_all()
    h          = _hash(questions)
    cached_h   = open(HASH_FILE).read().strip() if os.path.exists(HASH_FILE) else ""
    hash_match = (cached_h == h)

    if os.path.exists(CACHE_FILE) and hash_match:
        print("[VectorDB] Loading cached BM25/TF-IDF…")
        with open(CACHE_FILE, "rb") as f:
            c = pickle.load(f)
        _bm25, _tfidf, _mat = c["bm25"], c["tfidf"], c["mat"]
        print(f"[VectorDB] Cache loaded ({_mat.shape[0]:,} docs).")
    else:
        _bm25, _tfidf, _mat = _build_bm25_tfidf(search_docs)
        with open(CACHE_FILE, "wb") as f:
            pickle.dump({"bm25": _bm25, "tfidf": _tfidf, "mat": _mat}, f, protocol=4)
        open(HASH_FILE, "w").write(h)
        print("[VectorDB] BM25/TF-IDF saved.")

    group_docs: dict[str, list] = {g: [] for g in _faiss}
    group_ids:  dict[str, list] = {g: [] for g in _faiss}
    for idx, (doc, src) in enumerate(zip(search_docs, sources)):
        grp = SOURCE_GROUP.get(src)
        if grp and grp in group_docs:
            group_docs[grp].append(doc)
            group_ids[grp].append(idx)

    for grp, idx_obj in _faiss.items():
        cache_path   = _FAISS_CACHE[grp]
        hash_f       = cache_path + ".hash"
        cached_grp_h = open(hash_f).read().strip() if os.path.exists(hash_f) else ""
        if os.path.exists(cache_path) and cached_grp_h == h:
            idx_obj.load(cache_path)
        else:
            idx_obj.build(group_docs[grp], group_ids[grp])
            idx_obj.save(cache_path)
            if idx_obj.ready:
                open(hash_f, "w").write(h)

    _ModelLoader.get()

_init()
print("[VectorDB] ✓ Ready!")

# ══════════════════════════════════════════════════════════════
# LAYER 4 — KEYWORD SEARCH  (BM25 + TF-IDF)
# ══════════════════════════════════════════════════════════════
def _kw_search(query: str, top_k: int = 15) -> list[dict]:
    toks   = _tok(query)
    bm     = _bm25.scores(toks)
    bm_n   = bm / bm.max() if bm.max() > 0 else bm
    qv     = _tfidf.transform([query])
    tf     = cosine_similarity(qv, _mat)[0]
    scores = BM25_WEIGHT * bm_n + TFIDF_WEIGHT * tf
    idx    = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "global_idx": int(i),
            "kw_score":   float(scores[i]),
            "sem_score":  0.0,
            "question":   questions[i],
            "answer":     answers[i],
            "source":     sources[i],
        }
        for i in idx
    ]

# ══════════════════════════════════════════════════════════════
# LAYER 5 — SEMANTIC SEARCH  (all FAISS groups)
# ══════════════════════════════════════════════════════════════
def _sem_search(query: str, top_k: int = 15) -> list[dict]:
    merged: list[dict] = []
    per_k = max(top_k, 8)
    n = len(questions)
    for grp, idx_obj in _faiss.items():
        if not idx_obj.ready:
            continue
        for gidx, score in idx_obj.search(query, top_k=per_k):
            if int(gidx) >= n:
                continue
            merged.append({
                "global_idx": gidx,
                "kw_score":   0.0,
                "sem_score":  score,
                "question":   questions[gidx],
                "answer":     answers[gidx],
                "source":     sources[gidx],
            })
    merged.sort(key=lambda x: x["sem_score"], reverse=True)
    return merged[:top_k]

# ══════════════════════════════════════════════════════════════
# QUERY INTENT DETECTION
# ══════════════════════════════════════════════════════════════
_INTENT_DEFINITION = re.compile(
    r"^(what (is|are|does|do|are)|how (does|do|is|are|can|would|"
    r"cves?|vulnerabilit)|explain|define|describe|"
    r"tell me about|what.s|give me an? (overview|explanation|definition)|"
    r"can you explain|i want to (know|understand)|"
    r"why (is|are|do|does)|when (is|are|do|does)|"
    r"what (makes|causes|happens|role))\b",
    re.I,
)

# ✅ FIX 6: New pattern to detect mitigation/prevention/remediation queries
_INTENT_MITIGATION = re.compile(
    r"\b(mitigat|prevent|remediat|defend|protect|secur|fix|patch|"
    r"harden|countermeasur|best practice|how to (stop|avoid|block|"
    r"defend|protect|prevent|mitigat|remediat|fix|secure))\b",
    re.I,
)

_GREETING_PAT = re.compile(
    r"^(hi+|hey+|hello+|howdy|sup|what'?s up|good (morning|evening|afternoon)|"
    r"how are you|how r u|hru|yo+|greetings|namaste|hola|salut|ciao|"
    r"nothing|nm|nvm|lol|ok|okay|sure|thanks|thank you|bye|goodbye|"
    r"who are you|what are you|what can you do|help me|help)[\?!\.\s]*$",
    re.I,
)

def _is_greeting(query: str) -> bool:
    return bool(_GREETING_PAT.match(query.strip()))

def _is_definitional(query: str) -> bool:
    q = query.strip()
    if _INTENT_DEFINITION.match(q):
        return True
    _HOW_CONCEPT = re.compile(
        r"^how\b.{3,}\b(used?|work|help|relate|affect|prevent|detect|"
        r"exploit|cause|function|operate|differ|compare|protect)\b",
        re.I,
    )
    if _HOW_CONCEPT.search(q):
        return True
    word_count = len(q.split())
    if word_count <= 4:
        if not _CVE_ID_PAT.search(q) and not _TECHNIQUE_ID_PAT.search(q):
            return True
    return False

def _is_mitigation_query(query: str) -> bool:
    """✅ FIX 6: Returns True when user is asking how to fix/prevent/mitigate."""
    return bool(_INTENT_MITIGATION.search(query))

# ══════════════════════════════════════════════════════════════
# LAYER 6 — SCORE FUSION
# ══════════════════════════════════════════════════════════════
def _fuse(kw_hits: list[dict], sem_hits: list[dict], query: str = "") -> list[dict]:
    RRF_K      = 60
    RRF_WEIGHT = 0.20

    definitional  = _is_definitional(query)  if query else False
    # ✅ FIX 5 & 6: Detect mitigation intent and apply aggressive CVE penalty
    mitigation    = _is_mitigation_query(query) if query else False

    if mitigation:
        print(f"  [Fuse/Intent] mitigation query — vuln_chat/vuln_def boosted, CVE penalised")
    elif definitional:
        word_count = len(query.strip().split())
        print(f"  [Fuse/Intent] definitional query (words={word_count})")

    pool: dict[int, dict] = {}

    def _add(hit: dict, rank: int, is_sem: bool):
        gidx = hit["global_idx"]
        if gidx not in pool:
            pool[gidx] = {
                "global_idx":  gidx,
                "question":    hit["question"],
                "answer":      hit["answer"],
                "source":      hit["source"],
                "kw_score":    0.0,
                "tfidf_score": 0.0,
                "sem_score":   0.0,
                "rrf":         0.0,
            }
        pool[gidx]["rrf"] += 1.0 / (RRF_K + rank + 1)
        if is_sem:
            pool[gidx]["sem_score"] = max(pool[gidx]["sem_score"], hit["sem_score"])
        else:
            total_kw_w = BM25_WEIGHT + TFIDF_WEIGHT
            bm_share   = BM25_WEIGHT  / total_kw_w
            tf_share   = TFIDF_WEIGHT / total_kw_w
            pool[gidx]["kw_score"]    = max(pool[gidx]["kw_score"],
                                            hit["kw_score"] * bm_share)
            pool[gidx]["tfidf_score"] = max(pool[gidx]["tfidf_score"],
                                            hit["kw_score"] * tf_share)

    for rank, h in enumerate(kw_hits):  _add(h, rank, is_sem=False)
    for rank, h in enumerate(sem_hits): _add(h, rank, is_sem=True)

    kw_max  = max((v["kw_score"]    for v in pool.values()), default=1) or 1
    tf_max  = max((v["tfidf_score"] for v in pool.values()), default=1) or 1
    sem_max = max((v["sem_score"]   for v in pool.values()), default=1) or 1
    rrf_max = max((v["rrf"]         for v in pool.values()), default=1) or 1

    results = []
    for doc in pool.values():
        src = doc["source"]

        # ✅ FIX 5: Compute boost based on both mitigation and definitional intent
        if mitigation:
            if src in ("vuln_chat", "vuln_def"):
                boost = 1.80    # Strong boost — these datasets ARE mitigation Q&A
            elif src == "custom":
                boost = 1.50
            elif src in ("cyber", "attck"):
                boost = 1.20
            elif src == "cve":
                boost = 0.40    # Strong penalty — CVE records are not mitigation guides
            else:
                boost = SOURCE_BOOST.get(src, 1.0)
        elif definitional:
            word_count = len(query.strip().split())
            if word_count <= 2:
                _cve_boost_override     = 0.10
                _concept_boost_override = 1.50
            else:
                _cve_boost_override     = 0.18
                _concept_boost_override = 1.35
            if src == "cve":
                boost = _cve_boost_override
            elif src in ("cyber", "attck", "custom", "vuln_def", "vuln_chat"):
                boost = _concept_boost_override
            else:
                boost = SOURCE_BOOST.get(src, 1.0)
        else:
            boost = SOURCE_BOOST.get(src, 1.0)

        raw = (
            BM25_WEIGHT   * (doc["kw_score"]    / kw_max)  +
            TFIDF_WEIGHT  * (doc["tfidf_score"]  / tf_max)  +
            FAISS_WEIGHT  * (doc["sem_score"]    / sem_max) +
            RRF_WEIGHT    * (doc["rrf"]          / rrf_max)
        )
        results.append({**doc, "score": round(raw * boost, 4)})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results

# ══════════════════════════════════════════════════════════════
# ANSWER CLEANING
# ══════════════════════════════════════════════════════════════
_MD_LINK     = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_CITATION    = re.compile(r"\(Citation:[^)]+\)")
_MULTI_NL    = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+$", re.M)

def _clean_answer(answer: str, source: str) -> str:
    if not answer:
        return answer
    text = _MD_LINK.sub(r"\1", answer)
    text = _CITATION.sub("", text)
    text = _TRAILING_WS.sub("", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()

# ══════════════════════════════════════════════════════════════
# LAYER 7 — ANSWER VALIDATION
# ══════════════════════════════════════════════════════════════
def _validate(hit: dict, query: str) -> tuple[bool, str]:
    answer = (hit.get("answer") or "").strip()
    src    = hit.get("source", "")

    if len(answer) < 20:
        return False, "answer too short"

    pat = _ANSWER_SIGNALS.get(src)
    if pat and not pat.search(answer):
        return False, f"no domain signal for source='{src}'"

    _BOILERPLATE = re.compile(
        r"^(i (don'?t|do not) know|not sure|i'm not sure|"
        r"i cannot|i can'?t|no information|unavailable|"
        r"please (consult|contact|refer)|see (the )?documentation)[\.,]?$",
        re.I,
    )
    if _BOILERPLATE.match(answer[:120]):
        return False, "boilerplate answer"

    if src != "conv":
        _STOPWORDS = {"what", "is", "a", "an", "the", "how", "does",
                      "do", "i", "to", "of", "in", "and", "or", "for",
                      "me", "my", "can", "you", "please", "tell", "give",
                      "with", "about", "explain", "define", "describe",
                      # ✅ Added mitigation stopwords so "mitigate" alone
                      #    doesn't count as a meaningful query token
                      "mitigate", "prevent", "fix", "stop", "avoid"}
        q_tokens = set(_tok(query)) - _STOPWORDS
        if len(q_tokens) >= 2:
            combined = (hit.get("question", "") + " " + answer).lower()
            if not any(t in combined for t in q_tokens):
                return False, "no query token overlap in answer"

    return True, ""

# ══════════════════════════════════════════════════════════════
# LAYER 8 — GRACEFUL DEGRADATION
# ══════════════════════════════════════════════════════════════
def _fallback(query: str) -> str:
    q = query.lower()
    if re.search(r"cve-\d{4}-\d+", q):
        return (
            "That specific CVE ID is not in my local database. "
            "Try the NVD at https://nvd.nist.gov or search by "
            "affected product/vendor name instead."
        )
    if re.search(r"\bt\d{4}(\.\d{3})?\b", q):
        return (
            "That MITRE ATT&CK technique ID wasn't found. "
            "Try the technique name — e.g. 'spearphishing', "
            "'credential dumping', 'lateral movement'."
        )
    _HINTS = [
        (r"\b(cyber|infosec|security)\b",       "cybersecurity",               "What is cybersecurity?"),
        (r"\b(ransomware|ransom)\b",             "ransomware",                  "How does ransomware work?"),
        (r"\b(phish|phising|smish|vish)\b",      "phishing",                    "What is phishing?"),
        (r"\b(sqli|sql.?inject)\b",              "SQL injection",               "How to mitigate SQL injection?"),
        (r"\b(xss|cross.?site.?script)\b",       "cross-site scripting (XSS)",  "How to prevent XSS?"),
        (r"\b(bof|buffer.?over)\b",              "buffer overflow",             "What is a buffer overflow?"),
        (r"\b(ddos|dos attack)\b",               "DDoS attack",                 "What is a DDoS attack?"),
        (r"\b(privesc|privilege.?escal)\b",      "privilege escalation",        "What is privilege escalation?"),
        (r"\b(apt|advanced.?persist)\b",         "Advanced Persistent Threats", "What is an APT?"),
        (r"\b(mitre|attck|ttp)\b",               "MITRE ATT&CK",                "What is the MITRE ATT&CK framework?"),
        (r"\b(ml|machine.?learn)\b",             "machine learning",            "What is machine learning?"),
        (r"\b(llm|large.?language)\b",           "large language models",       "What is a large language model?"),
    ]
    for pattern, label, example in _HINTS:
        if re.search(pattern, q, re.I):
            return (
                f"I couldn't find a precise match for your {label} query. "
                f"Try a more specific question like: \"{example}\""
            )
    return (
        "No confident match found for your query. "
        "I can answer questions about:\n"
        "  • CVE vulnerabilities  (e.g. 'What is CVE-2021-44228?')\n"
        "  • MITRE ATT&CK         (e.g. 'Explain spearphishing')\n"
        "  • Cybersecurity topics (e.g. 'How does ransomware work?')\n"
        "  • AI/ML concepts       (e.g. 'What is a transformer model?')\n"
        "Try rephrasing with more specific terms."
    )

# ══════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════
def search_vector_db(
    query: str,
    db_type: str = "hybrid",
    context: list = [],
) -> dict:
    enriched, sem_query = _preprocess(query, context)
    print(f"\n[Search] raw='{query[:60]}'")
    print(f"[Search] enriched='{enriched[:120]}'")
    print(f"[Search] sem_query='{sem_query[:80]}'")

    # ── Greeting shortcut ─────────────────────────────────────
    if _is_greeting(query):
        kw_hits   = _kw_search(query, top_k=5)
        sem_hits  = _sem_search(query, top_k=5)
        fused     = _fuse(kw_hits, sem_hits, query=query)
        conv_hits = [h for h in fused if h["source"] == "conv"]
        if conv_hits:
            best = conv_hits[0]
            return {"answer": best["answer"], "score": best["score"], "source": "conv"}

    # ── Layers 4+5: keyword + semantic ────────────────────────
    kw_hits  = _kw_search(enriched,   top_k=15)
    sem_hits = _sem_search(sem_query, top_k=25)

    if enriched != query.lower().strip():
        raw_kw  = _kw_search(query, top_k=8)
        raw_sem = _sem_search(query, top_k=8)
        raw_top_kw  = raw_kw[0]["kw_score"]   if raw_kw  else 0
        raw_top_sem = raw_sem[0]["sem_score"]  if raw_sem else 0
        enr_top_kw  = kw_hits[0]["kw_score"]  if kw_hits else 0
        enr_top_sem = sem_hits[0]["sem_score"] if sem_hits else 0
        if raw_top_kw  > enr_top_kw:  kw_hits  = raw_kw  + kw_hits
        if raw_top_sem > enr_top_sem: sem_hits  = raw_sem + sem_hits

    # ── Layer 6: score fusion ──────────────────────────────────
    fused = _fuse(kw_hits, sem_hits, query=query)

    print(f"[Search] top-3 fused results (intent-adjusted):")
    for h in fused[:3]:
        print(f"  score={h['score']:.4f} [{h['source']}] '{h['question'][:65]}'")

    if not fused:
        return {"answer": _fallback(query), "score": 0.0, "source": "fallback"}

    # ── Prefer cyber-domain over weak conv/ai hit ─────────────
    best = fused[0]
    if best["source"] in ("conv", "ai"):
        for h in fused[1:6]:
            if h["source"] in ("custom", "cve", "attck", "cyber", "vuln_def", "vuln_chat"):
                if best["score"] - h["score"] <= CYBER_MARGIN:
                    best = h
                    print(f"  [Promote] [{h['source']}] '{h['question'][:55]}'")
                    break

    # ── Layer 7: answer validation ────────────────────────────
    valid, reason = _validate(best, query)
    if not valid:
        print(f"  [Validate] FAIL — {reason}. Trying next candidates…")
        for candidate in fused[1:8]:
            v, r = _validate(candidate, query)
            if v:
                best  = candidate
                valid = True
                print(f"  [Validate] PASS (fallback candidate) [{candidate['source']}]")
                break
        if not valid:
            print(f"  [Validate] All candidates failed — triggering fallback")
            return {"answer": _fallback(query), "score": 0.0, "source": "fallback"}

    # ── Layer 8: min-score gate ───────────────────────────────
    threshold = MIN_SCORE.get(best["source"], 0.12)
    if best["score"] < threshold:
        print(f"  [Gate] score={best['score']:.4f} < threshold={threshold:.4f} — fallback")
        return {"answer": _fallback(query), "score": 0.0, "source": "fallback"}

    print(f"  [Return] score={best['score']:.4f} [{best['source']}] ✓")

    # ── Layer 9: Phi-3 answer generation ──────────────────────
    cleaned_context = _clean_answer(best["answer"], best["source"])

    phi3 = _get_phi3()
    if phi3 and phi3.ready:
        print(f"  [Phi3] Generating response for [{best['source']}] answer...")
        final_answer = phi3.generate(
            query=query,
            context=cleaned_context,
            source=best["source"],
        )
    else:
        final_answer = cleaned_context

    return {
        "answer": final_answer,
        "score":  best["score"],
        "source": best["source"],
    }