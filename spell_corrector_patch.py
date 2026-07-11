# ══════════════════════════════════════════════════════════════
# LAYER 2 — FUZZY SPELL CORRECTION  (FIXED)
# ══════════════════════════════════════════════════════════════
#
#   FIX 1: Filter _ALIAS keys shorter than 6 chars out of the vocab.
#   FIX 2: Raise cutoff 0.82 → 0.88 for extra safety.
#   FIX 3: Only correct tokens ≥ 6 chars (was 5) to skip short words.
#

_SPELL_VOCAB: list[str] = sorted(set(
    # Only include alias keys that look like real words (len >= 6)
    # Short abbreviations like "attck", "mitm", "dos" must NOT be in vocab
    [k for k in _ALIAS.keys() if len(k) >= 6] + [
        "cybersecurity", "vulnerability", "authentication", "authorization",
        "encryption", "firewall", "malware", "ransomware", "phishing", "exploit",
        "intrusion", "penetration", "reconnaissance", "lateral", "escalation",
        "exfiltration", "persistence", "detection", "mitigation", "remediation",
        "incident", "forensics", "threat", "adversary", "payload", "backdoor",
        "trojan", "rootkit", "botnet", "spyware", "adware", "worm", "virus",
        "credential", "certificate", "signature", "anomaly", "baseline", "audit",
        "compliance", "regulation", "policy", "governance", "risk", "assessment",
        "injection", "overflow", "traversal", "spoofing", "hijacking", "forgery",
        "privilege", "execution", "persistence", "exfiltration", "discovery",
        "collection", "command", "control", "impact", "defense", "evasion",
    ]
))


def _spell_correct(text: str) -> str:
    tokens    = text.split()
    corrected = []
    for tok in tokens:
        # FIX: raised min length from 5 → 6 to protect short real words
        if len(tok) < 6 or tok in _SPELL_VOCAB:
            corrected.append(tok)
            continue
        # FIX: raised cutoff from 0.82 → 0.88 to prevent false corrections
        matches = get_close_matches(tok, _SPELL_VOCAB, n=1, cutoff=0.88)
        if matches and matches[0] != tok:
            corrected.append(matches[0])
            print(f"  [SpellCorrect] '{tok}' → '{matches[0]}'")
        else:
            corrected.append(tok)
    return " ".join(corrected)