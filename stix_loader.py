"""
stix_loader.py  (fixed - excludes beta/sample/versioned files)
"""

import os
import json
import glob
import pandas as pd
from typing import Optional


def _clean(text: str) -> str:
    return text.strip()[:1500] if text else ""


def _get_external_id(obj: dict) -> str:
    for ref in obj.get("external_references", []):
        if ref.get("source_name", "") in ("mitre-attack", "mitre-mobile-attack", "mitre-ics-attack"):
            return ref.get("external_id", "")
    return ""


def _parse_stix_object(obj: dict) -> Optional[dict]:
    if not isinstance(obj, dict):
        return None
    if obj.get("revoked") or obj.get("x_mitre_deprecated"):
        return None

    obj_type = obj.get("type", "")
    name     = obj.get("name", "").strip()
    desc     = _clean(obj.get("description", ""))
    att_id   = _get_external_id(obj)

    if not name or not desc:
        return None

    if obj_type == "attack-pattern":
        tactic_refs = [p.get("phase_name", "").replace("-", " ").title()
                       for p in obj.get("kill_chain_phases", [])]
        tactics   = ", ".join(tactic_refs) or "Unknown"
        platforms = ", ".join(obj.get("x_mitre_platforms", []))
        detection = _clean(obj.get("x_mitre_detection", ""))

        # ✅ Build answer only with fields that actually have data
        answer = desc
        answer += f"\n\nTactics: {tactics}"
        if platforms:                        # only add if not empty
            answer += f"\nPlatforms: {platforms}"
        if detection:                        # only add if not empty
            answer += f"\n\nDetection: {detection[:400]}"

        id_part = f" ({att_id})" if att_id else ""
        return {"question": f"What is {name}{id_part}?", "answer": answer[:1500]}

    elif obj_type == "course-of-action":
        id_part = f" ({att_id})" if att_id else ""
        return {"question": f"How to mitigate {name}{id_part}?", "answer": desc}

    elif obj_type == "intrusion-set":
        aliases   = obj.get("aliases", [])
        alias_str = ", ".join(a for a in aliases if a != name)
        answer    = f"Also known as: {alias_str}\n\n{desc}" if alias_str else desc
        id_part   = f" ({att_id})" if att_id else ""
        return {"question": f"What is {name}{id_part}?", "answer": answer[:1500]}

    elif obj_type == "malware":
        platforms = ", ".join(obj.get("x_mitre_platforms", []))
        id_part   = f" ({att_id})" if att_id else ""
        # ✅ Only add Platforms line if it has data
        answer = desc
        if platforms:
            answer += f"\n\nPlatforms: {platforms}"
        return {"question": f"What is {name} malware{id_part}?", "answer": answer[:1500]}

    elif obj_type == "tool":
        platforms = ", ".join(obj.get("x_mitre_platforms", []))
        id_part   = f" ({att_id})" if att_id else ""
        # ✅ Only add Platforms line if it has data
        answer = desc
        if platforms:
            answer += f"\n\nPlatforms: {platforms}"
        return {"question": f"What is {name} tool{id_part}?", "answer": answer[:1500]}

    elif obj_type == "x-mitre-tactic":
        # ✅ Tactics have no platforms or detection — just return description only
        #    This prevents Phi-3 from ever seeing empty fields to hallucinate about
        return {"question": f"What is the {name} tactic in MITRE ATT&CK?", "answer": desc}

    return None


def _is_valid_stix_file(filename: str) -> bool:
    """
    Only accept the canonical latest files:
      enterprise-attack.json
      mobile-attack.json
      ics-attack.json
    Reject everything else: versioned (v10.1), beta, sample, index files.
    """
    valid_names = {
        "enterprise-attack.json",
        "mobile-attack.json",
        "ics-attack.json",
    }
    return os.path.basename(filename) in valid_names


def load_stix_dataset(stix_repo_path: str) -> pd.DataFrame:
    if not os.path.isdir(stix_repo_path):
        print(f"[STIXLoader] ERROR: Folder not found: {stix_repo_path}")
        return pd.DataFrame(columns=["question", "answer"])

    # Find only the three canonical latest files
    target_files = []
    for domain in ["enterprise-attack", "mobile-attack", "ics-attack"]:
        latest = os.path.join(stix_repo_path, domain, f"{domain}.json")
        if os.path.isfile(latest):
            target_files.append(latest)

    if not target_files:
        # Fallback: walk but strictly filter
        all_files = glob.glob(os.path.join(stix_repo_path, "**", "*.json"), recursive=True)
        target_files = [f for f in all_files if _is_valid_stix_file(f)]

    print(f"[STIXLoader] Parsing {len(target_files)} canonical STIX file(s): "
          f"{[os.path.basename(f) for f in target_files]}")

    if not target_files:
        print("[STIXLoader] No valid STIX files found.")
        return pd.DataFrame(columns=["question", "answer"])

    seen_ids = set()
    records  = []

    for filepath in target_files:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[STIXLoader] Could not read {filepath}: {e}")
            continue

        if isinstance(data, list):
            objects = data
        elif isinstance(data, dict):
            objects = data.get("objects", [])
        else:
            continue

        print(f"[STIXLoader] {os.path.basename(filepath)}: {len(objects)} objects")

        for obj in objects:
            if not isinstance(obj, dict):
                continue
            obj_id = obj.get("id", "")
            if obj_id in seen_ids:
                continue
            seen_ids.add(obj_id)

            result = _parse_stix_object(obj)
            if result:
                records.append(result)

    if not records:
        print("[STIXLoader] WARNING: No usable STIX objects found.")
        return pd.DataFrame(columns=["question", "answer"])

    df = pd.DataFrame(records, columns=["question", "answer"])
    df = df.drop_duplicates(subset=["question"])
    df = df[df["question"].str.strip() != ""]
    df = df[df["answer"].str.strip()   != ""]
    df = df.reset_index(drop=True)
    print(f"[STIXLoader] Loaded {len(df)} ATT&CK records "
          f"(techniques, groups, malware, tools, mitigations, tactics)")
    return df