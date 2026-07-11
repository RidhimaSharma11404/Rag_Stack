"""
llm/rag_generator.py
====================
Calls llama-server (Phi-3) via HTTP.
START SERVER:
    llm_cpp\llama-server.exe -m Phi-3-mini-4k-instruct-q4.gguf --port 11434
"""
import requests
from llm.prompt_builder import build_prompt

LLAMA_URL = "http://127.0.0.1:11434"
TIMEOUT   = 180

def generate_answer(question: str, contexts: list, source: str = "default") -> str:
    """
    Args:
        question : user's question
        contexts : list of retrieved answer strings (up to 3 used)
        source   : source label — 'cve','attck','cyber','vuln_def','vuln_chat','custom','conv'
    Returns:
        Generated answer string, or 'LLM Error: …' on failure.
    """
    print("\n========== SOURCE ==========")
    print(source)

    print("\n========== CONTEXTS ==========")
    for i, c in enumerate(contexts):
        print(f"\n--- Context {i+1} ---")
        print(c)
    print("=============================\n")

    prompt = build_prompt(question, contexts, source=source)
    print("\n" + "="*80)
    print(prompt)
    print("="*80 +"\n")
    print("Prompt length:", len(prompt))
    print(prompt[:500])
    try:
        response = requests.post(
            f"{LLAMA_URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 8000,
                "stop": ["<|end|>", "<|user|>", "\n\n\n"],
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        print("========== LLM RESPONSE ==========")
        print(data)
        print("==================================")
        if "choices" not in data:
            return f"LLM Error: unexpected response: {data}"
        content = data["choices"][0]["message"]["content"].strip()
        if len(content) < 10 or content.startswith("<|"):
            return "LLM Error: model returned empty/malformed output."
        return content
    except requests.exceptions.Timeout:
        return "LLM Error: Timed out. Is llama-server running on port 11434?"
    except requests.exceptions.ConnectionError:
        return ("LLM Error: Cannot connect. "
                "Run: llm_cpp\\llama-server.exe -m Phi-3-mini-4k-instruct-q4.gguf --port 11434")
    except Exception as e:
        return f"LLM Error: {e}"