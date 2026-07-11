"""
llm/phi3_generator.py
=====================
FIX: This file was MISSING — caused:
    [VectorDB] WARNING: Phi-3 failed to load (No module named 'phi3_generator')

vector_db.py calls: from phi3_generator import get_phi3
This bridges it to the actual llama-server via rag_generator.
"""

import requests
from llm.prompt_builder import build_prompt
from llm.rag_generator  import generate_answer

LLAMA_HEALTH = "http://127.0.0.1:8080/health"


def _server_ok() -> bool:
    try:
        r = requests.get(LLAMA_HEALTH, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


class Phi3Wrapper:
    def __init__(self):
        self.ready = _server_ok()
        status = "reachable ✓" if self.ready else (
            "NOT reachable — start with:\n"
            "  llama-server -m Phi-3-mini-4k-instruct-q4.gguf --port 8080"
        )
        print(f"[Phi3Wrapper] llama-server {status}")

    def generate(self, query: str, context: str, source: str = "default") -> str:
        """
        Generate answer via Phi-3.
        Uses source-specific system prompt from prompt_builder.
        Falls back to raw context on any error.
        """
        if not self.ready:
            return context

        # Build source-aware prompt
        contexts = [{"answer": context, "score": 1.0}]
        prompt   = build_prompt(query, contexts, source=source)

        # Call llama-server directly with the built prompt
        try:
            import requests as _req
            response = _req.post(
                "http://127.0.0.1:8080/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 200,
                },
                timeout=180,
            )
            response.raise_for_status()
            data    = response.json()
            content = data["choices"][0]["message"]["content"].strip()

            if len(content) < 10 or content.startswith("<|"):
                return context

            return content

        except Exception as e:
            print(f"[Phi3Wrapper] Error: {e}")
            return context


_instance: Phi3Wrapper | None = None


def get_phi3() -> Phi3Wrapper:
    """Singleton — called by vector_db.py Layer 9."""
    global _instance
    if _instance is None:
        _instance = Phi3Wrapper()
    return _instance