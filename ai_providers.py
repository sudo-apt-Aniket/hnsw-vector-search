"""
ai_providers.py — embedding + generation backends for the RAG endpoints.

Design: try Ollama first (best privacy, zero cost, what the original
project used). If Ollama isn't reachable — which is the normal case on a
hosted deployment — fall back to:
  * embeddings  -> sentence-transformers, running locally in-process
                   (no external API, no key needed)
  * generation  -> Groq's hosted API (OpenAI-compatible, free tier,
                   fast enough for a live demo)

Once a backend produces the first successful embedding, it is *pinned*
for the rest of the process lifetime. Embeddings from different models
have different dimensionality and are not comparable, so silently
switching backends mid-session would corrupt the vector index.
"""

import os
import requests
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------
#  Ollama (local, optional)
# ---------------------------------------------------------------------

class OllamaClient:
    def __init__(self, host: str = None, port: int = 11434):
        host = host or os.environ.get("OLLAMA_HOST", "127.0.0.1")
        self.base_url = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model = "llama3.2"

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def embed(self, text: str) -> List[float]:
        try:
            r = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=30
            )
            if r.status_code != 200:
                return []
            return r.json().get("embedding", [])
        except Exception:
            return []

    def generate(self, prompt: str) -> str:
        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.gen_model, "prompt": prompt, "stream": False},
                timeout=180
            )
            if r.status_code != 200:
                return ""
            return r.json().get("response", "")
        except Exception:
            return ""

# ---------------------------------------------------------------------
#  sentence-transformers (local, no network needed after first download)
# ---------------------------------------------------------------------

class SentenceTransformerEmbedder:
    """Lazy-loaded so importing this module doesn't pull in torch unless
    it's actually needed (Ollama is still preferred when available)."""

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims, ~80MB

    def __init__(self):
        self._model = None
        self._load_failed = False

    def _load(self):
        if self._model is not None or self._load_failed:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL_NAME)
        except Exception:
            self._load_failed = True

    def is_available(self) -> bool:
        self._load()
        return self._model is not None

    def embed(self, text: str) -> List[float]:
        self._load()
        if self._model is None:
            return []
        try:
            return self._model.encode(text).tolist()
        except Exception:
            return []

# ---------------------------------------------------------------------
#  Groq (hosted, OpenAI-compatible chat completions API)
# ---------------------------------------------------------------------

class GroqClient:
    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY", "")
        self.model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            return ""
        try:
            r = requests.post(
                self.API_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                },
                timeout=30
            )
            if r.status_code != 200:
                return ""
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception:
            return ""

# ---------------------------------------------------------------------
#  AIProvider — the cascade the rest of the app talks to
# ---------------------------------------------------------------------

class AIProvider:
    def __init__(self):
        self.ollama = OllamaClient()
        self.st = SentenceTransformerEmbedder()
        self.groq = GroqClient()
        self._embed_backend: Optional[str] = None  # pinned after first success
        self._gen_backend: Optional[str] = None

    # -- embeddings ----------------------------------------------------

    def embed(self, text: str) -> Tuple[List[float], str]:
        """Returns (vector, backend_name). vector is [] if nothing worked."""
        # Once pinned, stick to the same backend so dimensions stay consistent.
        if self._embed_backend == "ollama":
            v = self.ollama.embed(text)
            return (v, "ollama") if v else ([], "ollama")
        if self._embed_backend == "sentence-transformers":
            v = self.st.embed(text)
            return (v, "sentence-transformers") if v else ([], "sentence-transformers")

        # Not pinned yet: try Ollama, then sentence-transformers.
        if self.ollama.is_available():
            v = self.ollama.embed(text)
            if v:
                self._embed_backend = "ollama"
                return v, "ollama"
        v = self.st.embed(text)
        if v:
            self._embed_backend = "sentence-transformers"
            return v, "sentence-transformers"
        return [], "none"

    # -- generation ------------------------------------------------------

    def generate(self, prompt: str) -> Tuple[str, str]:
        """Returns (answer, backend_name)."""
        if self.ollama.is_available():
            text = self.ollama.generate(prompt)
            if text:
                return text, "ollama"
        if self.groq.is_available():
            text = self.groq.generate(prompt)
            if text:
                return text, "groq"
        return "", "none"

    # -- status ----------------------------------------------------------

    def status(self, probe_sentence_transformers: bool = False) -> dict:
        """By default this does NOT trigger a sentence-transformers/torch
        import (that's slow on cold start) — it only reports what's already
        been probed by a real embed() call. Pass probe_sentence_transformers=True
        if you want to force the check (e.g. from a diagnostics call)."""
        ollama_up = self.ollama.is_available()
        if probe_sentence_transformers:
            st_ready = self.st.is_available()
        else:
            st_ready = self.st._model is not None
        return {
            "ollamaAvailable": ollama_up,
            "sentenceTransformersAvailable": st_ready,
            "groqConfigured": self.groq.is_available(),
            "embedBackend": self._embed_backend or "not yet used",
            "genModel": (
                self.ollama.gen_model if ollama_up
                else (self.groq.model if self.groq.is_available() else "none")
            ),
        }
