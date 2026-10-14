"""Local LLM clients. Both backends are free and run on the user's machine; no API key or account is needed.

* ``OllamaClient``      - talks to a locally running Ollama server (https://ollama.com) over its documented REST API.
* ``HuggingFaceClient`` - optional alternative using an open-weight model from the free Hugging Face Hub, run locally with
                          ``transformers``. The first use downloads the model once; afterwards it works offline.
The rest of the app only depends on the small ``LLMClient`` interface, so the dashboard works without either backend.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Protocol

import requests

import config
from src.utils.logging_config import get_logger

log = get_logger(__name__)


class LLMUnavailable(RuntimeError):
    """The language model cannot be used right now (not running, model missing, timeout...)."""


@dataclass
class LLMStatus:
    backend: str
    available: bool
    model: str
    message: str
    models: list[str] = field(default_factory=list)


class LLMClient(Protocol):
    name: str
    model: str

    def status(self) -> LLMStatus: ...

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.1, max_tokens: int = 700) -> str: ...


class OllamaClient:
    name = "ollama"

    def __init__(self, host: str | None = None, model: str | None = None, timeout: int | None = None) -> None:
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        self.model = model or config.OLLAMA_MODEL
        self.timeout = timeout or config.LLM_TIMEOUT_S

    def _installed(self) -> list[str]:
        r = requests.get(f"{self.host}/api/tags", timeout=3)
        r.raise_for_status()
        return [m.get("name", "") for m in r.json().get("models", [])]

    def _matches(self, installed: list[str]) -> bool:
        base = self.model.split(":")[0]
        return any(n == self.model or (":" not in self.model and n.split(":")[0] == base) for n in installed)

    def status(self) -> LLMStatus:
        try:
            installed = self._installed()
        except requests.RequestException:
            return LLMStatus(self.name, False, self.model, f"Ollama is not reachable at {self.host}. Start it with `ollama serve`.")
        except ValueError:
            return LLMStatus(self.name, False, self.model, "Ollama returned an unexpected response.")
        if not self._matches(installed):
            return LLMStatus(self.name, False, self.model, f"Ollama is running but model '{self.model}' is not installed. Run `ollama pull {self.model}`.", installed)
        return LLMStatus(self.name, True, self.model, "Ollama is running and the model is installed.", installed)

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.1, max_tokens: int = 700) -> str:
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "options": {"temperature": temperature, "num_predict": max_tokens}}
        try:
            r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        except requests.ConnectionError as e:
            raise LLMUnavailable(f"Ollama is not running at {self.host}.") from e
        except requests.Timeout as e:
            raise LLMUnavailable(f"The model did not answer within {self.timeout} seconds (try a smaller model).") from e
        except requests.RequestException as e:
            raise LLMUnavailable("The request to Ollama failed.") from e
        if r.status_code == 404:
            raise LLMUnavailable(f"Model '{self.model}' is not installed. Run `ollama pull {self.model}`.")
        if r.status_code != 200:
            raise LLMUnavailable(f"Ollama returned HTTP {r.status_code}.")
        try:
            text = r.json()["message"]["content"].strip()
        except (KeyError, ValueError) as e:
            raise LLMUnavailable("Ollama returned an unreadable response.") from e
        if not text:
            raise LLMUnavailable("The model returned an empty answer.")
        return text


_HF_PIPELINES: dict[str, object] = {}


class HuggingFaceClient:
    """Runs an open-weight instruction model from the Hugging Face Hub locally (CPU or GPU). Optional dependency."""

    name = "huggingface"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or config.HF_MODEL

    def status(self) -> LLMStatus:
        missing = [m for m in ("transformers", "torch") if importlib.util.find_spec(m) is None]
        if missing:
            return LLMStatus(self.name, False, self.model, f"Optional packages missing ({', '.join(missing)}). Install with `pip install transformers torch accelerate`.")
        return LLMStatus(self.name, True, self.model, "Hugging Face transformers is installed. The model downloads on first use (free, no token needed for public models).")

    def _pipe(self):
        if self.model not in _HF_PIPELINES:
            try:
                from transformers import pipeline  # lazy: heavy import
                _HF_PIPELINES[self.model] = pipeline("text-generation", model=self.model, device_map="auto")
            except Exception as e:
                raise LLMUnavailable(f"Could not load Hugging Face model '{self.model}': {type(e).__name__}") from e
        return _HF_PIPELINES[self.model]

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.1, max_tokens: int = 700) -> str:
        pipe = self._pipe()
        try:
            out = pipe(messages, max_new_tokens=max_tokens, do_sample=temperature > 0.2, temperature=max(temperature, 0.01), return_full_text=False)
            text = out[0]["generated_text"]
            if isinstance(text, list):
                text = text[-1]["content"]
            text = str(text).strip()
        except Exception as e:
            raise LLMUnavailable(f"Generation failed: {type(e).__name__}") from e
        if not text:
            raise LLMUnavailable("The model returned an empty answer.")
        return text


def get_llm_client(backend: str | None = None) -> LLMClient | None:
    """Pick a usable client. 'auto' tries Ollama first, then Hugging Face. Returns None for fallback-only mode."""
    backend = (backend or config.LLM_BACKEND).lower()
    if backend == "fallback":
        return None
    order: list[LLMClient] = []
    if backend in ("auto", "ollama"):
        order.append(OllamaClient())
    if backend in ("auto", "huggingface"):
        order.append(HuggingFaceClient())
    for client in order:
        if client.status().available:
            return client
    return None


def all_statuses() -> list[LLMStatus]:
    return [OllamaClient().status(), HuggingFaceClient().status()]
