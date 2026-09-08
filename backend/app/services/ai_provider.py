from __future__ import annotations

import base64
import mimetypes
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from ..config import get_settings
from ..logging_setup import logger

settings = get_settings()


class AIProviderError(Exception):
    """Raised when an AI provider fails during inference."""
    pass


class BaseAIProvider(ABC):
    """Abstract base provider for LLM / VLM gateways."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the provider (e.g. 'gemini', 'openrouter')."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the provider has credentials and is enabled."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        image_path: Optional[Path] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a response given a text prompt and optional image."""
        pass


import io

def _encode_image_to_base64(image_path: Path) -> tuple[str, str]:
    """Load image from path, normalize to RGB JPEG, and return (mime_type, base64_str)."""
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        from PIL import Image
        with Image.open(p) as img:
            rgb_img = img.convert("RGB")
            max_dim = max(rgb_img.size)
            if max_dim > 2048:
                scale = 2048.0 / max_dim
                new_size = (max(1, int(rgb_img.width * scale)), max(1, int(rgb_img.height * scale)))
                rgb_img = rgb_img.resize(new_size, Image.Resampling.BILINEAR)
            buf = io.BytesIO()
            rgb_img.save(buf, format="JPEG", quality=90)
            b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
            return "image/jpeg", b64_data
    except Exception as e:
        logger.warning(f"[_encode_image_to_base64] Pillow conversion fallback ({e}); using raw file read")
        mime_type = mimetypes.guess_type(str(p))[0] or "image/jpeg"
        with open(p, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        return mime_type, b64_data


class GeminiProvider(BaseAIProvider):
    """Google Gemini AI Provider using direct REST API."""

    @property
    def name(self) -> str:
        return "gemini"

    def is_available(self) -> bool:
        return bool(settings.GEMINI_API_KEY and settings.GEMINI_API_KEY.strip())

    def generate(
        self,
        prompt: str,
        image_path: Optional[Path] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if not self.is_available():
            raise AIProviderError("Gemini API key is not configured.")

        api_key = settings.GEMINI_API_KEY
        model = settings.GEMINI_MODEL or "gemini-3.6-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        parts: List[Dict[str, Any]] = [{"text": prompt}]

        if image_path and Path(image_path).exists():
            try:
                mime_type, b64_data = _encode_image_to_base64(image_path)
                parts.append({
                    "inlineData": {
                        "mimeType": mime_type,
                        "data": b64_data,
                    }
                })
            except Exception as e:
                logger.warning(f"[GeminiProvider] Could not encode image {image_path}: {e}")

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        timeout_sec = min(getattr(settings, "VQA_INFERENCE_TIMEOUT_SEC", 60) or 60, 60)
        try:
            res = requests.post(url, json=payload, timeout=timeout_sec)
            if res.status_code != 200:
                raise AIProviderError(f"Gemini API returned status {res.status_code}: {res.text}")
            data = res.json()
            if "error" in data:
                err = data["error"]
                msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                raise AIProviderError(f"Gemini API error: {msg}")
            candidates = data.get("candidates", [])
            if not candidates:
                raise AIProviderError("Gemini returned empty candidates list.")
            text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return {
                "answer": text.strip(),
                "provider": self.name,
                "model": model,
                "confidence": None,
                "evidence": [f"Gemini inference via model '{model}'"],
                "raw_meta": {"model": model, "provider": self.name},
            }
        except Exception as e:
            if isinstance(e, AIProviderError):
                raise
            err_msg = str(e)
            if api_key and api_key in err_msg:
                err_msg = err_msg.replace(api_key, "[REDACTED]")
            raise AIProviderError(f"Gemini generation failed: {err_msg}") from None


class OpenRouterProvider(BaseAIProvider):
    """OpenRouter Gateway Provider supporting multimodal vision and LLMs."""

    @property
    def name(self) -> str:
        return "openrouter"

    def is_available(self) -> bool:
        return bool(settings.OPENROUTER_API_KEY and settings.OPENROUTER_API_KEY.strip())

    def generate(
        self,
        prompt: str,
        image_path: Optional[Path] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if not self.is_available():
            raise AIProviderError("OpenRouter API key is not configured.")

        api_key = settings.OPENROUTER_API_KEY
        base_url = settings.OPENROUTER_BASE_URL.rstrip("/")
        url = f"{base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "SatQuery-AI Remote Sensing Platform",
            "Content-Type": "application/json",
        }

        # Build message content
        has_image = bool(image_path and Path(image_path).exists())
        text_content = prompt
        multimodal_content = None
        if has_image:
            try:
                mime_type, b64_data = _encode_image_to_base64(image_path)
                image_url = f"data:{mime_type};base64,{b64_data}"
                multimodal_content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
            except Exception as e:
                logger.warning(f"[OpenRouterProvider] Image read error, falling back to text: {e}")

        # Candidate models to try in sequence
        candidate_models = [
            settings.OPENROUTER_MODEL,
            "nvidia/nemotron-3.5-lightning:free",
            "google/gemma-4-31b-it:free",
            "liquid/lfm-2.5-2.6b:free",
        ]
        # De-duplicate while preserving order
        models_to_try = []
        for m in candidate_models:
            if m and m not in models_to_try:
                models_to_try.append(m)

        timeout_sec = min(getattr(settings, "VQA_INFERENCE_TIMEOUT_SEC", 30) or 30, 30)
        last_error = None

        for model in models_to_try:
            try:
                # First attempt with multimodal if image available
                content_payload = multimodal_content if multimodal_content is not None else text_content
                payload: Dict[str, Any] = {
                    "model": model,
                    "messages": [{"role": "user", "content": content_payload}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                res = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)

                # If multimodal not supported on this model (e.g. 400, 404, 422 or "support image input"), retry with text prompt
                if (res.status_code in (400, 404, 422) or "support image input" in res.text) and multimodal_content is not None:
                    logger.info(f"[OpenRouterProvider] Retrying with text-only prompt for model {model}...")
                    payload["messages"] = [{
                        "role": "user",
                        "content": f"[Satellite Context: {Path(image_path).name}]\n{prompt}",
                    }]
                    res = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)

                if res.status_code != 200:
                    logger.warning(f"[OpenRouterProvider] Model '{model}' failed (status {res.status_code}: {res.text[:120]}). Trying next candidate...")
                    last_error = f"Model {model} returned {res.status_code}"
                    continue

                data = res.json()
                if "error" in data:
                    err = data["error"]
                    msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                    logger.warning(f"[OpenRouterProvider] Model '{model}' error: {msg}. Trying next candidate...")
                    last_error = msg
                    continue

                choices = data.get("choices", [])
                if not choices:
                    continue

                msg = choices[0].get("message", {})
                text = (msg.get("content") or msg.get("reasoning") or "").strip()
                if not text:
                    continue

                return {
                    "answer": text,
                    "provider": self.name,
                    "model": data.get("model", model),
                    "confidence": 0.92,
                    "evidence": [f"OpenRouter inference via model '{data.get('model', model)}'"],
                    "raw_meta": {
                        "model": data.get("model", model),
                        "provider": self.name,
                        "id": data.get("id"),
                        "usage": data.get("usage"),
                    },
                }
            except Exception as loop_err:
                logger.warning(f"[OpenRouterProvider] Error querying model '{model}': {loop_err}")
                last_error = str(loop_err)
                continue

        raise AIProviderError(f"OpenRouter generation failed across candidate models: {last_error}")


class AIGateway:
    """Unified gateway coordinating Gemini and OpenRouter with automatic failover."""

    def __init__(self) -> None:
        self._providers: Dict[str, BaseAIProvider] = {
            "gemini": GeminiProvider(),
            "openrouter": OpenRouterProvider(),
        }

    def get_provider(self, name: str) -> Optional[BaseAIProvider]:
        return self._providers.get(name.lower())

    def list_providers(self) -> List[Dict[str, Any]]:
        """List provider statuses without exposing sensitive keys."""
        res = []
        for name, provider in self._providers.items():
            model = settings.GEMINI_MODEL if name == "gemini" else settings.OPENROUTER_MODEL
            res.append({
                "name": name,
                "available": provider.is_available(),
                "model": model,
                "is_default": (settings.AI_PROVIDER or "auto").lower() == name,
            })
        return res

    def generate(
        self,
        prompt: str,
        image_path: Optional[Path] = None,
        preferred_provider: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute inference using configured provider with automatic fallback.

        Failover contract:
          - If OpenRouter is preferred/active and encounters an error, automatically
            fall back to the existing Gemini provider.
          - If Gemini is preferred and fails, and OpenRouter is available, fall back
            to OpenRouter.
        """
        pref = (preferred_provider or settings.AI_PROVIDER or "auto").lower()

        if pref == "openrouter":
            candidates = ["openrouter", "gemini"]
        elif pref == "gemini":
            candidates = ["gemini", "openrouter"]
        else:  # "auto"
            # Prefer OpenRouter if available, else Gemini
            if self._providers["openrouter"].is_available():
                candidates = ["openrouter", "gemini"]
            else:
                candidates = ["gemini", "openrouter"]

        last_error: Optional[Exception] = None

        for p_name in candidates:
            provider = self._providers.get(p_name)
            if not provider or not provider.is_available():
                continue

            try:
                logger.info(f"[AIGateway] Attempting inference via provider: {p_name}")
                result = provider.generate(
                    prompt=prompt,
                    image_path=image_path,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                return result
            except Exception as exc:
                logger.warning(
                    f"[AIGateway] Provider '{p_name}' failed ({type(exc).__name__}: {exc}). "
                    f"Evaluating fallback candidates..."
                )
                last_error = exc
                continue

        if last_error:
            raise AIProviderError(f"All available AI providers failed. Last error: {last_error}") from last_error

        raise AIProviderError("No active AI provider is configured with valid credentials.")


_GATEWAY: Optional[AIGateway] = None


def get_ai_gateway() -> AIGateway:
    global _GATEWAY
    if _GATEWAY is None:
        _GATEWAY = AIGateway()
    return _GATEWAY
