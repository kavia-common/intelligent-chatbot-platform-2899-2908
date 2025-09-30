"""
OpenAI client utilities for the Company Chatbot Backend.

This module centralizes OpenAI API access, loading the API key from the environment
and providing safe, validated methods to call the API for RAG/Agentic AI flows.
"""

import os
import logging
from typing import List, Dict, Any, Optional

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger("uvicorn.error")


class OpenAISettings(BaseModel):
    """Configuration for OpenAI integration loaded from environment variables."""
    api_key: Optional[str] = Field(default=None, description="OpenAI API key taken from OPENAI_API_KEY")
    base_url: str = Field(default="https://api.openai.com/v1", description="OpenAI API base URL (override for proxies)")
    # Default model can be overridden via env OPENAI_MODEL
    default_model: str = Field(default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), description="Default model to use")


def get_openai_settings() -> OpenAISettings:
    """Load OpenAI settings from environment variables."""
    key = os.environ.get("OPENAI_API_KEY")  # Do not hardcode; must be provided by the user in .env
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return OpenAISettings(api_key=key, base_url=base_url)


# PUBLIC_INTERFACE
def ensure_openai_key() -> str:
    """Ensure an OpenAI API key is available in the environment.

    Raises:
        RuntimeError: If OPENAI_API_KEY is missing or empty.

    Returns:
        str: The API key value.
    """
    settings = get_openai_settings()
    if not settings.api_key or not settings.api_key.strip():
        logger.error("OPENAI_API_KEY is missing. Set it in your environment to enable OpenAI features.")
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
    return settings.api_key.strip()


# PUBLIC_INTERFACE
def openai_chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = 512,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Perform a chat completion request to OpenAI securely using env API key.

    Args:
        messages: List of {role: 'system'|'user'|'assistant', content: str}
        model: Override model name; falls back to env default.
        temperature: Sampling temperature.
        max_tokens: Optional cap for output tokens.
        extra_headers: Optional extra headers for custom gateways.

    Returns:
        Dict with response data including 'content' for assistant text.

    Raises:
        RuntimeError: If API key missing or on HTTP error from OpenAI.
    """
    settings = get_openai_settings()
    api_key = ensure_openai_key()
    mdl = model or settings.default_model

    url = f"{settings.base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    payload: Dict[str, Any] = {
        "model": mdl,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code == 401:
                logger.error("OpenAI authentication failed (401). Check OPENAI_API_KEY validity.")
                raise RuntimeError("Invalid OPENAI_API_KEY or unauthorized access.")
            if resp.status_code >= 400:
                logger.error(f"OpenAI API error {resp.status_code}: {resp.text}")
                raise RuntimeError(f"OpenAI API error: {resp.status_code}")

            data = resp.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return {
                "model": data.get("model", mdl),
                "content": content,
                "raw": data,
            }
    except httpx.RequestError as e:
        logger.exception("Network error while calling OpenAI: %s", str(e))
        raise RuntimeError("Network error while calling OpenAI API.") from e
