import json
import logging
import re
from typing import Optional

from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class LLMReasoner:
    """Wrapper around Groq API for LLM reasoning tasks."""

    def __init__(self, settings):
        self.settings = settings
        preset = settings.get_preset()
        self.model = preset["llm_model"]
        self.default_max_tokens = preset["max_tokens"]
        self.default_temperature = preset["temperature"]
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Call Groq API and return the response text."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        max_tok = max_tokens if max_tokens is not None else self.default_max_tokens
        temp = temperature if temperature is not None else self.default_temperature

        logger.debug(f"Calling LLM model={self.model}, max_tokens={max_tok}, temp={temp}")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tok,
            temperature=temp,
        )

        content = response.choices[0].message.content
        logger.debug(f"LLM response length: {len(content)} chars")
        return content

    def complete_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """Call complete() and parse JSON from the response.

        Handles markdown code blocks around JSON.
        """
        raw = self.complete(prompt, system=system, max_tokens=max_tokens, temperature=temperature)
        return self._parse_json(raw)

    def _parse_json(self, text: str) -> dict:
        """Parse JSON from text, stripping markdown code blocks if present."""
        text = text.strip()

        # Try to strip markdown code fences
        # Handles ```json ... ``` or ``` ... ```
        fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
        match = fence_pattern.search(text)
        if match:
            text = match.group(1).strip()

        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object or array in the text
        obj_match = re.search(r"\{[\s\S]*\}", text)
        arr_match = re.search(r"\[[\s\S]*\]", text)

        # Prefer the one that appears first
        candidates = []
        if obj_match:
            candidates.append((obj_match.start(), obj_match.group(0)))
        if arr_match:
            candidates.append((arr_match.start(), arr_match.group(0)))

        candidates.sort(key=lambda x: x[0])

        for _, candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        logger.error(f"Failed to parse JSON from LLM response: {text[:500]}")
        raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}")
