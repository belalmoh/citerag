
import asyncio
import logging

from openai import APIError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel

from app.core.config import get_settings


logger = logging.getLogger(__name__)

MAX_RETRIES = 3
INITIAL_BACKOFF = 1

class LLMResponse(BaseModel):
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int

class LLMClient:
    def __init__(self, model: str = None):
        settings = get_settings()
        if model is None:
            self.model = model or settings.openai_chat_model
        self._client = AsyncOpenAI(base_url=settings.openai_base_url, api_key=settings.openai_api_key)

    async def generate(self, user_message: str, system_prompt: str | None = None, temperature: float = 0.1, max_tokens: int = 1024) -> LLMResponse:
        
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant that answers questions "
                "based ONLY on the provided context. If the context "
                "does not contain enough information to answer the "
                "question, say so clearly. Do not make up facts."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        retries = 0
        backoff = INITIAL_BACKOFF

        while retries < MAX_RETRIES:
            try:
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return LLMResponse(
                    text=response.choices[0].message.content,
                    model=self.model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                )
            except RateLimitError as e:
                logger.warning(f"Rate limit hit, retrying in {backoff} seconds... (Attempt {retries + 1}/{MAX_RETRIES})")
                backoff = INITIAL_BACKOFF * (2 ** retries)
                retries += 1
                await asyncio.sleep(backoff)
            except APIError as e:
                logger.error(f"API error: {e}")
                raise e

        raise Exception("Failed to generate response after multiple attempts due to rate limits.")