"""
Abstraction over OpenAI / Anthropic. Swap providers via config.
"""
from backend.app.config import settings
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        elif self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.client = genai.GenerativeModel(settings.LLM_MODEL)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")
        logger.info(f"LLM Service initialized: {self.provider} / {settings.LLM_MODEL}")

    def generate(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Generate a cited answer given a query and retrieved context chunks.
        Each chunk dict has 'text', 'source', 'score'.
        """
        # Build context block
        context_block = "\n\n".join(
            f"[Source {i+1}: {c['source']}]\n{c['text']}"
            for i, c in enumerate(context_chunks)
        )

        system_prompt = (
            "You are a helpful document Q&A assistant. Answer the user's question "
            "based ONLY on the provided context. Cite your sources using [Source N] "
            "notation. If the context doesn't contain enough information to answer, "
            "say so explicitly. Do not make up information."
        )

        user_prompt = (
            f"Context:\n{context_block}\n\n"
            f"Question: {query}\n\n"
            f"Provide a detailed answer with citations."
        )

        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=settings.LLM_MODEL,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.2,
                max_tokens=1024,
            )
            return response.content[0].text

        elif self.provider == "gemini":
            # Gemini combines system + user into a single prompt
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self.client.generate_content(
                full_prompt,
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 1024,
                },
            )
            return response.text