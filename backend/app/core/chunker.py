"""
Recursive chunker: splits by paragraphs into sentences and further into words.
"""
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass(frozen=True)
class Chunk:
  text: str
  metadata: Dict = field(default_factory=dict)
  token_count: int = 0


class RecursiveChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _count_tokens(self, text: str) -> int:
        return int(len(text.split()) / 0.75)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex (no NLTK dependency)."""
        import re
        # Split on period, question mark, or exclamation followed by space or end to maintain natural boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s.strip()]

    def chunk(self, text: str, doc_metadata: Dict = None) -> List[Chunk]:
        doc_metadata = doc_metadata or {}
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        # Flatten paragraphs into sentences
        sentences = []
        for para in paragraphs:
            sentences.extend(self._split_into_sentences(para))

        chunks = []
        current_sentences = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            if current_tokens + sentence_tokens > self.chunk_size and current_sentences:
                # Finalize current chunk
                chunk_text = " ".join(current_sentences)
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={
                        **doc_metadata,
                        "chunk_index": len(chunks),
                        "token_count": current_tokens,
                    },
                    token_count=current_tokens,
                ))

                # Overlap: keep last few sentences that fit within overlap budget - to keep continuity (important for context)
                overlap_sentences = []
                overlap_tokens = 0
                for s in reversed(current_sentences):
                    s_tokens = self._count_tokens(s)
                    if overlap_tokens + s_tokens <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break
                current_sentences = overlap_sentences
                current_tokens = overlap_tokens

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    **doc_metadata,
                    "chunk_index": len(chunks),
                    "token_count": self._count_tokens(chunk_text),
                },
                token_count=self._count_tokens(chunk_text),
            ))

        return chunks