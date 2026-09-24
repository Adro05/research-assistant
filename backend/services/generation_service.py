"""
Text generation.

Given a user query and already-retrieved evidence (produced by
backend/services/retrieval_service.py), generates a natural-language,
evidence-grounded answer using a Hugging Face sequence-to-sequence
generation model (default: google/flan-t5-small).

This module does NOT perform retrieval itself — it consumes an
already-retrieved list of RetrievedChunk objects and never calls FAISS
search or the embedding model. It does not implement citation mapping,
claim-level citation markers, or evidence-support classification; those
are later milestones.

This module has no dependency on FastAPI.

Design (mirrors embedding_service.py):
- The actual generation model is never imported or loaded at module
  import time. `get_generation_model()` lazily imports `transformers`
  and loads the configured tokenizer + model, caching one instance per
  model name so it is not reloaded on every call.
- `generate_answer()` accepts an injectable `model` argument satisfying
  the `GenerationModel` protocol, so tests can supply a lightweight
  deterministic fake instead of a real Hugging Face model.
- Prompt construction is a separate, deterministic, directly-testable
  helper (`_build_prompt`), which enforces an explicit token budget by
  checking the token count of the actual assembled candidate prompt
  (instruction + accepted evidence so far + next candidate chunk +
  question/answer scaffolding) against the model's real
  `max_input_tokens` — not by summing independently-tokenized chunk
  counts, since tokenization is not guaranteed to be perfectly additive
  across separately-tokenized strings. This guarantees the prompt this
  function returns is itself within the model's real token budget,
  according to the model's own tokenizer, in every case that does not
  raise (see EvidenceContextTooLargeError).
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from backend.core.config import get_settings
from backend.services.retrieval_service import RetrievedChunk

# Reasonable, documented fallback used only when a loaded tokenizer does
# not report a specific model_max_length. Hugging Face tokenizers set an
# enormous sentinel value (commonly ~1e30) when no limit is configured;
# 512 is the standard context length for T5/FLAN-T5-family checkpoints,
# including google/flan-t5-small, and is used only as a fallback — the
# real value is read from the actual loaded tokenizer whenever possible.
DEFAULT_MAX_INPUT_TOKENS = 512
_UNSET_SENTINEL_THRESHOLD = 100_000


class GenerationError(Exception):
    """Base class for generation failures."""


class GenerationModelUnavailableError(GenerationError):
    """Raised when the generation model cannot be loaded (missing dependency, download failure, etc.)."""


class GenerationFailedError(GenerationError):
    """Raised when the model fails to produce an answer, or produces empty/unusable output."""


class NoEvidenceError(GenerationError):
    """
    Raised when there is no retrieved evidence to ground an answer in.

    Generating an answer from zero evidence would necessarily be
    ungrounded (invented rather than evidence-based), which this
    project's design explicitly avoids — so this is a distinct, explicit
    failure, and the generation model is never called in this case.
    """


class EvidenceContextTooLargeError(GenerationError):
    """
    Raised when evidence was supplied, but not even the single
    highest-ranked chunk fits within the generation model's actual input
    token budget alongside the fixed instruction/question/answer
    scaffolding.

    Evidence chunks are never truncated to force a fit (truncating would
    risk cutting evidence mid-sentence and silently changing its
    meaning), so when this occurs there is no way to construct a
    genuinely evidence-grounded prompt under the configured model's
    context limit. The generation model is never called in this case.
    """


class GenerationModel(Protocol):
    """Structural interface any generation model (real or fake) must satisfy."""

    def generate(self, prompt: str) -> str: ...
    def count_tokens(self, text: str) -> int: ...

    @property
    def max_input_tokens(self) -> int: ...


@dataclass
class GeneratedAnswer:
    """A generated answer, together with the exact evidence it was grounded in."""

    document_id: str
    query: str
    answer: str
    evidence: list[RetrievedChunk]


def _resolve_max_input_tokens(tokenizer) -> int:
    """Read the actual configured max input length from a loaded tokenizer, with a documented fallback."""
    limit = getattr(tokenizer, "model_max_length", None)
    if limit is None or limit > _UNSET_SENTINEL_THRESHOLD:
        return DEFAULT_MAX_INPUT_TOKENS
    return int(limit)


class _HuggingFaceGenerationModel:
    """
    Wraps a Hugging Face seq2seq tokenizer + model pair behind the
    GenerationModel protocol.

    `max_input_tokens` is inspected from the actual loaded tokenizer at
    construction time (not hardcoded), and `count_tokens()` uses that
    same tokenizer, so prompt budgeting in `_build_prompt` reflects the
    real model's real configuration rather than an estimate.
    """

    def __init__(self, tokenizer, model, max_new_tokens: int) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._max_new_tokens = max_new_tokens
        self._max_input_tokens = _resolve_max_input_tokens(tokenizer)

    @property
    def max_input_tokens(self) -> int:
        return self._max_input_tokens

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=True))

    def generate(self, prompt: str) -> str:
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_input_tokens,
        )
        output_ids = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens)
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)


@lru_cache(maxsize=4)
def get_generation_model(model_name: str) -> "GenerationModel":
    """
    Lazily load and cache a Hugging Face seq2seq generation model by name.

    Cached so the (potentially expensive) model load happens at most once
    per distinct `model_name` per process, rather than once per call —
    the same pattern as embedding_service.get_embedding_model().
    """
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise GenerationModelUnavailableError("transformers is not installed.") from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception as exc:
        raise GenerationModelUnavailableError(f"Failed to load generation model '{model_name}'.") from exc

    settings = get_settings()
    return _HuggingFaceGenerationModel(tokenizer, model, settings.generation_max_new_tokens)


def _build_prompt(
    query: str,
    evidence: list[RetrievedChunk],
    model: "GenerationModel",
) -> str:
    """
    Deterministically build a grounded-answer prompt from ranked evidence.

    Evidence is included in its existing ranked order (best match first,
    exactly as produced by retrieval). Inclusion is decided by building
    the actual complete candidate prompt (current evidence + the next
    chunk) and checking `model.count_tokens()` on that *whole assembled
    prompt* against `model.max_input_tokens` — not by summing
    independently-tokenized chunk counts, since tokenization is not
    guaranteed to be perfectly additive across separately-tokenized
    strings. This guarantees the prompt this function returns is itself
    within the model's real token budget, according to the model's own
    tokenizer, in every case that does not raise.

    A chunk is only ever included in full or excluded entirely — chunks
    are never truncated mid-text. As soon as one candidate chunk would
    push the assembled prompt over budget, no further (lower-ranked)
    chunks are considered either, preserving a simple, deterministic,
    order-respecting cutoff rather than skipping ahead to find a smaller
    chunk that might fit.

    The instruction text is deliberately explicit that the answer must
    come only from the supplied context, and must not invent information
    not present in it. No claim-level citation markers are added here —
    that is a later milestone.

    Raises:
        EvidenceContextTooLargeError: `evidence` is non-empty, but not
            even the single highest-ranked chunk fits within the model's
            budget. This function never silently falls back to returning
            a prompt with no evidence when evidence was actually
            supplied — that would call the generation model with
            effectively no grounding, which this project's design
            explicitly avoids.
    """
    instruction = (
        "Answer the question using only the context provided below. "
        "Do not use any outside knowledge, and do not invent information "
        "that is not present in the context. If the context does not "
        "contain enough information to answer, say so explicitly.\n\n"
    )
    question_section = f"\n\nQuestion:\n{query}\n\nAnswer:"

    def _assemble(chunk_texts: list[str]) -> str:
        context_section = "\n\n".join(chunk_texts)
        return f"{instruction}Context:\n{context_section}{question_section}"

    included_chunks: list[str] = []
    for chunk in evidence:
        candidate_chunks = included_chunks + [chunk.text.strip()]
        candidate_prompt = _assemble(candidate_chunks)
        if model.count_tokens(candidate_prompt) <= model.max_input_tokens:
            included_chunks = candidate_chunks
        else:
            break  # deterministic cutoff: stop at the first chunk that no longer fits

    if evidence and not included_chunks:
        raise EvidenceContextTooLargeError(
            "The highest-ranked evidence chunk alone exceeds the generation "
            "model's input token budget; cannot construct an evidence-grounded "
            "prompt without truncating evidence, which is not permitted."
        )

    return _assemble(included_chunks)


def generate_answer(
    query: str,
    evidence: list[RetrievedChunk],
    model: "GenerationModel",
    document_id: str | None = None,
) -> GeneratedAnswer:
    """
    Generate an evidence-grounded answer to `query` from already-retrieved
    `evidence`.

    Does not perform retrieval itself — `evidence` must already be the
    output of retrieval_service.retrieve() (or an equivalent list of
    RetrievedChunk objects). Does not duplicate FAISS search or query
    embedding logic.

    Raises:
        NoEvidenceError: `evidence` is empty. The generation model is
            never called in this case.
        EvidenceContextTooLargeError: `evidence` is non-empty, but not
            even the single highest-ranked chunk fits within the model's
            actual input token budget. Propagated unmodified from
            `_build_prompt()`. The generation model is never called in
            this case either.
        GenerationFailedError: the model raised while generating, or
            returned an empty/unusable answer.
    """
    if not evidence:
        raise NoEvidenceError("Cannot generate an evidence-grounded answer with no retrieved evidence.")

    prompt = _build_prompt(query, evidence, model)

    try:
        answer = model.generate(prompt)
    except GenerationError:
        raise
    except Exception as exc:
        raise GenerationFailedError("Failed to generate an answer from the model.") from exc

    if not isinstance(answer, str) or not answer.strip():
        raise GenerationFailedError("The generation model returned an empty answer.")

    return GeneratedAnswer(
        document_id=document_id if document_id is not None else evidence[0].document_id,
        query=query,
        answer=answer.strip(),
        evidence=evidence,
    )