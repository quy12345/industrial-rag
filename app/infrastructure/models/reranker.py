"""Lazy FastEmbed cross-encoder adapter."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from app.domain.reranking import CrossEncoderScore, RerankingError


class FastEmbedCrossEncoder:
    """Lazy adapter for FastEmbed 0.8 TextCrossEncoder's input-ordered scores."""

    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: str | None = None,
        threads: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads
        self._model: Any | None = None

    def score(
        self, query: str, documents: Sequence[str], *, batch_size: int
    ) -> Iterable[CrossEncoderScore]:
        try:
            model = self._get_model()
            scores = model.rerank(query, documents, batch_size=batch_size)
            return [
                CrossEncoderScore(candidate_index=index, score=float(score))
                for index, score in enumerate(scores)
            ]
        except RerankingError:
            raise
        except Exception as exc:
            raise RerankingError(f"Cross-encoder inference failed: {exc}") from exc

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                self._model = TextCrossEncoder(
                    model_name=self.model_name,
                    cache_dir=self.cache_dir,
                    threads=self.threads,
                    cuda=False,
                    lazy_load=True,
                )
            except Exception as exc:
                raise RerankingError(f"Unable to initialize cross-encoder: {exc}") from exc
        return self._model


def fastembed_model_metadata(model_name: str) -> dict[str, Any]:
    """Return FastEmbed's local registry metadata without constructing/downloading a model."""

    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        metadata = next(
            (
                item
                for item in TextCrossEncoder.list_supported_models()
                if item.get("model", "").casefold() == model_name.casefold()
            ),
            None,
        )
    except Exception as exc:
        raise RerankingError(f"Unable to inspect FastEmbed cross-encoder registry: {exc}") from exc
    if metadata is None:
        raise RerankingError(f"FastEmbed does not support rerank model: {model_name}")
    return metadata
