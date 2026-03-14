import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Singleton registry: model_name -> SentenceTransformer instance
_model_registry: dict[str, "SentenceTransformer"] = {}

BATCH_SIZE = 64
TQDM_THRESHOLD = 100  # Show progress bar only for batches larger than this


class Embedder:
    """Encode texts into normalized float32 embeddings using SentenceTransformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = self._get_or_load_model(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()
        logger.info(f"Embedder initialized: model='{model_name}', dim={self.dim}")

    def _get_or_load_model(self, model_name: str) -> SentenceTransformer:
        """Return a cached model instance or load a new one."""
        if model_name not in _model_registry:
            logger.info(f"Loading embedding model: '{model_name}'")
            _model_registry[model_name] = SentenceTransformer(model_name)
            logger.info(f"Model '{model_name}' loaded and cached")
        return _model_registry[model_name]

    def embed(self, texts: list[str], show_progress: Optional[bool] = None) -> np.ndarray:
        """
        Encode a list of texts into normalized float32 embeddings.

        Returns shape (N, dim) float32 array.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        use_progress = show_progress if show_progress is not None else len(texts) > TQDM_THRESHOLD

        logger.info(f"Embedding {len(texts)} texts with model '{self.model_name}'")

        embeddings = self._model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=use_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Unit-normalize for cosine via dot product
        )

        embeddings = embeddings.astype(np.float32)
        logger.info(f"Embedding complete: shape={embeddings.shape}")
        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text. Returns shape (dim,) float32 array."""
        embedding = self._model.encode(
            [text],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding[0].astype(np.float32)

    @staticmethod
    def papers_to_texts(papers: list[dict]) -> list[str]:
        """Convert paper dicts to text strings for embedding."""
        texts = []
        for paper in papers:
            title = paper.get("title", "").strip()
            abstract = paper.get("abstract", "").strip()
            if title and abstract:
                texts.append(f"{title}. {abstract}")
            elif title:
                texts.append(title)
            elif abstract:
                texts.append(abstract)
            else:
                texts.append("")
        return texts
