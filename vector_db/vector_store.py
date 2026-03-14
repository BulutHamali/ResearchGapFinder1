import logging
import os
import pickle
from typing import Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-backed vector store with metadata support."""

    def __init__(self, dim: int, index_type: str = "flat"):
        self.dim = dim
        self.index_type = index_type
        self._index = self._create_index(dim, index_type)
        self._metadata: list[dict] = []
        logger.info(f"VectorStore created: dim={dim}, type={index_type}")

    def _create_index(self, dim: int, index_type: str) -> faiss.Index:
        """Create a FAISS index.

        IndexFlatIP uses inner product (= cosine similarity when vectors are normalized).
        """
        if index_type == "flat":
            return faiss.IndexFlatIP(dim)
        elif index_type == "ivf":
            # IVF index for large-scale search (requires training)
            quantizer = faiss.IndexFlatIP(dim)
            nlist = 100
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            return index
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            return index
        else:
            logger.warning(f"Unknown index_type '{index_type}', defaulting to 'flat'")
            return faiss.IndexFlatIP(dim)

    def add(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """
        Add vectors and associated metadata to the store.

        embeddings: shape (N, dim), float32
        metadata: list of N dicts
        """
        if len(embeddings) == 0:
            return

        assert embeddings.shape[1] == self.dim, (
            f"Embedding dim mismatch: expected {self.dim}, got {embeddings.shape[1]}"
        )
        assert len(embeddings) == len(metadata), (
            f"Length mismatch: {len(embeddings)} embeddings vs {len(metadata)} metadata"
        )

        vecs = np.ascontiguousarray(embeddings.astype(np.float32))

        # Train IVF index if needed
        if isinstance(self._index, faiss.IndexIVFFlat) and not self._index.is_trained:
            logger.info(f"Training IVF index on {len(vecs)} vectors")
            self._index.train(vecs)

        self._index.add(vecs)
        self._metadata.extend(metadata)
        logger.info(f"VectorStore: added {len(embeddings)} vectors (total={self.size})")

    def search(self, query_vector: np.ndarray, k: int = 10) -> list[tuple[dict, float]]:
        """
        Search for k nearest neighbors.

        Returns list of (metadata_dict, similarity_score) tuples.
        """
        if self.size == 0:
            return []

        k = min(k, self.size)
        query = np.ascontiguousarray(query_vector.reshape(1, -1).astype(np.float32))

        scores, indices = self._index.search(query, k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            results.append((self._metadata[idx], float(score)))

        return results

    def save(self, path: str) -> None:
        """Save FAISS index and metadata to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        faiss.write_index(self._index, f"{path}.faiss")
        with open(f"{path}.meta.pkl", "wb") as f:
            pickle.dump({"metadata": self._metadata, "dim": self.dim, "index_type": self.index_type}, f)
        logger.info(f"VectorStore saved to '{path}' ({self.size} vectors)")

    def load(self, path: str) -> None:
        """Load FAISS index and metadata from disk."""
        self._index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.meta.pkl", "rb") as f:
            state = pickle.load(f)
        self._metadata = state["metadata"]
        self.dim = state["dim"]
        self.index_type = state.get("index_type", "flat")
        logger.info(f"VectorStore loaded from '{path}' ({self.size} vectors)")

    @property
    def size(self) -> int:
        """Number of vectors stored."""
        return self._index.ntotal

    def get_all_embeddings(self) -> np.ndarray:
        """Reconstruct all stored embeddings (only works with IndexFlatIP)."""
        if not isinstance(self._index, faiss.IndexFlatIP) or self.size == 0:
            raise RuntimeError("get_all_embeddings only supported for IndexFlatIP")
        return faiss.rev_swig_ptr(self._index.get_xb(), self.size * self.dim).reshape(self.size, self.dim).copy()

    def get_all_metadata(self) -> list[dict]:
        """Return all stored metadata."""
        return list(self._metadata)
