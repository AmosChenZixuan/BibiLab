"""ChromaDB chunk embedding step."""

import logging
from pathlib import Path

from locus.adapters.base import VideoMeta
from locus.config import LocusConfig, locus_home
from locus.pipeline.chunk import RagChunk

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "locus_transcripts"


def _embedding_model_dir() -> Path:
    return locus_home() / "models" / "embedding"


def _default_embedding_function():
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2  # noqa: PLC0415

    class LocalONNXMiniLM(ONNXMiniLM_L6_V2):
        DOWNLOAD_PATH = _embedding_model_dir()

    _embedding_model_dir().mkdir(parents=True, exist_ok=True)
    return LocalONNXMiniLM()


def _get_collection(cfg: LocusConfig):
    import chromadb  # noqa: PLC0415

    client = chromadb.PersistentClient(path=str(locus_home() / "chroma"))
    return client.get_or_create_collection(
        _COLLECTION_NAME,
        embedding_function=_default_embedding_function(),
    )


def embed_chunks(
    chunks: list[RagChunk],
    meta: VideoMeta,
    list_id: str,
    cfg: LocusConfig,
) -> None:
    if not chunks:
        return

    collection = _get_collection(cfg)

    # Remove any existing chunks for this video (idempotent re-run)
    try:
        collection.delete(where={"note_id": meta.video_id})
    except Exception:
        pass  # collection may be empty

    ids = [f"{meta.video_id}_{chunk.sequence_index}" for chunk in chunks]
    documents = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "note_id": meta.video_id,
            "list_id": list_id,
            "video_title": meta.title,
            "timestamp_start": chunk.timestamp_start,
            "timestamp_end": chunk.timestamp_end,
            "sequence_index": chunk.sequence_index,
        }
        for chunk in chunks
    ]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    logger.info("Embedded %d chunks for %s", len(chunks), meta.video_id)


def clear_embeddings_for_list(list_id: str, cfg: LocusConfig) -> None:
    """Delete all ChromaDB chunks belonging to the given list."""
    collection = _get_collection(cfg)
    try:
        collection.delete(where={"list_id": list_id})
    except Exception:
        pass
