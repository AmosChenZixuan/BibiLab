"""ChromaDB chunk embedding step."""

import logging

from locus.adapters.base import VideoMeta
from locus.config import LocusConfig, locus_home
from locus.pipeline.chunk import RagChunk

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "locus_transcripts"


def _get_collection(cfg: LocusConfig):
    import chromadb  # noqa: PLC0415

    client = chromadb.PersistentClient(path=str(locus_home() / "chroma"))
    return client.get_or_create_collection(_COLLECTION_NAME)


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
