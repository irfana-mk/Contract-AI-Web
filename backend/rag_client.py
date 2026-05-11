import os
import re
import uuid
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from extractor import extract_text_from_pdf

QDRANT_PATH = os.path.join(os.path.dirname(__file__), "qdrant_data")
COLLECTION_NAME = "conditions_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384

os.makedirs(QDRANT_PATH, exist_ok=True)

_client: Optional[QdrantClient] = None
_embedder: Optional[SentenceTransformer] = None


def get_client() -> Optional[QdrantClient]:
    global _client
    if _client is None:
        try:
            _client = QdrantClient(path=QDRANT_PATH)
            existing_names = [c.name for c in _client.get_collections().collections]
            if COLLECTION_NAME not in existing_names:
                _client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
                )
        except Exception as e:
            print(f"Qdrant unavailable (another instance may be running): {e}")
            return None
    return _client


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def get_conditions_count() -> int:
    try:
        client = get_client()
        if client is None:
            return 0
        info = client.get_collection(COLLECTION_NAME)
        return info.points_count or 0
    except Exception:
        return 0


def parse_policy_tiers(text: str) -> list[dict]:
    """Parse tier rows from pipe-delimited policy text. Returns list of tier dicts."""
    tiers = []
    for line in text.split('\n'):
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 3:
            continue
        name_col, uptime_col, fee_col = parts[0], parts[1], parts[2]
        uptime_match = re.search(r'(\d+\.?\d*)\s*%', uptime_col)
        fee_digits = re.sub(r'[^0-9]', '', fee_col)
        if not uptime_match or not fee_digits:
            continue
        try:
            uptime = float(uptime_match.group(1))
            fee = int(fee_digits)
            if uptime > 0 and fee > 1000 and len(name_col) > 3:
                tiers.append({"name": name_col, "uptime_percent": uptime, "monthly_fee": fee})
        except ValueError:
            continue
    return tiers


def index_conditions_pdf(pdf_path: str) -> tuple[bool, str]:
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        return False, ""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    if not chunks:
        return False, ""

    embedder = get_embedder()
    client = get_client()
    if client is None:
        print("Qdrant unavailable — skipping index.")
        return False, ""

    # Replace existing collection with fresh data
    client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
    points = [
        PointStruct(
            id=uuid.uuid4().int >> 64,
            vector=embedding,
            payload={"text": chunk, "chunk_index": i},
        )
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Indexed {len(points)} chunks into Qdrant.")
    return True, text


def query_conditions(field_names: list[str], n_results: int = 4) -> str:
    """Query Qdrant for policy chunks relevant to each field name."""
    try:
        if get_conditions_count() == 0:
            return ""

        client = get_client()
        if client is None:
            return ""

        embedder = get_embedder()
        seen_chunks: set[str] = set()
        all_chunks: list[str] = []

        for field_name in field_names:
            query = field_name.replace("_", " ")
            query_vector = embedder.encode(query).tolist()

            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=n_results,
            ).points

            for hit in results:
                if hit.payload and "text" in hit.payload:
                    chunk_text = hit.payload["text"]
                    chunk_key = chunk_text[:100]
                    if chunk_key not in seen_chunks:
                        seen_chunks.add(chunk_key)
                        all_chunks.append(chunk_text)

        print(f"RAG retrieved {len(all_chunks)} unique chunks for {len(field_names)} fields.")
        return "\n\n---\n\n".join(all_chunks)
    except Exception as e:
        print(f"RAG query failed (skipping): {e}")
        return ""
