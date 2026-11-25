"""
mongo_store.py
==============
MongoDB helpers for saving and reading live prediction records.
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timezone

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)

DEFAULT_DB_NAME = os.environ.get("MONGODB_DB_NAME", "stock_predictor")
DEFAULT_COLLECTION_NAME = os.environ.get("MONGODB_COLLECTION_NAME", "predictions")
# Phase 4 keeps LSTM records in their own collection so the existing Random
# Forest records and dashboard conventions are not disturbed.
DEFAULT_LSTM_COLLECTION_NAME = os.environ.get(
    "MONGODB_LSTM_COLLECTION_NAME", "lstm_predictions"
)


def connect_mongodb(uri: str | None = None, db_name: str | None = None, collection_name: str | None = None) -> Collection:
    mongo_uri = uri or os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    db = db_name or DEFAULT_DB_NAME
    collection = collection_name or DEFAULT_COLLECTION_NAME
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client[db][collection]


def save_prediction(
    symbol: str,
    prediction: str,
    confidence: float,
    source: str,
    model: str = "Random Forest",
    meta: dict | None = None,
    collection: Collection | None = None,
) -> str:
    payload = {
        "symbol": symbol,
        "prediction": prediction,
        "confidence": round(float(confidence), 4),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "model": model,
    }
    if meta:
        payload["meta"] = meta

    try:
        col = collection or connect_mongodb()
        result = col.insert_one(payload)
        return str(result.inserted_id)
    except PyMongoError as exc:
        logger.warning("Failed to save prediction in MongoDB: %s", exc)
        return ""


def fetch_recent_predictions(limit: int = 20, collection: Collection | None = None) -> list[dict]:
    try:
        col = collection or connect_mongodb()
        rows = list(col.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
        return rows
    except PyMongoError as exc:
        logger.warning("Failed to fetch recent predictions from MongoDB: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Phase 4 — LSTM inference records (deduplicated)
# ---------------------------------------------------------------------------

def connect_lstm_collection(collection: Collection | None = None) -> Collection:
    """Return the dedicated LSTM prediction collection."""
    if collection is not None:
        return collection
    return connect_mongodb(collection_name=DEFAULT_LSTM_COLLECTION_NAME)


def ensure_lstm_indexes(collection: Collection | None = None) -> None:
    """Create the unique dedup index and the timestamp index (idempotent)."""
    try:
        col = connect_lstm_collection(collection)
        col.create_index("dedup_key", unique=True)
        col.create_index([("timestamp", -1)])
    except PyMongoError as exc:
        logger.warning("Could not create LSTM prediction indexes: %s", exc)


def save_lstm_prediction(record: dict, collection: Collection | None = None) -> str:
    """
    Insert one LSTM inference record, returning its document id.

    A deterministic ``dedup_key`` (symbol + exact model weights + input session
    + feature schema) prevents duplicate records on repeated dashboard clicks.
    Re-running the same inference returns the existing id and inserts nothing.
    MongoDB is optional: an unavailable database never breaks inference.
    """
    key = record.get("dedup_key")
    if not key:
        raise ValueError("LSTM prediction record is missing its dedup_key")

    try:
        col = connect_lstm_collection(collection)
        ensure_lstm_indexes(col)

        existing = col.find_one({"dedup_key": key}, {"_id": 1})
        if existing is not None:
            logger.info("Duplicate LSTM prediction skipped (dedup_key=%s)", key[:12])
            return str(existing["_id"])

        result = col.insert_one(dict(record))
        return str(result.inserted_id)
    except PyMongoError as exc:
        logger.warning("Failed to save LSTM prediction in MongoDB: %s", exc)
        return ""


def lstm_prediction_exists(
    dedup_key: str, collection: Collection | None = None
) -> bool:
    """Return True when a prediction with this dedup key is already stored."""
    if not dedup_key:
        return False
    try:
        col = connect_lstm_collection(collection)
        return col.find_one({"dedup_key": dedup_key}, {"_id": 1}) is not None
    except PyMongoError as exc:
        logger.warning("Could not check LSTM prediction duplicate: %s", exc)
        return False


def update_lstm_prediction_meta(
    dedup_key: str,
    meta_updates: dict,
    collection: Collection | None = None,
) -> bool:
    """
    Attach pipeline provenance to an existing deduplicated prediction.

    The Phase 4 dedup key deliberately makes one record per symbol + input
    session + exact weights. Re-running the distributed pipeline for the same
    session must not create a duplicate, but it should record the new HDFS /
    Spark run provenance. Returns True when a document was updated.
    """
    if not dedup_key or not meta_updates:
        return False
    try:
        col = connect_lstm_collection(collection)
        result = col.update_one(
            {"dedup_key": dedup_key},
            {"$set": {f"meta.{k}": v for k, v in meta_updates.items()}},
        )
        return bool(getattr(result, "matched_count", 0))
    except PyMongoError as exc:
        logger.warning("Could not update LSTM prediction provenance: %s", exc)
        return False


def fetch_lstm_predictions(
    limit: int = 20,
    symbol: str | None = None,
    collection: Collection | None = None,
) -> list[dict]:
    """Fetch recent LSTM inference records, newest first."""
    try:
        col = connect_lstm_collection(collection)
        query = {"symbol": symbol} if symbol else {}
        return list(col.find(query, {"_id": 0}).sort("timestamp", -1).limit(limit))
    except PyMongoError as exc:
        logger.warning("Failed to fetch LSTM predictions from MongoDB: %s", exc)
        return []

