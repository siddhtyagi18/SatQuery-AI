import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import get_settings
from ..logging_setup import logger

settings = get_settings()

_firestore_client = None
_storage_bucket = None
_firebase_initialized = False
_firebase_error: Optional[str] = None


def _load_service_account_info() -> Optional[Dict[str, Any]]:
    sa = settings.FIREBASE_SERVICE_ACCOUNT_JSON
    if not sa:
        return None
    sa = sa.strip()
    if sa.startswith("@"):
        path = Path(sa[1:])
        if not path.exists():
            raise FileNotFoundError(f"Firebase service account file not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        return json.loads(sa)
    except json.JSONDecodeError as e:
        raise ValueError(f"FIREBASE_SERVICE_ACCOUNT_JSON is not valid JSON: {e}")


def _ensure_initialized() -> bool:
    global _firestore_client, _storage_bucket, _firebase_initialized, _firebase_error

    if _firebase_initialized:
        return _firestore_client is not None

    if not settings.FIREBASE_ENABLED:
        logger.info("Firebase disabled (FIREBASE_ENABLED=false). Using only local persistence.")
        _firebase_initialized = True
        return False

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore, storage
    except Exception as e:
        _firebase_error = f"firebase-admin package not installed: {e}"
        logger.warning(_firebase_error + " — falling back to local persistence.")
        _firebase_initialized = True
        return False

    try:
        if not firebase_admin._apps:
            sa_info = _load_service_account_info()
            opts: Dict[str, Any] = {"projectId": settings.FIREBASE_PROJECT_ID}
            if settings.FIREBASE_STORAGE_BUCKET:
                opts["storageBucket"] = settings.FIREBASE_STORAGE_BUCKET
            if sa_info:
                cred = credentials.Certificate(sa_info)
                firebase_admin.initialize_app(cred, opts)
            else:
                firebase_admin.initialize_app(None, opts)

        _firestore_client = firestore.client()

        if settings.FIREBASE_STORAGE_BUCKET:
            _storage_bucket = storage.bucket(settings.FIREBASE_STORAGE_BUCKET)
        else:
            _storage_bucket = None

        _firebase_initialized = True
        logger.info(f"Firebase initialized (project={settings.FIREBASE_PROJECT_ID}, storage={_storage_bucket is not None})")
        return True
    except Exception as e:
        _firebase_error = f"Firebase initialization failed: {e}"
        logger.warning(_firebase_error + " — falling back to local persistence.")
        _firebase_initialized = True
        return False


def is_firebase_enabled() -> bool:
    return _ensure_initialized()


def get_firestore_error() -> Optional[str]:
    _ensure_initialized()
    return _firebase_error


def firestore_db():
    _ensure_initialized()
    return _firestore_client


def storage_bucket():
    _ensure_initialized()
    return _storage_bucket


class FirebaseRepository:
    """Repository layer over Firestore for analysis records.

    All methods degrade gracefully: if Firebase is disabled or unavailable,
    operations are no-ops that return None so the caller (SQLite persistence
    layer) still functions correctly.
    """

    COLLECTION = settings.FIREBASE_COLLECTION_ANALYSES

    def save_analysis(self, analysis_id: str, payload: Dict[str, Any]) -> Optional[bool]:
        """Persist a complete analysis record (JSON-serializable dict)."""
        try:
            if not _ensure_initialized():
                return None
            db = firestore_db()
            if db is None:
                return None
            db.collection(self.COLLECTION).document(analysis_id).set(payload)
            return True
        except Exception as e:
            logger.warning(f"Firestore save_analysis({analysis_id}) failed: {e}")
            return None

    def get_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        try:
            if not _ensure_initialized():
                return None
            db = firestore_db()
            if db is None:
                return None
            doc = db.collection(self.COLLECTION).document(analysis_id).get()
            return doc.to_dict() if doc.exists else None
        except Exception as e:
            logger.warning(f"Firestore get_analysis({analysis_id}) failed: {e}")
            return None

    def update_analysis(self, analysis_id: str, updates: Dict[str, Any]) -> Optional[bool]:
        try:
            if not _ensure_initialized():
                return None
            db = firestore_db()
            if db is None:
                return None
            db.collection(self.COLLECTION).document(analysis_id).update(updates)
            return True
        except Exception as e:
            logger.warning(f"Firestore update_analysis({analysis_id}) failed: {e}")
            return None


class FirebaseStorageService:
    """Repository layer over Firebase Cloud Storage for upload blobs."""

    def upload_file(self, local_path: Path, storage_path: str, content_type: Optional[str] = None) -> Optional[str]:
        try:
            if not _ensure_initialized():
                return None
            bucket = storage_bucket()
            if bucket is None:
                return None
            blob = bucket.blob(storage_path)
            blob.upload_from_filename(str(local_path), content_type=content_type)
            blob.make_public()
            return blob.public_url
        except Exception as e:
            logger.warning(f"Firebase Storage upload_file({local_path.name} → {storage_path}) failed: {e}")
            return None

    def download_to_temp(self, storage_path: str) -> Optional[Path]:
        try:
            if not _ensure_initialized():
                return None
            bucket = storage_bucket()
            if bucket is None:
                return None
            blob = bucket.blob(storage_path)
            if not blob.exists():
                return None
            tmp = Path(tempfile.NamedTemporaryFile(delete=False, suffix=Path(storage_path).suffix).name)
            blob.download_to_filename(str(tmp))
            return tmp
        except Exception as e:
            logger.warning(f"Firebase Storage download_to_temp({storage_path}) failed: {e}")
            return None
