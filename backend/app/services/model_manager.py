import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TypeVar

from ..config import get_settings
from ..logging_setup import logger

settings = get_settings()

T = TypeVar("T")


class ModelLoadingError(Exception):
    """Raised when a model cannot be loaded (missing package, download failure, OOM, etc.)."""
    pass


class InferenceRuntimeError(Exception):
    """Raised when inference on a previously-loaded model fails unexpectedly."""
    pass


@dataclass
class LoadedModel:
    """Cache entry for a loaded model and its supporting components."""

    model_id: str
    model_object: Any
    processor_object: Any
    loaded_at: float = field(default_factory=time.time)
    load_duration_sec: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def age_sec(self) -> float:
        return time.time() - self.loaded_at


class ModelManager:
    """A thread-safe singleton model loader/cache.

    Responsibilities:
      - Load a model ONCE on first request; return the cached instance on subsequent calls.
      - Serialize concurrent load attempts via a per-model lock so we never trigger
        duplicate downloads/allocations.
      - Surface a single `ModelLoadingError` on failure and remember the failure for a
        cooldown window so repeated calls don't thrash.
      - Report `is_loaded()` status and basic metadata for health checks.

    The manager is intentionally **model-agnostic**: it doesn't know *how* to load any
    specific model. The caller supplies a `load_fn` closure. This keeps the manager
    reusable for future VQA adapters, caption models, grounding backbones, etc.
    """

    _instance: Optional["ModelManager"] = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> "ModelManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._cache: Dict[str, LoadedModel] = {}
        self._lock = threading.RLock()
        self._per_model_locks: Dict[str, threading.Lock] = {}
        self._failed_models: Dict[str, float] = {}
        self._failure_cooldown_sec = 120.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_loaded(self, model_id: str) -> bool:
        with self._lock:
            return model_id in self._cache

    def get(self, model_id: str) -> Optional[LoadedModel]:
        with self._lock:
            return self._cache.get(model_id)

    def loaded_model_ids(self):
        with self._lock:
            return list(self._cache.keys())

    def load(
        self,
        model_id: str,
        load_fn,
        force_reload: bool = False,
    ) -> LoadedModel:
        """Load `model_id` using `load_fn`, cache the result, or return the cached entry.

        `load_fn` must be a zero-argument callable that returns the tuple:
            `(model_object, processor_object, metadata_dict)`
        and raises `ModelLoadingError` (or any Exception) on failure.
        """
        now = time.time()

        with self._lock:
            mlock = self._per_model_locks.setdefault(model_id, threading.Lock())
            if not force_reload and model_id in self._cache:
                return self._cache[model_id]
            failure_time = self._failed_models.get(model_id)
            if failure_time and (now - failure_time) < self._failure_cooldown_sec:
                remaining = self._failure_cooldown_sec - (now - failure_time)
                raise ModelLoadingError(
                    f"Model {model_id} failed to load recently; cooldown remaining: "
                    f"{remaining:.0f}s. Set force_reload=True to retry immediately."
                )

        with mlock:
            with self._lock:
                if not force_reload and model_id in self._cache:
                    return self._cache[model_id]

            t0 = time.time()
            try:
                logger.info(f"[ModelManager] Loading model '{model_id}' (first use)...")
                model_obj, processor_obj, meta = load_fn()
                duration = time.time() - t0
                entry = LoadedModel(
                    model_id=model_id,
                    model_object=model_obj,
                    processor_object=processor_obj,
                    loaded_at=time.time(),
                    load_duration_sec=round(duration, 2),
                    metadata=meta or {},
                )
                with self._lock:
                    self._cache[model_id] = entry
                    self._failed_models.pop(model_id, None)
                logger.info(
                    f"[ModelManager] Model '{model_id}' loaded successfully "
                    f"in {duration:.1f}s."
                )
                return entry
            except ModelLoadingError:
                with self._lock:
                    self._failed_models[model_id] = time.time()
                raise
            except Exception as e:
                with self._lock:
                    self._failed_models[model_id] = time.time()
                raise ModelLoadingError(
                    f"Failed to load model '{model_id}': {e}"
                ) from e

    def unload(self, model_id: str) -> bool:
        """Evict a model from the cache. Returns True if an entry was removed."""
        with self._lock:
            if model_id in self._cache:
                entry = self._cache.pop(model_id)
                del entry
                self._failed_models.pop(model_id, None)
                logger.info(f"[ModelManager] Unloaded model '{model_id}'.")
                return True
            return False

    def reset(self) -> None:
        """Clear the entire cache (primarily for tests)."""
        with self._lock:
            self._cache.clear()
            self._failed_models.clear()
            self._per_model_locks.clear()


def get_model_manager() -> ModelManager:
    """Module-level accessor for the singleton ModelManager."""
    return ModelManager()
