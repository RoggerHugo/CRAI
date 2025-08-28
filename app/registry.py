from __future__ import annotations
import json, pickle, pathlib, re, os
from typing import Optional, Dict, Any, List
import numpy as np

from .settings import settings

TIMESTAMP_RE = re.compile(r"^\d{8}-\d{6}$")  # ej. 20250821-153947

def _has_artifacts(d: pathlib.Path) -> bool:
    return (d / "modelos.pkl").exists() and (d / "metadata.json").exists()

def _list_candidate_versions(base: pathlib.Path) -> List[pathlib.Path]:
    if not base.exists():
        return []
    return [p for p in base.iterdir() if p.is_dir() and _has_artifacts(p)]

def _resolve_latest_dir(base: pathlib.Path) -> pathlib.Path:
    """
    Estrategia:
      1) Si hay current.txt -> usar esa versión si existe.
      2) Si no, elegir por nombre tipo timestamp (max)
      3) Si tampoco, elegir por mtime más reciente
    """
    # 1) current.txt
    current = base / "current.txt"
    if current.exists():
        ver = current.read_text(encoding="utf-8").strip()
        d = base / ver
        if _has_artifacts(d):
            return d

    # 2) por nombre timestamp
    children = _list_candidate_versions(base)
    ts_named = [p for p in children if TIMESTAMP_RE.match(p.name)]
    if ts_named:
        return sorted(ts_named, key=lambda p: p.name, reverse=True)[0]

    # 3) por mtime
    if children:
        return sorted(children, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    raise FileNotFoundError(
        f"No se encontraron artefactos en '{base}'. Esperaba subcarpetas con modelos.pkl/metadata.json."
    )

def _resolve_dir(base: pathlib.Path, version: Optional[str]) -> pathlib.Path:
    """
    Prioridad:
      1) Si version explícita -> esa subcarpeta
      2) Si hay subcarpetas válidas -> la más reciente (o current.txt)
      3) Solo si NO hay subcarpetas -> modo legacy (artefactos en la raíz)
    """
    # 1) versión explícita
    if version and str(version).strip().lower() != "auto":
        d = base / str(version).strip()
        if _has_artifacts(d):
            return d
        raise FileNotFoundError(f"La versión '{version}' no existe o está incompleta en {base}")

    # 2) subcarpetas (current.txt / timestamp / mtime)
    children = _list_candidate_versions(base)
    if children:
        return _resolve_latest_dir(base)

    # 3) legacy en la raíz
    if (base / "modelos.pkl").exists() and (base / "metadata.json").exists():
        return base

    raise FileNotFoundError(
        f"No se encontraron artefactos en '{base}'. Esperaba subcarpetas con modelos.pkl/metadata.json."
    )

class Signature:
    def __init__(self, d: dict):
        self.order = d.get("features_order", [])
        self.defaults = d.get("defaults", {})
        self.threshold = float(d.get("threshold", 0.5))

    def vectorize(self, features: dict):
        x = []
        for k in self.order:
            v = features.get(k, self.defaults.get(k))
            if v is None:
                raise ValueError(f"Falta la feature '{k}' y no hay default")
            x.append(v)
        return np.array([x], dtype=float)

class Bundle:
    """
    Carga artefactos desde la última versión (o la solicitada).
    Lee:
      - settings.MODELS_BASE_PATH como base
      - settings.MODEL_VERSION para 'auto' o versión fija
    """
    def __init__(self, base_path: Optional[str] = None, version: Optional[str] = None):
        base = pathlib.Path(base_path or settings.MODELS_BASE_PATH)
        ver  = version if version is not None else settings.MODEL_VERSION

        self.base_dir = base
        self.model_dir = _resolve_dir(base, ver)

        with open(self.model_dir / "modelos.pkl", "rb") as f:
            bundle = pickle.load(f)

        self.models: Dict[str, Any] = bundle["modelos"]
        self.scaler = bundle.get("scaler")
        self.encoders = bundle.get("encoders", {})

        self.meta = json.loads((self.model_dir / "metadata.json").read_text(encoding="utf-8"))
        sig_path = self.model_dir / "signature.json"
        self.sig = Signature(json.loads(sig_path.read_text(encoding="utf-8"))) if sig_path.exists() else Signature({})
        self.version = str(self.meta.get("version") or self.model_dir.name)
        self._current_token = self._read_current_token()

    def _read_current_token(self) -> str:
        cur = self.base_dir / "current.txt"
        if cur.exists():
            try:
                st = cur.stat()
                return f"{cur.read_text(encoding='utf-8').strip()}::{int(st.st_mtime)}"
            except Exception:
                return "unknown"
        try:
            st = self.model_dir.stat()
            return f"{self.model_dir.name}::{int(st.st_mtime)}"
        except Exception:
            return "unknown"

    # ---------- Selección de modelos (DENTRO de la clase) ----------
    def candidates(self) -> list[str]:
        return list(self.models.keys())

    def default_model(self) -> str:
        meta_default = (self.meta or {}).get("default_model")
        if meta_default and meta_default in self.models:
            return meta_default
        for pref in ("XGBoost", "RandomForest", "SVM"):
            if pref in self.models:
                return pref
        if self.models:
            return next(iter(self.models.keys()))
        raise RuntimeError("No hay modelos en el bundle.")

    def _metric_val(self, model_name: str, metric: str = "auc"):
        metrics = (self.meta or {}).get("metrics", {})
        m = metrics.get(model_name)
        if m is None:
            return None
        val = m.get(metric) if isinstance(m, dict) else m
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def best_by_metric(self, metric: str = "auc") -> str:
        cands = self.candidates()
        if not cands:
            raise RuntimeError("No hay modelos para seleccionar.")
        scored = [(mn, self._metric_val(mn, metric)) for mn in cands]
        scored = [(mn, v) for mn, v in scored if v is not None]
        if scored:
            scored.sort(key=lambda t: t[1], reverse=True)
            return scored[0][0]
        return self.default_model()

    def best_by_auc(self) -> str:
        return self.best_by_metric("auc")

# ---- Singleton ----
_BUNDLE = None

def get_bundle() -> Bundle:
    global _BUNDLE
    if _BUNDLE is None:
        # NO pasar argumentos aquí: evitamos TypeError si la firma cambia
        _BUNDLE = Bundle()
    return _BUNDLE

def reload_bundle(force_version: str | None = None) -> Bundle:
    """
    Si quieres forzar versión, cambiamos settings.MODEL_VERSION de forma temporal.
    """
    global _BUNDLE
    if force_version is None:
        _BUNDLE = Bundle()
        return _BUNDLE

    prev = settings.MODEL_VERSION
    try:
        # fija versión, crea instancia nueva, y luego restaura el valor
        settings.MODEL_VERSION = force_version
        _BUNDLE = Bundle()
    finally:
        settings.MODEL_VERSION = prev
    return _BUNDLE
