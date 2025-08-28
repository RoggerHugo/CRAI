# app/service.py
from __future__ import annotations
from typing import Any, Dict, Tuple
from collections import Counter
import numpy as np
import pandas as pd
from .settings import settings
from .registry import get_bundle

BUNDLE = get_bundle()

# ---------- Utilidades de entrada ----------
def _coerce_features(payload: Any) -> Dict[str, Any]:
    if hasattr(payload, "features") and isinstance(payload.features, dict):
        return payload.features
    if isinstance(payload, dict) and "features" in payload and isinstance(payload["features"], dict):
        return payload["features"]
    if isinstance(payload, dict):
        return payload
    raise ValueError("Entrada inválida. Envía JSON plano o {'features': {...}}.")

def _get_expected_cols(fallback_keys: list[str] | None = None) -> list[str]:
    if BUNDLE.scaler is not None and hasattr(BUNDLE.scaler, "feature_names_in_"):
        return list(BUNDLE.scaler.feature_names_in_)
    feats = getattr(BUNDLE, "meta", {}).get("features")
    if feats:
        return list(feats)
    try:
        if hasattr(BUNDLE, "sig") and BUNDLE.sig.order:
            return list(BUNDLE.sig.order)
    except Exception:
        pass
    if fallback_keys:
        return list(fallback_keys)
    raise ValueError("No encuentro columnas esperadas. Guarda 'features' en metadata.json o usa scaler con 'feature_names_in_'.")

def _align_dataframe(raw_features: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([raw_features])
    expected = _get_expected_cols(list(df.columns))
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan
    return df[expected]

def _apply_encoders_df(df: pd.DataFrame) -> pd.DataFrame | np.ndarray:
    enc = BUNDLE.encoders or {}
    if not enc:
        return df
    if isinstance(enc, dict) and "ct" in enc:   # ColumnTransformer
        return enc["ct"].transform(df)
    df2 = df.copy()
    for col, le in enc.items():
        if col in df2.columns:
            if hasattr(le, "classes_"):
                relleno = le.classes_[0]
                s = df2[col].astype(str).fillna(relleno).str.strip()
                known = set(map(str, le.classes_.tolist()))
                s = s.apply(lambda x: x if x in known else relleno)
                df2[col] = le.transform(s)
            else:
                df2[col] = le.transform(df2[col])
    return df2

def _to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.fillna(0)

# --- NUEVO: elegir representación compatible por estimador ---
def _ensure_compatible_X(est, X):
    """
    Si el estimador fue entrenado SIN nombres (sin feature_names_in_), pásale ndarray.
    Si fue entrenado CON nombres, pásale DataFrame con esas columnas.
    """
    import pandas as pd
    # Caso DataFrame -> quizá convertir a ndarray
    if isinstance(X, pd.DataFrame):
        if hasattr(est, "feature_names_in_"):
            # asegurar orden correcto si el estimador conoce columnas
            cols = list(est.feature_names_in_)
            # agrega columnas faltantes si hubiera (con 0); no debería ocurrir si tu pipeline alinea bien
            missing = [c for c in cols if c not in X.columns]
            if missing:
                X = X.copy()
                for c in missing:
                    X[c] = 0
            return X[cols]
        else:
            # el modelo no conoce nombres -> mejor ndarray para evitar warning
            return X.values
    # Caso ndarray -> quizá convertir a DataFrame
    else:
        if hasattr(est, "feature_names_in_"):
            cols = list(est.feature_names_in_)
            try:
                import pandas as pd
                if X.shape[1] == len(cols):
                    return pd.DataFrame(X, columns=cols)
            except Exception:
                pass
        return X


def _prepare_matrices(raw_features: Dict[str, Any]) -> Tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray]:
    """
    Construye:
      - X_rf_xgb: para RF/XGB (mantén DataFrame si es posible)
      - X_svm:    para SVM (tras escalar, reconstituye DataFrame con las MISMAS columnas)
    """
    df = _align_dataframe(raw_features)

    # Encoders -> puede regresar DF o ndarray (si usas ColumnTransformer con one-hot)
    X_rf_xgb = _apply_encoders_df(df)

    # Asegura numérico si seguimos en DF
    if isinstance(X_rf_xgb, pd.DataFrame):
        X_rf_xgb = _to_numeric_df(X_rf_xgb)

    # SVM: escalar y PRESERVAR nombres de columnas para evitar warnings
    if BUNDLE.scaler is not None:
        if isinstance(X_rf_xgb, pd.DataFrame):
            cols = list(X_rf_xgb.columns)
            X_svm_arr = BUNDLE.scaler.transform(X_rf_xgb)  # ndarray
            X_svm = pd.DataFrame(X_svm_arr, columns=cols)  # <- rearmar DF con los mismos nombres
        else:
            X_svm_arr = BUNDLE.scaler.transform(X_rf_xgb)  # ndarray de entrada/salida
            # intenta recuperar nombres desde el scaler o desde el SVM
            cols = None
            if hasattr(BUNDLE.scaler, "feature_names_in_"):
                cols = list(BUNDLE.scaler.feature_names_in_)
            elif "SVM" in BUNDLE.models and hasattr(BUNDLE.models["SVM"], "feature_names_in_"):
                fni = BUNDLE.models["SVM"].feature_names_in_
                cols = list(fni) if fni is not None else None
            if cols is not None and len(cols) == X_svm_arr.shape[1]:
                X_svm = pd.DataFrame(X_svm_arr, columns=cols)
            else:
                X_svm = X_svm_arr
    else:
        # Sin scaler: intenta mantener DF (con nombres) si es posible
        if isinstance(X_rf_xgb, pd.DataFrame):
            X_svm = X_rf_xgb.copy()
        else:
            cols = None
            if "SVM" in BUNDLE.models and hasattr(BUNDLE.models["SVM"], "feature_names_in_"):
                fni = BUNDLE.models["SVM"].feature_names_in_
                if fni is not None and len(fni) == X_rf_xgb.shape[1]:
                    cols = list(fni)
            X_svm = pd.DataFrame(X_rf_xgb, columns=cols) if cols else X_rf_xgb

    return X_rf_xgb, X_svm


# ---------- Predicción por modelo ----------
def _probability(estimator, X) -> float | None:
    if hasattr(estimator, "predict_proba"):
        p = estimator.predict_proba(X)
        return float(p[0, 1])
    if hasattr(estimator, "decision_function"):
        d = float(estimator.decision_function(X).ravel()[0])
        return float(1.0 / (1.0 + np.exp(-d)))
    return None

def predecir_con(nombre: str, modelos: Dict[str, Any], X_rf_xgb, X_svm) -> Tuple[np.ndarray, np.ndarray | None]:
    est = modelos.get(nombre)
    if est is None:
        raise ValueError(f"Modelo '{nombre}' no disponible.")
    # SVM usa X_svm (escalado); RF/XGB usan X_rf_xgb
    X_base = X_svm if nombre.upper().startswith("SVM") else X_rf_xgb
    X_in = _ensure_compatible_X(est, X_base)   # <-- adaptar aquí

    pred = est.predict(X_in).ravel()
    pr = _probability(est, X_in)
    pr_arr = np.array([pr], dtype=float) if pr is not None else None
    return pred, pr_arr

# ---------- Selección / umbral ----------
def _threshold() -> float:
    try:
        if hasattr(BUNDLE, "sig") and BUNDLE.sig.threshold is not None:
            return float(BUNDLE.sig.threshold)
    except Exception:
        pass
    return 0.5

def _select_model(override: str | None) -> str:
    if override and override in BUNDLE.candidates():
        return override
    strat = settings.SELECT_STRATEGY
    if strat == "default":
        return BUNDLE.default_model()
    if strat == "best_auc":
        return BUNDLE.best_by_auc()
    if strat == "soft_voting":
        return "__ensemble__"
    return BUNDLE.best_by_auc()

# ---------- API ----------
def predict(payload: Any):
    # 1) features + matrices
    features = _coerce_features(payload)
    requested = getattr(payload, "use_model", None) if hasattr(payload, "use_model") else None
    X_rf_xgb, X_svm = _prepare_matrices(features)

    # 2) correr TODOS para detalle/ensamble
    preds: Dict[str, int] = {}
    probas: Dict[str, float] = {}

    for nombre in BUNDLE.models.keys():
        p, pr = predecir_con(nombre, BUNDLE.models, X_rf_xgb, X_svm)
        preds[nombre] = int(p[0])
        if pr is not None:
            probas[nombre] = float(pr[0])

    # 3) decidir si ensamble
    do_ensemble = bool(requested) and settings.ENSEMBLE_ON_SELECTION
    if _select_model(requested) == "__ensemble__":
        do_ensemble = True

    thr = _threshold()
    used: str
    score: float | None
    final_label: int

    if do_ensemble:
        # votos y probas
        conteo = Counter(preds.values())                 # {0: n0, 1: n1}
        voto_label = conteo.most_common(1)[0][0] if preds else 0
        votos_total = sum(conteo.values()) or 1
        vote_conf = conteo[1] / votos_total              # proporción de 1s
        proba_prom = (sum(probas.values()) / len(probas)) if probas else None

        rule = (settings.ENSEMBLE_RULE or "hybrid").lower()
        if rule not in ("vote", "proba", "hybrid"):
            rule = "hybrid"

        if rule == "vote":
            final_label = voto_label
            score = proba_prom if proba_prom is not None else vote_conf

        elif rule == "proba":
            if proba_prom is not None:
                final_label = 1 if proba_prom >= thr else 0
                score = proba_prom
            else:
                # si no hay probabilidades, usar proporción de votos como proxy
                final_label = 1 if vote_conf >= thr else 0
                score = vote_conf

        else:  # hybrid = voto, y en empate usa proba_prom; si no hay proba -> conservador
            empate = (len(conteo) == 2) and (conteo[0] == conteo[1])
            if empate and proba_prom is not None:
                final_label = 1 if proba_prom >= thr else 0
                score = proba_prom
            elif empate and proba_prom is None:
                final_label = 0
                score = vote_conf
            else:
                final_label = voto_label
                score = proba_prom if proba_prom is not None else vote_conf

        used = f"Ensemble({len(preds)})"

    else:
        # selección clásica, pero ya tenemos detalle
        strategy = _select_model(requested)
        if strategy not in BUNDLE.models:
            strategy = BUNDLE.best_by_auc()
        used = strategy
        score = probas.get(strategy, float(preds.get(strategy, 0)))
        final_label = 1 if score >= thr else 0

    if not probas:
        probas = {}

    pred_label = "Riesgo de Abandono" if final_label == 1 else "Sin Riesgo de Abandono"

    return {
        "prediction_label": pred_label,
        "score": float(score) if score is not None else None,
        "model_used": used,
        "model_version": BUNDLE.version,
        "models": {
            "prediction": {k: int(v) for k, v in preds.items()},       # <-- cast a int nativo
            "probabilities": {k: float(v) for k, v in probas.items()}   # <-- cast a float nativo
        }
    }

