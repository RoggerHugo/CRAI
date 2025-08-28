# app/main.py (fragmentos)
from fastapi import FastAPI, HTTPException, Body, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.gzip import GZipMiddleware
from .settings import settings
from .security import require_api_key, require_metrics_key
from .service import predict as service_predict
from .registry import get_bundle, reload_bundle
from .schemas import PredictIn, PredictOut

app = FastAPI(
    title="ml-service",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# Host y CORS estrictos
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)

# Cabeceras seguras + límite de cuerpo
@app.middleware("http")
async def secure_headers_and_limits(request: Request, call_next):
    # Límite de tamaño (DoS)
    body = await request.body()
    if len(body) > settings.MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Payload too large")

    # Continuar
    response = await call_next(request)

    # No cachear respuestas sensibles
    response.headers["Cache-Control"] = "no-store"
    # Anti-MIME sniff
    response.headers["X-Content-Type-Options"] = "nosniff"
    # Clickjacking
    response.headers["X-Frame-Options"] = "DENY"
    # Referrer minimizado
    response.headers["Referrer-Policy"] = "no-referrer"
    # Content Security Policy (no aplica mucho a JSON, pero segura por defecto)
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'; base-uri 'none'"
    return response

# app/main.py (rutas)
@app.get("/v1/model", tags=["admin"], dependencies=[Depends(require_api_key)])
def model_info():
    b = get_bundle()
    return {
        "version": b.version,
        "model_dir": str(b.model_dir.resolve()),
        "base_dir": str(b.base_dir.resolve()),
        "candidates": b.candidates(),
        "available_metrics": b.meta.get("metrics", {})
    }

@app.post("/admin/reload", tags=["admin"], dependencies=[Depends(require_api_key)])
def admin_reload(body: dict = Body(default={})):
    ver = (body or {}).get("version", "auto")
    # Sanitiza 'version' para evitar path traversal: solo timestamps
    import re
    if ver not in ("auto", None) and not re.fullmatch(r"\d{8}-\d{6}", str(ver)):
        raise HTTPException(status_code=400, detail="Invalid version format")
    b = reload_bundle(None if str(ver).lower() == "auto" else str(ver))
    return {"status": "ok", "reloaded_version": b.version}

# Predicción protegida
@app.post("/v1/predict", response_model=PredictOut, response_model_exclude_none=True,
          dependencies=[Depends(require_api_key)])
def predict_route(body: PredictIn):
    try:
        out = service_predict(body)
        return out
    except Exception as e:
        import logging; logging.exception("Prediction failed")
        # Evita filtrar trazas internas en respuesta
        raise HTTPException(status_code=400, detail="Invalid request or model error")
