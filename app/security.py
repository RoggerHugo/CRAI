import hmac
from fastapi import Header, HTTPException, status, Depends
from .settings import settings

def _eq(a: str|bytes, b: str|bytes) -> bool:
    if a is None or b is None:
        return False
    if isinstance(a, str): a = a.encode()
    if isinstance(b, str): b = b.encode()
    return hmac.compare_digest(a, b)

async def require_api_key(x_api_key: str = Header(default=None)):
    if not _eq(x_api_key, settings.API_KEY):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

async def require_metrics_key(x_api_key: str = Header(default=None)):
    if not _eq(x_api_key, settings.METRICS_KEY):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
