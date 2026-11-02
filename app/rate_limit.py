from __future__ import annotations

import os

from fastapi import Request
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address


ASK_LIMIT_DEFAULT = "5/minute;50/day"
VERIFY_LIMIT_DEFAULT = "10/minute;100/day"


def client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    # Trust only the first proxy-supplied address; it preserves the caller identity in hosted deployments.
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=client_ip, storage_uri="memory://")


def endpoint_limits(name: str, default: str) -> str:
    return os.getenv(name, "").strip() or default
