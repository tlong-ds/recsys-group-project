"""Security helpers for the serving API."""

from __future__ import annotations

import os
import secrets
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import Header, HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

DEFAULT_API_KEYS_ENV_VAR = "RECSYS_API_KEYS"
DEFAULT_PUBLIC_PATHS = {"/health"}


@dataclass(frozen=True)
class SecuritySettings:
    """Runtime security settings for API routes."""

    enabled: bool = False
    api_keys_env_var: str = DEFAULT_API_KEYS_ENV_VAR
    api_keys: tuple[str, ...] = ()
    public_paths: frozenset[str] = frozenset(DEFAULT_PUBLIC_PATHS)
    rate_limit_per_minute: int = 120
    max_body_bytes: int = 65_536
    docs_enabled: bool = True

    @classmethod
    def from_serving_config(cls, serving_config: dict[str, Any]) -> SecuritySettings:
        raw = serving_config.get("security", {})
        cfg = raw if isinstance(raw, dict) else {}
        enabled = _as_bool(cfg.get("enabled", False))
        api_keys_env_var = str(cfg.get("api_keys_env_var", DEFAULT_API_KEYS_ENV_VAR))
        api_keys = _load_api_keys(api_keys_env_var)

        if enabled and not api_keys:
            raise RuntimeError(
                "API security is enabled but "
                f"{api_keys_env_var} has no configured keys."
            )

        public_paths = cfg.get("public_paths", sorted(DEFAULT_PUBLIC_PATHS))
        if not isinstance(public_paths, list):
            public_paths = sorted(DEFAULT_PUBLIC_PATHS)

        docs_enabled = _as_bool(cfg.get("docs_enabled", not enabled))
        return cls(
            enabled=enabled,
            api_keys_env_var=api_keys_env_var,
            api_keys=api_keys,
            public_paths=frozenset(str(path) for path in public_paths),
            rate_limit_per_minute=max(0, int(cfg.get("rate_limit_per_minute", 120))),
            max_body_bytes=max(0, int(cfg.get("max_body_bytes", 65_536))),
            docs_enabled=docs_enabled,
        )


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose declared body size exceeds the configured limit."""

    def __init__(self, app: Any, max_body_bytes: int) -> None:
        super().__init__(app)
        self.max_body_bytes = max_body_bytes

    async def dispatch(
        self, request: Request, call_next: Callable[..., Any]
    ) -> Response:
        if self.max_body_bytes > 0:
            content_length = request.headers.get("content-length")
            exceeds_limit = (
                content_length is not None
                and _content_length(content_length) > self.max_body_bytes
            )
            if exceeds_limit:
                return Response(
                    "Request body too large",
                    status_code=413,
                )
        return await call_next(request)


class InMemoryRateLimiter:
    """Simple per-identity fixed-window limiter for app-level abuse protection."""

    def __init__(self, requests_per_minute: int) -> None:
        self.requests_per_minute = requests_per_minute
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def check(self, identity: str) -> None:
        if self.requests_per_minute <= 0:
            return

        now = time.monotonic()
        window_start = now - 60.0
        events = self._events[identity]
        while events and events[0] < window_start:
            events.popleft()
        if len(events) >= self.requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded.",
            )
        events.append(now)


def auth_dependency(settings: SecuritySettings) -> Callable[..., str | None]:
    """Return a FastAPI dependency that validates bearer API keys."""

    limiter = InMemoryRateLimiter(settings.rate_limit_per_minute)

    async def verify_api_key(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> str | None:
        if not settings.enabled or request.url.path in settings.public_paths:
            return None

        token = _bearer_token(authorization)
        if token is None or not _is_valid_api_key(token, settings.api_keys):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        limiter.check(token)
        return token

    return verify_api_key


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token.strip()


def _is_valid_api_key(candidate: str, api_keys: tuple[str, ...]) -> bool:
    return any(secrets.compare_digest(candidate, api_key) for api_key in api_keys)


def _load_api_keys(env_var: str) -> tuple[str, ...]:
    raw = os.getenv(env_var, "")
    return tuple(key.strip() for key in raw.split(",") if key.strip())


def _content_length(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
