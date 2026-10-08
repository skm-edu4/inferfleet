from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=32_000)
    model: Optional[str] = None
    max_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str = "stop"


class GenerateResponse(BaseModel):
    id: str
    object: str = "text.completion"
    model: str
    choices: list[Choice]
    usage: Usage
    cached: bool = False
    cache_kind: Optional[str] = None
    queue_time_ms: float = 0.0
    latency_ms: float = 0.0


class MeResponse(BaseModel):
    user_id: str
    key_id: str
    tier: str
    rate_limit_rpm: int
    monthly_quota_tokens: int
    tokens_used: int
    tokens_reserved: int
    remaining_tokens: int


class ApiKeyOut(BaseModel):
    id: str
    label: str
    tier: str
    key_prefix: str
    revoked: bool
    created_at: str


class CreateKeyRequest(BaseModel):
    label: str = Field(default="default", max_length=64)
    tier: Literal["free", "pro", "enterprise"] = "free"


class HistoryItem(BaseModel):
    id: str
    model: str
    prompt: str
    completion: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cached: bool
    status: str
    created_at: str


class HistoryPage(BaseModel):
    items: list[HistoryItem]
    next_cursor: Optional[str] = None


class ErrorBody(BaseModel):
    error: dict[str, Any]
