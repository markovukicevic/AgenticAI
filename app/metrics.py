import threading
from dataclasses import dataclass


@dataclass
class ResearchMetrics:
    requests_total: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    retries_total: int = 0

    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    total_tokens: int = 0


_metrics = ResearchMetrics()
_lock = threading.Lock()


def record_request_started() -> None:
    with _lock:
        _metrics.requests_total += 1


def record_request_success() -> None:
    with _lock:
        _metrics.requests_success += 1


def record_request_failed() -> None:
    with _lock:
        _metrics.requests_failed += 1


def record_retry() -> None:
    with _lock:
        _metrics.retries_total += 1


def record_usage(
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
) -> None:
    with _lock:
        _metrics.prompt_tokens_total += prompt_tokens
        _metrics.completion_tokens_total += completion_tokens
        _metrics.total_tokens += total_tokens


def get_metrics() -> dict:
    with _lock:
        return {
            "requests_total": _metrics.requests_total,
            "requests_success": _metrics.requests_success,
            "requests_failed": _metrics.requests_failed,
            "retries_total": _metrics.retries_total,
            "prompt_tokens_total": _metrics.prompt_tokens_total,
            "completion_tokens_total": _metrics.completion_tokens_total,
            "total_tokens": _metrics.total_tokens,
        }