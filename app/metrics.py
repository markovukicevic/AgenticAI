from prometheus_client import Counter


# Prometheus counters for research metrics.
requests_total = Counter(
    "research_requests",
    "Total number of research requests",
)

requests_success = Counter(
    "research_requests_success",
    "Total number of successful research requests",
)

requests_failed = Counter(
    "research_requests_failed",
    "Total number of failed research requests",
)

retries_total = Counter(
    "research_retries",
    "Total number of LLM retries",
)

prompt_tokens_total = Counter(
    "research_prompt_tokens",
    "Total number of prompt tokens used",
)

completion_tokens_total = Counter(
    "research_completion_tokens",
    "Total number of completion tokens used",
)

_total_tokens_counter = Counter(
    "research_tokens",
    "Total number of tokens used",
)

request_duration_seconds_total = Counter(
    "research_request_duration_seconds",
    "Total research request duration in seconds",
)


def record_request_started() -> None:
    requests_total.inc()


def record_request_success() -> None:
    requests_success.inc()


def record_request_failed() -> None:
    requests_failed.inc()


def record_retry() -> None:
    retries_total.inc()


def record_usage(
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
) -> None:
    prompt_tokens_total.inc(prompt_tokens)
    completion_tokens_total.inc(completion_tokens)
    _total_tokens_counter.inc(total_tokens)


def record_request_duration(duration_seconds: float) -> None:
    request_duration_seconds_total.inc(duration_seconds)


def get_metrics() -> dict:
    return {
        "requests_total": requests_total._value.get(),
        "requests_success": requests_success._value.get(),
        "requests_failed": requests_failed._value.get(),
        "retries_total": retries_total._value.get(),
        "prompt_tokens_total": prompt_tokens_total._value.get(),
        "completion_tokens_total": completion_tokens_total._value.get(),
        "total_tokens": _total_tokens_counter._value.get(),
        "request_duration_seconds_total": request_duration_seconds_total._value.get(),
    }
