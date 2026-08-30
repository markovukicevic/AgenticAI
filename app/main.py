"""
app/main.py

HTTP/API concerns only: routing, request validation, response shaping,
and translating errors into HTTP status codes. No OpenRouter/prompt/
LLM code lives here — that's app/research.py's job. This file just
calls into it:

    result = ask_research_question(request.question)

    HTTP layer (this file)
        |
        v
    research logic (app/research.py)
        |
        v
    LLM (OpenRouter)

Run from the project root with:
    uvicorn app.main:app --reload

Then visit:
    http://127.0.0.1:8000/         -> basic hello response
    http://127.0.0.1:8000/health   -> health check
    http://127.0.0.1:8000/docs     -> interactive API docs (auto-generated)
"""

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, field_validator
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from app.metrics import get_metrics
from app.research import ask_research_question

app = FastAPI(title="Research Agent API")


class ResearchRequest(BaseModel):
    """Shape of the incoming POST /research request body."""

    question: str

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        # str already guarantees the field is present and is a string.
        # This extra check rejects "" or "   ", which would otherwise
        # sail through and waste an LLM call for no question at all.
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be empty")
        return stripped


class ResearchResponse(BaseModel):
    """Shape of the outgoing POST /research response body."""

    question: str
    answer: str
    key_points: list[str]


@app.get("/")
async def root():
    """Simple root endpoint, just so hitting the base URL isn't a 404."""
    return {"status": "ok", "message": "Research Agent API is running"}


@app.get("/health")
async def health():
    """
    Health check endpoint.

    This doesn't do anything clever on purpose — it exists so that
    infrastructure (a load balancer, an orchestrator, a deploy platform,
    a monitoring probe) has a cheap, dependency-free way to ask "is this
    process alive and able to respond to HTTP?" without triggering an
    LLM call or touching any external service. If this endpoint is slow
    or failing, something is wrong with the app process itself, not
    with OpenRouter.
    """
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    """
    Validate the incoming question, hand it to the research service,
    and return the structured result. The route itself contains no
    LLM/prompt/OpenRouter logic — that all lives in research_service.py
    so the CLI (research.py) and this API share exactly the same code.
    """
    try:
        result = ask_research_question(request.question)
    except RuntimeError as e:
        # The service already raises a clean, user-facing message on
        # any LLM/network failure. We surface it as a 502 (this server
        # failed to get a good response from an upstream service)
        # rather than a 500 (which would imply a bug in our own code).
        raise HTTPException(status_code=502, detail=str(e))

    return result

@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )