"""
FastAPI server — exposes the SupportTicketEnv over HTTP.

Endpoints:
  POST /reset            — start a new episode
  POST /step             — submit an action
  GET  /state            — inspect current state
  GET  /tasks            — list available tasks
  GET  /health           — liveness check
  GET  /docs             — auto-generated Swagger UI (FastAPI default)

The server maintains one environment instance per task_id session.
For concurrent use, supply a session_id header; sessions are stored in memory.
"""
from __future__ import annotations

import os
import uuid
from typing import Dict, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.support_env import SupportTicketEnv, VALID_TASK_IDS, MAX_STEPS
from env.models import Action, StepResult, Observation

app = FastAPI(
    title="Support Ticket OpenEnv",
    description=(
        "A real-world customer support ticket resolution environment for AI agent training. "
        "Implements the full OpenEnv step()/reset()/state() API over HTTP."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: session_id → SupportTicketEnv
_sessions: Dict[str, SupportTicketEnv] = {}


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "easy_triage"
    session_id: Optional[str] = None


class ResetResponse(BaseModel):
    session_id: str
    task_id: str
    observation: Observation


class StepRequest(BaseModel):
    action: Action
    session_id: str


class TaskInfo(BaseModel):
    task_id: str
    name: str
    difficulty: str
    max_steps: int
    description: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_env(session_id: str) -> SupportTicketEnv:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call /reset first.")
    return _sessions[session_id]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "environment": "support-ticket-env", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            TaskInfo(
                task_id="easy_triage",
                name="Ticket Triage & Categorisation",
                difficulty="easy",
                max_steps=MAX_STEPS["easy_triage"],
                description="Assign correct category, urgency, and routing note to a single inbound ticket.",
            ),
            TaskInfo(
                task_id="medium_response",
                name="Draft Customer Response",
                difficulty="medium",
                max_steps=MAX_STEPS["medium_response"],
                description="Draft and send a policy-compliant, empathetic reply to a billing dispute.",
            ),
            TaskInfo(
                task_id="hard_escalation",
                name="Multi-Ticket Escalation Workflow",
                difficulty="hard",
                max_steps=MAX_STEPS["hard_escalation"],
                description="Manage 5 concurrent tickets: triage, respond, escalate, merge duplicates, submit handoff.",
            ),
        ]
    }


from fastapi import Body
from typing import Optional

@app.post("/reset", response_model=ResetResponse)
def reset(body: Optional[ResetRequest] = Body(default=None)):

    if body is None:
        task_id = "easy_triage"
        session_id = str(uuid.uuid4())
    else:
        task_id = body.task_id
        session_id = body.session_id or str(uuid.uuid4())

    if task_id not in VALID_TASK_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"task_id must be one of {list(VALID_TASK_IDS)}"
        )

    env = SupportTicketEnv(task_id=task_id)
    obs = env.reset()

    _sessions[session_id] = env

    return ResetResponse(
        session_id=session_id,
        task_id=task_id,
        observation=obs
    )


@app.post("/step", response_model=StepResult)
def step(body: StepRequest):
    env = _get_env(body.session_id)
    result = env.step(body.action)
    return result


@app.get("/state")
def get_state(session_id: str):
    env = _get_env(session_id)
    return env.state()


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    if session_id in _sessions:
        del _sessions[session_id]
        return {"deleted": session_id}
    raise HTTPException(status_code=404, detail="Session not found.")
