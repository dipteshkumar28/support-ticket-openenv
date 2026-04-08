"""
Typed Pydantic models for the Support Ticket OpenEnv environment.
Defines Observation, Action, Reward, and all nested structures.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
 

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TicketCategory(str, Enum):
    BILLING = "BILLING"
    TECHNICAL = "TECHNICAL"
    ACCOUNT = "ACCOUNT"
    SHIPPING = "SHIPPING"
    RETURNS = "RETURNS"
    GENERAL = "GENERAL"
    UNKNOWN = "UNKNOWN"


class UrgencyLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TicketStatus(str, Enum):
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    AWAITING_CUSTOMER = "AWAITING_CUSTOMER"
    ESCALATED = "ESCALATED"
    CLOSED = "CLOSED"
    MERGED = "MERGED"


class ActionType(str, Enum):
    READ_TICKET = "READ_TICKET"        # Focus on a ticket; reveals full body
    SEARCH_KB = "SEARCH_KB"            # Query knowledge-base with a search string
    SET_CATEGORY = "SET_CATEGORY"      # Assign category to active ticket
    SET_URGENCY = "SET_URGENCY"        # Assign urgency to active ticket
    ADD_NOTE = "ADD_NOTE"              # Add internal routing/note text
    DRAFT_RESPONSE = "DRAFT_RESPONSE"  # Stage a customer-facing draft
    SEND_RESPONSE = "SEND_RESPONSE"    # Finalise and send staged draft
    ESCALATE = "ESCALATE"              # Escalate ticket; payload = reason
    MERGE_TICKETS = "MERGE_TICKETS"    # Merge ticket_id into active ticket
    CLOSE_TICKET = "CLOSE_TICKET"      # Mark ticket resolved
    SUBMIT_HANDOFF = "SUBMIT_HANDOFF"  # Submit shift-handoff summary (hard task)


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class TicketSummary(BaseModel):
    """Lightweight view shown in the queue."""
    ticket_id: str
    subject: str
    customer_name: str
    status: TicketStatus
    category: TicketCategory = TicketCategory.UNKNOWN
    urgency: UrgencyLevel = UrgencyLevel.LOW
    created_at: str  # ISO-8601 string kept simple for env use


class TicketDetail(BaseModel):
    """Full ticket revealed after READ_TICKET."""
    ticket_id: str
    subject: str
    customer_name: str
    customer_email: str
    body: str
    attachments: List[str] = Field(default_factory=list)
    previous_replies: List[Dict[str, str]] = Field(default_factory=list)
    status: TicketStatus
    category: TicketCategory
    urgency: UrgencyLevel
    internal_notes: List[str] = Field(default_factory=list)
    draft_response: Optional[str] = None
    created_at: str
    tags: List[str] = Field(default_factory=list)


class KBArticle(BaseModel):
    article_id: str
    title: str
    content: str
    related_categories: List[TicketCategory]


class ActionHistoryEntry(BaseModel):
    step: int
    action_type: ActionType
    payload: Optional[str]
    result_summary: str


# ---------------------------------------------------------------------------
# Core OpenEnv Models
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """What the agent sees at each step."""
    task_id: str
    step: int
    max_steps: int
    steps_remaining: int
    queue: List[TicketSummary]           # All tickets in the current episode
    active_ticket: Optional[TicketDetail] = None  # Set after READ_TICKET
    kb_results: List[KBArticle] = Field(default_factory=list)  # Latest KB search
    action_history: List[ActionHistoryEntry] = Field(default_factory=list)
    system_message: str = ""             # Environment feedback to the agent
    episode_done: bool = False

    class Config:
        use_enum_values = True


class Action(BaseModel):
    """What the agent submits each step."""
    action_type: ActionType
    ticket_id: Optional[str] = None     # Target ticket (defaults to active)
    payload: Optional[str] = None       # Free-text: draft, note, search query, etc.
    merge_target_id: Optional[str] = None  # For MERGE_TICKETS

    class Config:
        use_enum_values = True


class Reward(BaseModel):
    """Rich reward signal returned with every step."""
    value: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)
    reason: str = ""

    class Config:
        use_enum_values = True


class StepResult(BaseModel):
    """Full return value of step()."""
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
