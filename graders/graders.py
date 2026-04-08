"""
Deterministic agent graders for all three tasks.

Each grader receives the final episode state and returns a score in [0.0, 1.0]
with a detailed breakdown dict for transparency.

Graders are stateless functions — same inputs always produce same outputs.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple
from env.models import (
    TicketCategory, TicketStatus, UrgencyLevel, ActionType, ActionHistoryEntry,
)
from data.tickets import (
    TASK_EASY_GROUND_TRUTH, TASK_MEDIUM_GROUND_TRUTH, TASK_HARD_GROUND_TRUTH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_contains_any(text: str, keywords: List[str]) -> List[str]:
    """Return which keywords (case-insensitive) appear in text."""
    t = text.lower()
    return [k for k in keywords if k.lower() in t]


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


# ---------------------------------------------------------------------------
# Task 1 — Easy Grader
# ---------------------------------------------------------------------------

def grade_easy(episode_state: dict) -> Tuple[float, Dict[str, float]]:
    """
    Score the triage task.
    Breakdown:
      - category_correct   : 0.40
      - urgency_correct    : 0.35
      - note_quality       : 0.25 (partial credit for each keyword hit)
    """
    gt = TASK_EASY_GROUND_TRUTH
    ticket = episode_state.get("ticket", {})
    history: List[ActionHistoryEntry] = episode_state.get("history", [])

    breakdown: Dict[str, float] = {}

    # --- Category (0.40) ---
    assigned_cat = ticket.get("category", TicketCategory.UNKNOWN)
    breakdown["category_correct"] = 0.40 if assigned_cat == gt["correct_category"] else 0.0

    # --- Urgency (0.35) ---
    assigned_urg = ticket.get("urgency", UrgencyLevel.LOW)
    urgency_scores = {
        UrgencyLevel.CRITICAL: 0.25,
        UrgencyLevel.HIGH: 0.35,    # correct
        UrgencyLevel.MEDIUM: 0.15,
        UrgencyLevel.LOW: 0.0,
    }
    breakdown["urgency_correct"] = urgency_scores.get(assigned_urg, 0.0)

    # --- Note quality (0.25) ---
    def _get_h(entry, key):
        return entry.get(key) if isinstance(entry, dict) else getattr(entry, key, None)
    notes_text = " ".join(
        _get_h(e, "payload") or "" for e in history if _get_h(e, "action_type") == ActionType.ADD_NOTE
    )
    hits = _text_contains_any(notes_text, gt["required_note_keywords"])
    breakdown["note_quality"] = _clamp(0.25 * len(hits) / len(gt["required_note_keywords"]))

    total = _clamp(sum(breakdown.values()))
    return total, breakdown


# ---------------------------------------------------------------------------
# Task 2 — Medium Grader
# ---------------------------------------------------------------------------

def grade_medium(episode_state: dict) -> Tuple[float, Dict[str, float]]:
    """
    Score the response-drafting task.
    Breakdown:
      - sent_response      : 0.10  (did the agent actually send?)
      - acknowledgement    : 0.20  (empathy/apology language)
      - resolution_info    : 0.30  (policy facts: timeline, ref number)
      - no_bad_phrases     : 0.20  (no hostile/incorrect statements)
      - length_appropriate : 0.10  (80-300 words)
      - tone_positive      : 0.10  (professional positive language)
    """
    gt = TASK_MEDIUM_GROUND_TRUTH
    history: List[ActionHistoryEntry] = episode_state.get("history", [])
    final_response: str = episode_state.get("sent_response", "") or ""

    breakdown: Dict[str, float] = {}

    # --- Sent response ---
    breakdown["sent_response"] = 0.10 if final_response.strip() else 0.0

    if not final_response.strip():
        # Nothing sent — remaining scores are 0
        for k in ("acknowledgement", "resolution_info", "no_bad_phrases",
                  "length_appropriate", "tone_positive"):
            breakdown[k] = 0.0
        return 0.10, breakdown

    text = final_response.lower()

    # --- Acknowledgement (0.20) ---
    ack_hits = _text_contains_any(final_response, gt["must_acknowledge"])
    breakdown["acknowledgement"] = _clamp(0.20 * len(ack_hits) / len(gt["must_acknowledge"]))

    # --- Resolution info (0.30) ---
    info_hits = _text_contains_any(final_response, gt["must_include_info"])
    breakdown["resolution_info"] = _clamp(0.30 * len(info_hits) / len(gt["must_include_info"]))

    # --- No bad phrases (0.20) ---
    bad_hits = _text_contains_any(final_response, gt["must_not_include"])
    breakdown["no_bad_phrases"] = 0.0 if bad_hits else 0.20

    # --- Length (0.10) ---
    wc = _word_count(final_response)
    if gt["min_words"] <= wc <= gt["max_words"]:
        breakdown["length_appropriate"] = 0.10
    elif wc < gt["min_words"]:
        breakdown["length_appropriate"] = _clamp(0.10 * wc / gt["min_words"])
    else:
        # Too long — slight penalty
        breakdown["length_appropriate"] = _clamp(0.10 * gt["max_words"] / wc)

    # --- Tone (0.10) ---
    tone_hits = _text_contains_any(final_response, gt["tone_keywords_positive"])
    breakdown["tone_positive"] = _clamp(0.10 * min(len(tone_hits), 3) / 3)

    total = _clamp(sum(breakdown.values()))
    return total, breakdown


# ---------------------------------------------------------------------------
# Task 3 — Hard Grader
# ---------------------------------------------------------------------------

def grade_hard(episode_state: dict) -> Tuple[float, Dict[str, float]]:
    """
    Score the multi-ticket escalation workflow.
    Breakdown (each sub-score normalised to its weight):
      - triage_accuracy     : 0.20  (% of tickets correctly categorised + urgency)
      - duplicate_detected  : 0.10  (TKT-3001 & TKT-3002 merged)
      - escalation_correct  : 0.15  (TKT-3003 escalated)
      - responses_sent      : 0.20  (% of must-respond tickets with a sent response)
      - handoff_quality     : 0.20  (mentions required ticket IDs, >50 words)
      - efficiency          : 0.15  (steps used vs budget — partial credit)
    """
    gt = TASK_HARD_GROUND_TRUTH
    tickets: Dict[str, dict] = episode_state.get("tickets", {})
    history: List[ActionHistoryEntry] = episode_state.get("history", [])
    handoff: str = episode_state.get("handoff_summary", "") or ""
    steps_used: int = episode_state.get("steps_used", 30)
    max_steps: int = episode_state.get("max_steps", 30)

    breakdown: Dict[str, float] = {}

    # --- Triage accuracy (0.20) ---
    cat_correct = sum(
        1 for tid, expected in gt["correct_categories"].items()
        if tickets.get(tid, {}).get("category") == expected
    )
    urg_correct = sum(
        1 for tid, expected in gt["correct_urgencies"].items()
        if tickets.get(tid, {}).get("urgency") == expected
    )
    n = len(gt["correct_categories"])
    triage_ratio = (cat_correct + urg_correct) / (2 * n)
    breakdown["triage_accuracy"] = _clamp(0.20 * triage_ratio)

    # --- Duplicate detected (0.10) ---
    dup_a, dup_b = gt["duplicate_pair"]
    merged_tickets = {
        tid for tid, t in tickets.items()
        if t.get("status") == TicketStatus.MERGED
    }
    # Accept either being merged into the other
    breakdown["duplicate_detected"] = 0.10 if (
        dup_a in merged_tickets or dup_b in merged_tickets
    ) else 0.0

    # --- Escalation (0.15) ---
    escalated = {
        tid for tid, t in tickets.items()
        if t.get("status") == TicketStatus.ESCALATED
    }
    required_esc = gt["must_escalate"]
    if required_esc.issubset(escalated):
        breakdown["escalation_correct"] = 0.15
    elif escalated & required_esc:
        breakdown["escalation_correct"] = 0.075
    else:
        breakdown["escalation_correct"] = 0.0

    # --- Responses sent (0.20) ---
    # Support both ActionHistoryEntry objects and plain dicts (e.g. in tests)
    def _get(entry, key):
        return entry.get(key) if isinstance(entry, dict) else getattr(entry, key, None)

    sent_responses = {
        _get(e, "ticket_id") for e in history
        if _get(e, "action_type") == ActionType.SEND_RESPONSE and _get(e, "ticket_id")
    }
    must_respond = gt["must_respond"]
    resp_ratio = len(sent_responses & must_respond) / len(must_respond)
    breakdown["responses_sent"] = _clamp(0.20 * resp_ratio)

    # --- Handoff quality (0.20) ---
    if handoff.strip():
        mention_count = sum(
            1 for tid in gt["handoff_must_mention"] if tid in handoff
        )
        mention_score = mention_count / len(gt["handoff_must_mention"])
        word_score = min(1.0, _word_count(handoff) / 60)
        breakdown["handoff_quality"] = _clamp(0.20 * (0.6 * mention_score + 0.4 * word_score))
    else:
        breakdown["handoff_quality"] = 0.0

    # --- Efficiency (0.15) ---
    # Full credit for using ≤ 60% of budget; scales down linearly to 0 at 100%
    usage_ratio = steps_used / max_steps
    if usage_ratio <= 0.60:
        eff = 1.0
    else:
        eff = max(0.0, 1.0 - (usage_ratio - 0.60) / 0.40)
    breakdown["efficiency"] = _clamp(0.15 * eff)

    total = _clamp(sum(breakdown.values()))
    return total, breakdown
