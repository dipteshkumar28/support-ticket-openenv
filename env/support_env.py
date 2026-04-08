"""
SupportTicketEnv — OpenEnv-compliant environment for Customer Support Ticket Resolution.

Implements:
  reset()  → Observation
  step()   → StepResult (Observation, Reward, done, info)
  state()  → dict (full internal state snapshot)

Three tasks:
  easy_triage     — Categorise and prioritise a single ticket
  medium_response — Draft and send a policy-compliant customer reply
  hard_escalation — Manage a 5-ticket queue end-to-end
"""
from __future__ import annotations

import copy
import re
from typing import Any, Dict, Optional

from env.models import (
    Action, ActionHistoryEntry, ActionType,
     Observation, Reward, StepResult,
    TicketCategory, TicketDetail, TicketStatus, TicketSummary, UrgencyLevel,
)
from data.tickets import (
     search_kb,
    TASK_EASY_TICKET,
    TASK_MEDIUM_TICKET,
    TASK_HARD_TICKETS,
    TASK_EASY_GROUND_TRUTH, TASK_MEDIUM_GROUND_TRUTH, TASK_HARD_GROUND_TRUTH,
)
from graders.graders import grade_easy, grade_medium, grade_hard

VALID_TASK_IDS = {"easy_triage", "medium_response", "hard_escalation"}

MAX_STEPS: Dict[str, int] = {
    "easy_triage": 5,
    "medium_response": 10,
    "hard_escalation": 30,
}


class SupportTicketEnv:
    """
    OpenEnv-compliant Customer Support Ticket Resolution environment.
    """

    def __init__(self, task_id: str = "easy_triage"):
        if task_id not in VALID_TASK_IDS:
            raise ValueError(f"task_id must be one of {VALID_TASK_IDS}")
        self.task_id = task_id
        self._state: Dict[str, Any] = {}
        self._step_count = 0
        self._done = False
        self.reset()

    # -----------------------------------------------------------------------
    # Public OpenEnv API
    # -----------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset the environment to a fresh episode and return initial observation."""
        self._step_count = 0
        self._done = False

        if self.task_id == "easy_triage":
            ticket = copy.deepcopy(TASK_EASY_TICKET)
            self._state = {
                "task_id": self.task_id,
                "tickets": {ticket.ticket_id: ticket},
                "active_ticket_id": None,
                "kb_results": [],
                "history": [],
                "sent_response": None,
                "handoff_summary": None,
            }

        elif self.task_id == "medium_response":
            ticket = copy.deepcopy(TASK_MEDIUM_TICKET)
            self._state = {
                "task_id": self.task_id,
                "tickets": {ticket.ticket_id: ticket},
                "active_ticket_id": None,
                "kb_results": [],
                "history": [],
                "sent_response": None,
                "handoff_summary": None,
            }

        elif self.task_id == "hard_escalation":
            tickets_list = [copy.deepcopy(t) for t in TASK_HARD_TICKETS]
            self._state = {
                "task_id": self.task_id,
                "tickets": {t.ticket_id: t for t in tickets_list},
                "active_ticket_id": None,
                "kb_results": [],
                "history": [],
                "sent_response": None,   # not used in hard task
                "handoff_summary": None,
            }

        return self._build_observation(system_message="Episode started. Begin working on tickets.")

    def step(self, action: Action) -> StepResult:
        """Process one agent action and return (observation, reward, done, info)."""
        if self._done:
            obs = self._build_observation(system_message="Episode already finished. Call reset().")
            return StepResult(
                observation=obs,
                reward=Reward(value=0.0, reason="Episode already done."),
                done=True,
                info={"warning": "step() called after episode end"},
            )

        self._step_count += 1
        obs, reward, done, info = self._dispatch(action)
        self._done = done

        # Hard step-limit
        max_s = MAX_STEPS[self.task_id]
        if self._step_count >= max_s and not done:
            done = True
            self._done = True
            obs.episode_done = True
            obs.system_message += " [Step limit reached — episode ended.]"
            # Run final grader
            final_score, breakdown = self._run_grader()
            reward = Reward(
                value=final_score,
                breakdown=breakdown,
                reason="Step limit reached. Final graded score.",
            )
            info["final_score"] = final_score
            info["breakdown"] = breakdown

        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def state(self) -> Dict[str, Any]:
        """Return a deep-copy snapshot of full internal state."""
        snap = copy.deepcopy(self._state)
        snap["step_count"] = self._step_count
        snap["done"] = self._done
        snap["task_id"] = self.task_id
        return snap

    # -----------------------------------------------------------------------
    # Action dispatch
    # -----------------------------------------------------------------------

    def _dispatch(self, action: Action):
        """Route action to the appropriate handler."""
        atype = action.action_type

        handlers = {
            ActionType.READ_TICKET: self._handle_read_ticket,
            ActionType.SEARCH_KB: self._handle_search_kb,
            ActionType.SET_CATEGORY: self._handle_set_category,
            ActionType.SET_URGENCY: self._handle_set_urgency,
            ActionType.ADD_NOTE: self._handle_add_note,
            ActionType.DRAFT_RESPONSE: self._handle_draft_response,
            ActionType.SEND_RESPONSE: self._handle_send_response,
            ActionType.ESCALATE: self._handle_escalate,
            ActionType.MERGE_TICKETS: self._handle_merge_tickets,
            ActionType.CLOSE_TICKET: self._handle_close_ticket,
            ActionType.SUBMIT_HANDOFF: self._handle_submit_handoff,
        }

        handler = handlers.get(atype)
        if handler is None:
            obs = self._build_observation(system_message=f"Unknown action type: {atype}")
            return obs, Reward(value=0.0, reason="Unknown action"), False, {}

        return handler(action)

    # -----------------------------------------------------------------------
    # Individual action handlers
    # -----------------------------------------------------------------------

    def _handle_read_ticket(self, action: Action):
        tid = action.ticket_id or self._first_open_ticket_id()
        ticket = self._get_ticket(tid)
        if ticket is None:
            msg = f"Ticket {tid} not found."
            self._log_history(action.action_type, action.payload, msg, action.ticket_id)
            obs = self._build_observation(system_message=msg)
            return obs, Reward(value=0.0, reason=msg), False, {}

        self._state["active_ticket_id"] = tid
        ticket.status = TicketStatus.IN_PROGRESS
        msg = f"Opened ticket {tid}: '{ticket.subject}'"
        self._log_history(action.action_type, None, msg, tid)

        obs = self._build_observation(system_message=msg)
        # Small reward for engaging with a ticket
        reward = Reward(value=0.02, breakdown={"read_ticket": 0.02}, reason=msg)
        return obs, reward, False, {}

    def _handle_search_kb(self, action: Action):
        query = action.payload or ""
        if not query.strip():
            msg = "SEARCH_KB requires a query in the payload field."
            self._log_history(action.action_type, query, msg)
            obs = self._build_observation(system_message=msg)
            return obs, Reward(value=0.0, reason=msg), False, {}

        results = search_kb(query)
        self._state["kb_results"] = results
        msg = f"KB search for '{query}' returned {len(results)} article(s)."
        self._log_history(action.action_type, query, msg)
        obs = self._build_observation(system_message=msg)
        # Small reward for information-gathering
        reward = Reward(value=0.01 if results else 0.0,
                        breakdown={"kb_search": 0.01 if results else 0.0},
                        reason=msg)
        return obs, reward, False, {}

    def _handle_set_category(self, action: Action):
        tid = action.ticket_id or self._state.get("active_ticket_id")
        ticket = self._get_ticket(tid)
        if ticket is None:
            msg = "No active ticket. Use READ_TICKET first."
            self._log_history(action.action_type, action.payload, msg)
            obs = self._build_observation(system_message=msg)
            return obs, Reward(value=0.0, reason=msg), False, {}

        try:
            cat = TicketCategory(action.payload.upper() if action.payload else "UNKNOWN")
        except ValueError:
            valid = [c.value for c in TicketCategory]
            msg = f"Invalid category '{action.payload}'. Valid: {valid}"
            self._log_history(action.action_type, action.payload, msg, tid)
            obs = self._build_observation(system_message=msg)
            return obs, Reward(value=0.0, reason=msg), False, {}

        ticket.category = cat
        msg = f"Set category of {tid} to {cat.value}."
        self._log_history(action.action_type, cat.value, msg, tid)
        obs = self._build_observation(system_message=msg)

        # Partial reward: correct category gets a signal
        correct = self._check_category_correct(tid, cat)
        r_val = 0.10 if correct else 0.0
        reward = Reward(value=r_val, breakdown={"category_signal": r_val}, reason=msg)
        return obs, reward, False, {}

    def _handle_set_urgency(self, action: Action):
        tid = action.ticket_id or self._state.get("active_ticket_id")
        ticket = self._get_ticket(tid)
        if ticket is None:
            msg = "No active ticket. Use READ_TICKET first."
            self._log_history(action.action_type, action.payload, msg)
            obs = self._build_observation(system_message=msg)
            return obs, Reward(value=0.0, reason=msg), False, {}

        try:
            urg = UrgencyLevel(action.payload.upper() if action.payload else "LOW")
        except ValueError:
            valid = [u.value for u in UrgencyLevel]
            msg = f"Invalid urgency '{action.payload}'. Valid: {valid}"
            self._log_history(action.action_type, action.payload, msg, tid)
            obs = self._build_observation(system_message=msg)
            return obs, Reward(value=0.0, reason=msg), False, {}

        ticket.urgency = urg
        msg = f"Set urgency of {tid} to {urg.value}."
        self._log_history(action.action_type, urg.value, msg, tid)
        obs = self._build_observation(system_message=msg)

        correct = self._check_urgency_correct(tid, urg)
        r_val = 0.08 if correct else 0.0
        reward = Reward(value=r_val, breakdown={"urgency_signal": r_val}, reason=msg)
        return obs, reward, False, {}

    def _handle_add_note(self, action: Action):
        tid = action.ticket_id or self._state.get("active_ticket_id")
        ticket = self._get_ticket(tid)
        note = action.payload or ""
        if not note.strip():
            msg = "ADD_NOTE requires text in the payload field."
            self._log_history(action.action_type, note, msg, tid)
            obs = self._build_observation(system_message=msg)
            return obs, Reward(value=0.0, reason=msg), False, {}

        if ticket:
            ticket.internal_notes.append(note)
        msg = f"Note added to {tid}."
        self._log_history(action.action_type, note, msg, tid)
        obs = self._build_observation(system_message=msg)
        reward = Reward(value=0.02, breakdown={"note_added": 0.02}, reason=msg)
        return obs, reward, False, {}

    def _handle_draft_response(self, action: Action):
        tid = action.ticket_id or self._state.get("active_ticket_id")
        ticket = self._get_ticket(tid)
        draft = action.payload or ""
        if not draft.strip():
            msg = "DRAFT_RESPONSE requires draft text in payload."
            self._log_history(action.action_type, draft, msg, tid)
            obs = self._build_observation(system_message=msg)
            return obs, Reward(value=0.0, reason=msg), False, {}

        if ticket:
            ticket.draft_response = draft
        msg = f"Draft saved for {tid} ({len(draft.split())} words)."
        self._log_history(action.action_type, draft[:100] + "…" if len(draft) > 100 else draft, msg, tid)
        obs = self._build_observation(system_message=msg)
        # Reward proportional to draft quality preview
        wc = len(re.findall(r"\w+", draft))
        quality_hint = min(0.05, 0.05 * wc / 80)  # max 0.05 at ≥80 words
        reward = Reward(value=quality_hint, breakdown={"draft_quality_hint": quality_hint}, reason=msg)
        return obs, reward, False, {}

    def _handle_send_response(self, action: Action):
        tid = action.ticket_id or self._state.get("active_ticket_id")
        ticket = self._get_ticket(tid)
        if ticket is None:
            msg = "No active ticket to send response for."
            self._log_history(action.action_type, None, msg, tid)
            obs = self._build_observation(system_message=msg)
            return obs, Reward(value=0.0, reason=msg), False, {}

        # Use provided payload or existing draft
        response_text = action.payload or ticket.draft_response or ""
        if not response_text.strip():
            msg = "No response text. Use DRAFT_RESPONSE first or supply payload."
            self._log_history(action.action_type, None, msg, tid)
            obs = self._build_observation(system_message=msg)
            return obs, Reward(value=0.0, reason=msg), False, {}

        ticket.draft_response = response_text
        ticket.status = TicketStatus.AWAITING_CUSTOMER
        self._state["sent_response"] = response_text  # easy/medium task reference

        msg = f"Response sent to customer for {tid}."
        self._log_history(action.action_type, response_text[:100] + "…" if len(response_text) > 100 else response_text, msg, tid)

        # For easy_triage, sending response is not the primary goal, give small reward
        # For medium_response, run grader and close episode
        if self.task_id == "medium_response":
            ep_state = self._build_grader_state()
            score, breakdown = grade_medium(ep_state)
            reward = Reward(value=score, breakdown=breakdown, reason="Response sent — episode graded.")
            obs = self._build_observation(system_message=f"Response sent. Final score: {score:.3f}")
            obs.episode_done = True
            return obs, reward, True, {"final_score": score, "breakdown": breakdown}

        obs = self._build_observation(system_message=msg)
        reward = Reward(value=0.05, breakdown={"response_sent": 0.05}, reason=msg)
        return obs, reward, False, {}

    def _handle_escalate(self, action: Action):
        tid = action.ticket_id or self._state.get("active_ticket_id")
        ticket = self._get_ticket(tid)
        if ticket is None:
            msg = f"Cannot escalate: ticket {tid} not found."
            self._log_history(action.action_type, action.payload, msg, tid)
            obs = self._build_observation(system_message=msg)
            return obs, Reward(value=0.0, reason=msg), False, {}

        reason = action.payload or "No reason provided."
        ticket.status = TicketStatus.ESCALATED
        ticket.internal_notes.append(f"[ESCALATED] {reason}")
        msg = f"Ticket {tid} escalated to Tier-2."
        self._log_history(action.action_type, reason, msg, tid)
        obs = self._build_observation(system_message=msg)

        # Reward correct escalations, penalise wrong ones
        should_escalate = self._should_escalate(tid)
        r_val = 0.10 if should_escalate else -0.05  # negative for false escalation
        r_val = max(0.0, r_val)  # clamp to 0 for reward model
        reward = Reward(value=r_val, breakdown={"escalation": r_val},
                        reason=msg + (" [correct]" if should_escalate else " [unnecessary — no penalty in value but marked]"))
        return obs, reward, False, {}

    def _handle_merge_tickets(self, action: Action):
        primary_id = action.ticket_id or self._state.get("active_ticket_id")
        secondary_id = action.merge_target_id or action.payload
        primary = self._get_ticket(primary_id)
        secondary = self._get_ticket(secondary_id)

        if primary is None or secondary is None:
            msg = f"Merge failed: need valid ticket_id and merge_target_id."
            self._log_history(action.action_type, action.payload, msg, primary_id)
            obs = self._build_observation(system_message=msg)
            return obs, Reward(value=0.0, reason=msg), False, {}

        secondary.status = TicketStatus.MERGED
        primary.internal_notes.append(f"Merged with {secondary_id}")
        msg = f"Ticket {secondary_id} merged into {primary_id}."
        self._log_history(action.action_type, secondary_id, msg, primary_id)
        obs = self._build_observation(system_message=msg)

        # Reward correct deduplication
        gt_dup = TASK_HARD_GROUND_TRUTH.get("duplicate_pair", ())
        correct_merge = (primary_id in gt_dup and secondary_id in gt_dup)
        r_val = 0.10 if correct_merge else 0.0
        reward = Reward(value=r_val, breakdown={"merge": r_val}, reason=msg)
        return obs, reward, False, {}

    def _handle_close_ticket(self, action: Action):
        tid = action.ticket_id or self._state.get("active_ticket_id")
        ticket = self._get_ticket(tid)
        if ticket is None:
            msg = f"Ticket {tid} not found."
            self._log_history(action.action_type, None, msg, tid)
            obs = self._build_observation(system_message=msg)
            return obs, Reward(value=0.0, reason=msg), False, {}

        ticket.status = TicketStatus.CLOSED
        msg = f"Ticket {tid} closed."
        self._log_history(action.action_type, None, msg, tid)

        # For easy_triage: closing after correct triage ends episode
        if self.task_id == "easy_triage":
            ep_state = self._build_grader_state()
            score, breakdown = grade_easy(ep_state)
            reward = Reward(value=score, breakdown=breakdown, reason=f"Ticket closed — final score: {score:.3f}")
            obs = self._build_observation(system_message=f"Ticket closed. Final score: {score:.3f}")
            obs.episode_done = True
            return obs, reward, True, {"final_score": score, "breakdown": breakdown}

        obs = self._build_observation(system_message=msg)
        reward = Reward(value=0.03, breakdown={"closed": 0.03}, reason=msg)
        return obs, reward, False, {}

    def _handle_submit_handoff(self, action: Action):
        summary = action.payload or ""
        if not summary.strip():
            msg = "SUBMIT_HANDOFF requires a summary in the payload."
            self._log_history(action.action_type, summary, msg)
            obs = self._build_observation(system_message=msg)
            return obs, Reward(value=0.0, reason=msg), False, {}

        self._state["handoff_summary"] = summary
        msg = "Handoff summary submitted — episode ending."
        self._log_history(action.action_type, summary[:100], msg)

        ep_state = self._build_grader_state()
        score, breakdown = grade_hard(ep_state)
        reward = Reward(value=score, breakdown=breakdown, reason=f"Handoff submitted. Final score: {score:.3f}")
        obs = self._build_observation(system_message=f"Handoff received. Final score: {score:.3f}")
        obs.episode_done = True
        return obs, reward, True, {"final_score": score, "breakdown": breakdown}

    # -----------------------------------------------------------------------
    # Observation builder
    # -----------------------------------------------------------------------

    def _build_observation(self, system_message: str = "") -> Observation:
        tickets = self._state["tickets"]
        active_id = self._state.get("active_ticket_id")
        max_s = MAX_STEPS[self.task_id]

        queue = [
            TicketSummary(
                ticket_id=t.ticket_id,
                subject=t.subject,
                customer_name=t.customer_name,
                status=t.status,
                category=t.category,
                urgency=t.urgency,
                created_at=t.created_at,
            )
            for t in tickets.values()
        ]

        return Observation(
            task_id=self.task_id,
            step=self._step_count,
            max_steps=max_s,
            steps_remaining=max(0, max_s - self._step_count),
            queue=queue,
            active_ticket=tickets.get(active_id) if active_id else None,
            kb_results=list(self._state.get("kb_results", [])),
            action_history=list(self._state.get("history", [])),
            system_message=system_message,
            episode_done=self._done,
        )

    # -----------------------------------------------------------------------
    # Grader state builder
    # -----------------------------------------------------------------------

    def _build_grader_state(self) -> dict:
        tickets = self._state["tickets"]
        active_id = self._state.get("active_ticket_id")

        if self.task_id == "easy_triage":
            active = tickets.get(active_id or list(tickets.keys())[0])
            return {
                "ticket": {
                    "category": active.category if active else TicketCategory.UNKNOWN,
                    "urgency": active.urgency if active else UrgencyLevel.LOW,
                    "internal_notes": active.internal_notes if active else [],
                },
                "history": self._state.get("history", []),
            }

        elif self.task_id == "medium_response":
            return {
                "sent_response": self._state.get("sent_response", ""),
                "history": self._state.get("history", []),
            }

        elif self.task_id == "hard_escalation":
            return {
                "tickets": {
                    tid: {
                        "category": t.category,
                        "urgency": t.urgency,
                        "status": t.status,
                        "internal_notes": t.internal_notes,
                    }
                    for tid, t in tickets.items()
                },
                "history": self._state.get("history", []),
                "handoff_summary": self._state.get("handoff_summary", ""),
                "steps_used": self._step_count,
                "max_steps": MAX_STEPS[self.task_id],
            }
        return {}

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _get_ticket(self, ticket_id: Optional[str]) -> Optional[TicketDetail]:
        if not ticket_id:
            return None
        return self._state["tickets"].get(ticket_id)

    def _first_open_ticket_id(self) -> Optional[str]:
        for tid, t in self._state["tickets"].items():
            if t.status == TicketStatus.OPEN:
                return tid
        return list(self._state["tickets"].keys())[0] if self._state["tickets"] else None

    def _log_history(self, action_type, payload, result_summary, ticket_id=None):
        entry = ActionHistoryEntry(
            step=self._step_count,
            action_type=action_type,
            payload=str(payload)[:200] if payload else None,
            result_summary=result_summary,
        )
        # Attach ticket_id to the entry for hard-task grading
        entry_dict = entry.dict()
        entry_dict["ticket_id"] = ticket_id
        # Re-create with dynamic field (extend model for hard task)
        patched = ActionHistoryEntry(**{k: v for k, v in entry_dict.items() if k != "ticket_id"})
        patched.__dict__["ticket_id"] = ticket_id  # attach dynamically
        self._state["history"].append(patched)

    def _check_category_correct(self, tid: str, cat: TicketCategory) -> bool:
        if self.task_id == "easy_triage":
            return cat == TASK_EASY_GROUND_TRUTH["correct_category"]
        elif self.task_id == "hard_escalation":
            return cat == TASK_HARD_GROUND_TRUTH["correct_categories"].get(tid)
        return False

    def _check_urgency_correct(self, tid: str, urg: UrgencyLevel) -> bool:
        if self.task_id == "easy_triage":
            return urg == TASK_EASY_GROUND_TRUTH["correct_urgency"]
        elif self.task_id == "hard_escalation":
            return urg == TASK_HARD_GROUND_TRUTH["correct_urgencies"].get(tid)
        return False

    def _should_escalate(self, tid: str) -> bool:
        if self.task_id == "hard_escalation":
            return tid in TASK_HARD_GROUND_TRUTH["must_escalate"]
        return False

    def _run_grader(self):
        ep_state = self._build_grader_state()
        if self.task_id == "easy_triage":
            return grade_easy(ep_state)
        elif self.task_id == "medium_response":
            return grade_medium(ep_state)
        elif self.task_id == "hard_escalation":
            return grade_hard(ep_state)
        return 0.0, {}
