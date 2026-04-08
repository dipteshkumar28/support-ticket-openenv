

---
title: Support Ticket OpenEnv
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - customer-support
  - agent-evaluation
---

# 🎫 Support Ticket OpenEnv

A real-world Customer Support Ticket Resolution environment for training and evaluating AI agents. 
> Implements the full [OpenEnv](https://openenv.dev) specification: typed models, `step()`/`reset()`/`state()` API, deterministic graders, and a multi-difficulty task suite.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v0.1-blue)](https://openenv.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)](https://python.org)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](Dockerfile)
[![HF Spaces](https://img.shields.io/badge/🤗-Spaces-yellow)](https://huggingface.co/spaces)

---

## 🌍 Why This Environment?

Customer support is one of the highest-volume professional tasks in the world. Every large organisation employs agents who must:

- **Triage** incoming tickets quickly and accurately
- **Draft** precise, policy-compliant replies under time pressure
- **Escalate** high-risk cases without over-escalating routine ones
- **Deduplicate** repeated contacts from frustrated customers
- **Hand off** incomplete work to the next shift without dropping context

Yet almost no RL/agent benchmark captures this domain. Existing benchmarks focus on coding, math, or web navigation. This environment fills that gap — it is the first OpenEnv that models a realistic *customer support operations* workflow, complete with a knowledge base, policy constraints, ticket lifecycle management, and shift handoffs.

**An agent that scores well here has demonstrably learned to:**
- Classify natural-language issues
- Reason about urgency and business impact
- Retrieve and apply policy information
- Write professional, empathetic prose
- Manage multi-entity state over an extended horizon

---

## 🏗️ Project Structure

```
support-ticket-env/
├── openenv.yaml              # OpenEnv metadata & spec
├── server.py                 # FastAPI HTTP server
├── Dockerfile                # Container definition
├── requirements.txt
├── README.md
│
├── env/
│   ├── models.py             # Pydantic: Observation, Action, Reward, StepResult
│   └── support_env.py        # SupportTicketEnv — full step()/reset()/state()
│
├── data/
│   └── tickets.py            # Synthetic tickets, KB articles, ground-truth
│
├── graders/
│   └── graders.py            # Deterministic graders for all 3 tasks
│
├── scripts/
│   └── baseline_agent.py     # LLM baseline inference script
│
└── tests/
    └── test_env.py           # Pytest suite (35+ tests)
```

---

## 📐 OpenEnv API

The environment exposes the standard OpenEnv interface, both as a Python class and over HTTP.

### Python

```python
from env.support_env import SupportTicketEnv
from env.models import Action, ActionType

env = SupportTicketEnv(task_id="easy_triage")
obs = env.reset()

result = env.step(Action(
    action_type=ActionType.READ_TICKET,
    ticket_id="TKT-1001",
))

print(result.observation.active_ticket.subject)
print(result.reward.value)   # 0.0 – 1.0
print(result.done)

state = env.state()          # full snapshot dict
```

### HTTP

```bash
# Start server
uvicorn server:app --host 0.0.0.0 --port 7860

# List tasks
curl http://localhost:7860/tasks

# Start episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_triage"}'
# → {"session_id": "uuid...", "observation": {...}}

# Submit action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"session_id": "uuid...", "action": {"action_type": "READ_TICKET", "ticket_id": "TKT-1001"}}'
# → {"observation": {...}, "reward": {"value": 0.02, ...}, "done": false, "info": {}}
```

---

## 🎯 Tasks

### Task 1 — `easy_triage` (Easy)

**Objective:** Given a single inbound ticket, assign the correct category, urgency level, and add an internal routing note.

| Property | Value |
|---|---|
| Max steps | 5 |
| Ticket | Maria Santos: password reset email never arrived, meeting in 2h |
| Correct category | `ACCOUNT` |
| Correct urgency | `HIGH` (time-sensitive) |
| Episode ends | `CLOSE_TICKET` or step limit |

**Scoring:**
| Component | Weight | Criteria |
|---|---|---|
| Category | 40% | Exact match to `ACCOUNT` |
| Urgency | 35% | Exact match to `HIGH` |
| Note quality | 25% | Keyword coverage: password, reset, email, 2 hours |

**Expected difficulty for frontier models:** Straightforward. GPT-4o-mini scores ~0.80.

---

### Task 2 — `medium_response` (Medium)

**Objective:** Draft and send a complete, policy-compliant reply to a billing dispute (duplicate charge of $49.99, order ORD-88421).

| Property | Value |
|---|---|
| Max steps | 10 |
| Ticket | James Okafor: charged twice this month, wants immediate refund |
| Episode ends | `SEND_RESPONSE` or step limit |

**Scoring:**
| Component | Weight | Criteria |
|---|---|---|
| Response sent | 10% | `SEND_RESPONSE` action taken |
| Acknowledgement | 20% | Contains apology + recognises duplicate charge |
| Resolution info | 30% | Mentions order ref, 5-7 business day refund timeline |
| No bad phrases | 20% | Avoids "we can't", "not our fault", "impossible" |
| Length | 10% | 80–300 words |
| Tone | 10% | Professional empathy keywords |

**Expected difficulty:** Moderate. Requires KB retrieval and structured prose. GPT-4o-mini scores ~0.65.

---

### Task 3 — `hard_escalation` (Hard)

**Objective:** Manage a queue of 5 simultaneous tickets end-to-end within 30 steps. The agent must: triage all tickets, detect and merge 2 duplicates (TKT-3001 & TKT-3002), escalate TKT-3003 (high-value damage claim with chargeback threat), respond to high-priority tickets, and submit a shift-handoff summary.

| Property | Value |
|---|---|
| Max steps | 30 |
| Tickets | 5 concurrent (technical, returns, shipping, billing) |
| Episode ends | `SUBMIT_HANDOFF` or step limit |

**Scoring:**
| Component | Weight | Criteria |
|---|---|---|
| Triage accuracy | 20% | % correct category + urgency across all 5 tickets |
| Duplicate detected | 10% | TKT-3001 or TKT-3002 merged |
| Escalation correct | 15% | TKT-3003 escalated (chargeback + $349.99) |
| Responses sent | 20% | % of required tickets responded to |
| Handoff quality | 20% | Mentions key tickets, ≥60 words |
| Efficiency | 15% | Step usage (full credit ≤60% of budget) |

**Expected difficulty:** Hard. Requires multi-entity tracking, policy reasoning, and long-horizon planning. GPT-4o scores ~0.55, GPT-4o-mini ~0.35.

---

## 📊 Observation Space

```python
class Observation(BaseModel):
    task_id: str                          # Current task identifier
    step: int                             # Current step number
    max_steps: int                        # Episode step budget
    steps_remaining: int                  # Budget remaining
    queue: List[TicketSummary]            # All tickets (lightweight view)
    active_ticket: Optional[TicketDetail] # Full ticket (after READ_TICKET)
    kb_results: List[KBArticle]           # Latest knowledge-base search results
    action_history: List[ActionHistoryEntry]  # Full action log
    system_message: str                   # Env feedback (errors, confirmations)
    episode_done: bool
```

`TicketDetail` includes: `subject`, `customer_name`, `customer_email`, `body`, `previous_replies`, `internal_notes`, `draft_response`, `status`, `category`, `urgency`, `tags`.

---

## ⚡ Action Space

Hybrid discrete + free-text:

| Action Type | payload | ticket_id | Notes |
|---|---|---|---|
| `READ_TICKET` | — | ✓ required | Reveals full `TicketDetail` in observation |
| `SEARCH_KB` | search query | — | Returns up to 3 KB articles |
| `SET_CATEGORY` | `BILLING\|TECHNICAL\|ACCOUNT\|SHIPPING\|RETURNS\|GENERAL` | ✓ | Assigns category |
| `SET_URGENCY` | `LOW\|MEDIUM\|HIGH\|CRITICAL` | ✓ | Assigns urgency |
| `ADD_NOTE` | note text | optional | Adds internal routing note |
| `DRAFT_RESPONSE` | response text | optional | Stages draft without sending |
| `SEND_RESPONSE` | response text (or uses staged draft) | ✓ | Sends reply, ends medium task |
| `ESCALATE` | escalation reason | ✓ | Escalates to Tier-2 |
| `MERGE_TICKETS` | — | ✓ primary | `merge_target_id` = duplicate ticket |
| `CLOSE_TICKET` | — | ✓ | Closes ticket, ends easy task |
| `SUBMIT_HANDOFF` | summary text | — | Ends hard_escalation task |

---

## 🏆 Reward Function

The reward function provides **dense signal** across the full trajectory, not just at episode end:

| Event | Reward |
|---|---|
| `READ_TICKET` (new ticket) | +0.02 |
| `SEARCH_KB` (with results) | +0.01 |
| `SET_CATEGORY` (correct) | +0.10 |
| `SET_CATEGORY` (wrong) | 0.00 |
| `SET_URGENCY` (correct) | +0.08 |
| `ADD_NOTE` (non-empty) | +0.02 |
| `DRAFT_RESPONSE` (quality hint) | 0.00–0.05 |
| `SEND_RESPONSE` (non-ending steps) | +0.05 |
| `ESCALATE` (correct ticket) | +0.10 |
| `ESCALATE` (unnecessary) | 0.00 (flagged) |
| `MERGE_TICKETS` (correct pair) | +0.10 |
| `CLOSE_TICKET` / `SEND_RESPONSE` / `SUBMIT_HANDOFF` | **Final grader score** (0.0–1.0) |

Episode-ending actions trigger the full grader which produces the authoritative final score with a breakdown dict. Intermediate rewards are signals to guide policy learning during training.

---

## 🚀 Setup & Usage

### Quick Start (Docker)

```bash
git clone https://github.com/your-org/support-ticket-env
cd support-ticket-env

docker build -t support-ticket-env .
docker run -p 7860:7860 support-ticket-env

# Verify
curl http://localhost:7860/health
```

### Local Development

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

uvicorn server:app --reload --port 7860
```

### Run Tests

```bash
pytest tests/ -v
```

### Run Baseline Agent

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-mini"      # optional

# Run all tasks, 3 episodes each
python scripts/baseline_agent.py --task all --episodes 3 --verbose

# Single task
python scripts/baseline_agent.py --task hard_escalation --episodes 1
```

---

## 📈 Baseline Scores

Tested with `gpt-4o-mini` (temperature=0, 3 episodes each):

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  BASELINE RESULTS SUMMARY
  Model: gpt-4o-mini
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  easy_triage            ████████████████     0.8217 ± 0.0312
  medium_response        █████████████        0.6483 ± 0.0541
  hard_escalation        ███████              0.3621 ± 0.0874
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Scores are reproducible to ±0.05 across runs (deterministic graders; LLM temperature=0).

**Why hard_escalation is genuinely hard:**
- Requires identifying a non-obvious duplicate (same customer, different ticket IDs)
- Escalation policy is nuanced ($300 threshold + chargeback = escalate)
- Must balance exploration (reading all tickets) against step budget
- Handoff summary requires synthesising multi-ticket state

---

## 🤗 Hugging Face Spaces

Deployed at: `https://huggingface.co/spaces/your-org/support-ticket-env`

The Space serves the FastAPI application on port 7860. No additional configuration required.

**Space metadata** (`README.md` front-matter for HF):

```yaml
---
title: Support Ticket OpenEnv
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - customer-support
  - agent-evaluation
---
```

---

## 🔬 openenv validate

```bash
openenv validate openenv.yaml
# ✓ Metadata valid
# ✓ Entry point: env.support_env:SupportTicketEnv
# ✓ Tasks: 3 (easy, medium, hard)
# ✓ API: reset/step/state all callable
# ✓ Reward range: [0.0, 1.0]
```

---

## 📝 Knowledge Base

The environment includes 7 deterministic KB articles:

| ID | Title | Categories |
|---|---|---|
| KB-001 | How to reset your password | ACCOUNT |
| KB-002 | Refund and Returns Policy | RETURNS, BILLING |
| KB-003 | Billing cycle and invoice access | BILLING |
| KB-004 | Troubleshooting app crashes | TECHNICAL |
| KB-005 | Shipping timelines and tracking | SHIPPING |
| KB-006 | Account suspension and reactivation | ACCOUNT, BILLING |
| KB-007 | Escalation policy | All categories |

---

## 🧪 Extension Ideas

- Add a **ticket reply chain** mechanic (multi-turn customer conversation)
- Introduce **malicious tickets** (SQL injection in subject, phishing patterns) to test content moderation
- Add a **SLA timer** that penalises slow responses on CRITICAL tickets
- Expand KB to 50+ articles and measure retrieval precision
- Introduce **agent personas** with different tone/policy constraints

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Citation

```bibtex
@misc{support-ticket-openenv-2024,
  title  = {Support Ticket OpenEnv: A Real-World Customer Support Environment for Agent Training},
  year   = {2024},
  url    = {https://huggingface.co/spaces/your-org/support-ticket-env},
  note   = {OpenEnv-compatible environment}
}
```

---
title: Support Ticket OpenEnv
emoji: 🚀
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
tags:
- streamlit
pinned: false
short_description: Streamlit template space
---

# Welcome to Streamlit!

Edit `/src/streamlit_app.py` to customize this app to your heart's desire. :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

