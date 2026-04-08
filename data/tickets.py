"""
Static dataset: tickets and knowledge-base articles used by all three tasks.
All data is deterministic — seeded by ticket_id strings, not random — so
episodes are fully reproducible without fixing a random seed externally.
"""
from __future__ import annotations
from typing import Dict, List
from env.models import (
    TicketDetail, TicketCategory, UrgencyLevel,
    TicketStatus, KBArticle,
)

# ---------------------------------------------------------------------------
# Knowledge-Base Articles
# ---------------------------------------------------------------------------

KB_ARTICLES: List[KBArticle] = [
    KBArticle(
        article_id="KB-001",
        title="How to reset your password",
        content=(
            "To reset your password: (1) Go to login page. (2) Click 'Forgot Password'. "
            "(3) Enter your registered email. (4) Check your inbox for a reset link — "
            "valid for 24 hours. If you don't receive the email within 5 minutes, "
            "check your spam folder or contact support."
        ),
        related_categories=[TicketCategory.ACCOUNT],
    ),
    KBArticle(
        article_id="KB-002",
        title="Refund and Returns Policy",
        content=(
            "We accept returns within 30 days of delivery for unused, unopened items. "
            "Digital products are non-refundable once downloaded. To initiate a return: "
            "log in, go to Orders, select the item, click 'Return'. Refunds are processed "
            "within 5-7 business days to the original payment method."
        ),
        related_categories=[TicketCategory.RETURNS, TicketCategory.BILLING],
    ),
    KBArticle(
        article_id="KB-003",
        title="Billing cycle and invoice access",
        content=(
            "Billing occurs on the same day each month as your sign-up date. "
            "Invoices are emailed automatically and are also accessible under "
            "Account > Billing > Invoices. For discrepancies, contact billing@support.example.com "
            "within 60 days of the charge."
        ),
        related_categories=[TicketCategory.BILLING],
    ),
    KBArticle(
        article_id="KB-004",
        title="Troubleshooting app crashes",
        content=(
            "Common fixes: (1) Update to latest app version. (2) Clear app cache: "
            "Settings > Apps > [App Name] > Clear Cache. (3) Restart device. "
            "(4) Reinstall if issue persists. If crashes occur on a specific screen, "
            "note the steps to reproduce and contact technical support with your device model "
            "and OS version."
        ),
        related_categories=[TicketCategory.TECHNICAL],
    ),
    KBArticle(
        article_id="KB-005",
        title="Shipping timelines and tracking",
        content=(
            "Standard shipping: 5-7 business days. Express: 2-3 business days. "
            "Overnight: next business day (order by 2pm local time). "
            "Tracking numbers are emailed upon dispatch, typically within 24h of order. "
            "International orders may face customs delays beyond our control."
        ),
        related_categories=[TicketCategory.SHIPPING],
    ),
    KBArticle(
        article_id="KB-006",
        title="Account suspension and reactivation",
        content=(
            "Accounts may be suspended for: missed payments, policy violations, or "
            "suspicious activity. To reactivate: (1) Settle any outstanding balance. "
            "(2) Email accounts@support.example.com with subject 'Reactivation Request'. "
            "Include your account ID and a brief explanation. Review takes 1-2 business days."
        ),
        related_categories=[TicketCategory.ACCOUNT, TicketCategory.BILLING],
    ),
    KBArticle(
        article_id="KB-007",
        title="Escalation policy",
        content=(
            "Escalate to Tier-2 when: (1) Issue unresolved after 2 contacts. "
            "(2) Customer requests manager. (3) Potential legal/regulatory concern. "
            "(4) Data breach suspected. (5) Refund > $500. "
            "Always include ticket history and a brief reason when escalating."
        ),
        related_categories=[
            TicketCategory.BILLING, TicketCategory.TECHNICAL,
            TicketCategory.ACCOUNT, TicketCategory.GENERAL,
        ],
    ),
]

KB_INDEX: Dict[str, KBArticle] = {a.article_id: a for a in KB_ARTICLES}


def search_kb(query: str) -> List[KBArticle]:
    """Simple keyword search over KB articles. Deterministic."""
    q = query.lower()
    results = []
    for article in KB_ARTICLES:
        score = 0
        if any(word in article.title.lower() for word in q.split()):
            score += 2
        if any(word in article.content.lower() for word in q.split()):
            score += 1
        if score > 0:
            results.append((score, article))
    results.sort(key=lambda x: -x[0])
    return [a for _, a in results[:3]]


# ---------------------------------------------------------------------------
# Task 1 — Easy: Single Ticket Triage
# ---------------------------------------------------------------------------

TASK_EASY_TICKET = TicketDetail(
    ticket_id="TKT-1001",
    subject="Can't log in — forgot my password",
    customer_name="Maria Santos",
    customer_email="maria.santos@example.com",
    body=(
        "Hi, I've been trying to log in for the past hour but I can't remember my password. "
        "I tried the reset link but never received an email. I checked spam too. "
        "My account email is maria.santos@example.com. Please help urgently, "
        "I have an important meeting in 2 hours and need to access my documents."
    ),
    status=TicketStatus.OPEN,
    category=TicketCategory.UNKNOWN,
    urgency=UrgencyLevel.LOW,
    created_at="2024-03-15T08:42:00Z",
    tags=["password", "login"],
)

# Ground truth for grader
TASK_EASY_GROUND_TRUTH = {
    "correct_category": TicketCategory.ACCOUNT,
    "correct_urgency": UrgencyLevel.HIGH,   # time-sensitive meeting
    "required_note_keywords": ["password", "reset", "email", "2 hours"],
}


# ---------------------------------------------------------------------------
# Task 2 — Medium: Draft Full Response
# ---------------------------------------------------------------------------

TASK_MEDIUM_TICKET = TicketDetail(
    ticket_id="TKT-2001",
    subject="Charged twice for my subscription this month",
    customer_name="James Okafor",
    customer_email="j.okafor@example.com",
    body=(
        "Hello, I just checked my bank statement and I was charged $49.99 twice "
        "this month — on March 1st and again on March 15th. My billing date should "
        "be the 1st. This is a mistake on your end and I want an immediate refund "
        "for the duplicate charge. Order reference: ORD-88421. Very disappointed."
    ),
    status=TicketStatus.OPEN,
    category=TicketCategory.BILLING,
    urgency=UrgencyLevel.HIGH,
    created_at="2024-03-16T10:05:00Z",
    tags=["duplicate-charge", "billing", "refund"],
    previous_replies=[],
)

TASK_MEDIUM_GROUND_TRUTH = {
    "must_acknowledge": ["duplicate charge", "inconvenience", "apology"],
    "must_include_info": ["5-7 business days", "refund", "ORD-88421"],
    "must_not_include": ["we can't", "impossible", "not our fault"],
    "tone_keywords_positive": ["apologize", "sorry", "understand", "resolve"],
    "min_words": 80,
    "max_words": 300,
}


# ---------------------------------------------------------------------------
# Task 3 — Hard: Multi-Ticket Escalation Workflow
# ---------------------------------------------------------------------------

TASK_HARD_TICKETS: List[TicketDetail] = [
    TicketDetail(
        ticket_id="TKT-3001",
        subject="App crashes every time I open it",
        customer_name="Priya Nair",
        customer_email="priya.nair@example.com",
        body=(
            "Your app has been crashing immediately on launch for 3 days. "
            "I've already updated and reinstalled. Device: iPhone 14, iOS 17.3. "
            "This is unacceptable — I'm a paid subscriber."
        ),
        status=TicketStatus.OPEN,
        category=TicketCategory.UNKNOWN,
        urgency=UrgencyLevel.LOW,
        created_at="2024-03-17T07:00:00Z",
        tags=[],
    ),
    TicketDetail(
        ticket_id="TKT-3002",
        subject="Same crashing app problem",
        customer_name="Priya Nair",
        customer_email="priya.nair@example.com",
        body=(
            "I submitted a ticket yesterday (about the app crashing) but heard nothing. "
            "Still broken. Please help."
        ),
        status=TicketStatus.OPEN,
        category=TicketCategory.UNKNOWN,
        urgency=UrgencyLevel.LOW,
        created_at="2024-03-18T09:15:00Z",
        tags=[],
    ),
    TicketDetail(
        ticket_id="TKT-3003",
        subject="Refund request — item arrived damaged",
        customer_name="Carlos Mendez",
        customer_email="c.mendez@example.com",
        body=(
            "My order (ORD-55812) arrived with a cracked screen. I have photos. "
            "I want a full refund of $349.99 immediately. If I don't hear back today "
            "I'm disputing with my credit card company."
        ),
        status=TicketStatus.OPEN,
        category=TicketCategory.UNKNOWN,
        urgency=UrgencyLevel.LOW,
        created_at="2024-03-17T14:30:00Z",
        tags=[],
    ),
    TicketDetail(
        ticket_id="TKT-3004",
        subject="When will my order ship?",
        customer_name="Aiko Tanaka",
        customer_email="a.tanaka@example.com",
        body=(
            "I placed order ORD-61033 five days ago and it still shows 'processing'. "
            "Estimated delivery was today. What's going on?"
        ),
        status=TicketStatus.OPEN,
        category=TicketCategory.UNKNOWN,
        urgency=UrgencyLevel.LOW,
        created_at="2024-03-18T11:00:00Z",
        tags=[],
    ),
    TicketDetail(
        ticket_id="TKT-3005",
        subject="Need invoice for tax purposes — URGENT",
        customer_name="Robert Klein",
        customer_email="r.klein@example.com",
        body=(
            "I require an official invoice for all charges in 2023 for my tax filing. "
            "Deadline is tomorrow. Account ID: ACC-7743. Please send immediately."
        ),
        status=TicketStatus.OPEN,
        category=TicketCategory.UNKNOWN,
        urgency=UrgencyLevel.LOW,
        created_at="2024-03-18T13:45:00Z",
        tags=[],
    ),
]

TASK_HARD_GROUND_TRUTH = {
    # Correct triage for each ticket
    "correct_categories": {
        "TKT-3001": TicketCategory.TECHNICAL,
        "TKT-3002": TicketCategory.TECHNICAL,
        "TKT-3003": TicketCategory.RETURNS,
        "TKT-3004": TicketCategory.SHIPPING,
        "TKT-3005": TicketCategory.BILLING,
    },
    "correct_urgencies": {
        "TKT-3001": UrgencyLevel.HIGH,   # paid subscriber, 3 days
        "TKT-3002": UrgencyLevel.HIGH,   # duplicate, unresponsive
        "TKT-3003": UrgencyLevel.CRITICAL,  # >$300, chargeback threat
        "TKT-3004": UrgencyLevel.MEDIUM,
        "TKT-3005": UrgencyLevel.CRITICAL,  # tax deadline tomorrow
    },
    # TKT-3001 and TKT-3002 are duplicates (same customer, same issue)
    "duplicate_pair": ("TKT-3001", "TKT-3002"),
    # Tickets that must be escalated (refund > $300 or chargeback threat)
    "must_escalate": {"TKT-3003"},
    # Tickets that must be responded to (not just triaged)
    "must_respond": {"TKT-3003", "TKT-3004", "TKT-3005"},
    # Handoff summary must mention these ticket IDs
    "handoff_must_mention": {"TKT-3001", "TKT-3003", "TKT-3005"},
}
