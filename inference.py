#!/usr/bin/env python3
"""
baseline_agent.py
-----------------
Baseline inference script for Support Ticket OpenEnv
Using HuggingFace instead of OpenAI
"""

from __future__ import annotations

import json
import os
import sys
import time
import statistics

import requests
from huggingface_hub import InferenceClient


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SERVER = os.getenv("ENV_SERVER_URL", "http://localhost:7860")

# Recommended Free Model
DEFAULT_MODEL = os.getenv(
    "HF_MODEL",
    "meta-llama/Meta-Llama-3-8B-Instruct"
)

SYSTEM_PROMPT = """You are an expert customer support agent operating inside a ticket management system.

You will receive a JSON observation describing the current state of a ticket queue.
You must respond with a JSON action to take next.

Action format:
{
  "action_type": "<ACTION_TYPE>",
  "ticket_id": "<optional ticket ID>",
  "payload": "<optional text or value>",
  "merge_target_id": "<optional, only for MERGE_TICKETS>"
}

Available action types:
- READ_TICKET
- SEARCH_KB
- SET_CATEGORY
- SET_URGENCY
- ADD_NOTE
- DRAFT_RESPONSE
- SEND_RESPONSE
- ESCALATE
- MERGE_TICKETS
- CLOSE_TICKET
- SUBMIT_HANDOFF

Guidelines:
- Always READ_TICKET before acting
- Use SEARCH_KB before drafting
- Merge duplicate tickets
- Escalate high financial risk
- Write empathetic responses

Respond with ONLY valid JSON.
Do not include:
- explanations
- markdown
- text before JSON
- text after JSON

Return JSON only.
"""


# ---------------------------------------------------------------------------
# HuggingFace LLM Call
# ---------------------------------------------------------------------------

from huggingface_hub import InferenceClient
import re

def hf_generate(model, api_key, messages):

    client = InferenceClient(
        model=model,
        api_key=api_key
    )

    response = client.chat.completions.create(
        messages=messages,
        temperature=0,
        max_tokens=200
    )

    text = response.choices[0].message.content

    # Extract JSON only
    match = re.search(r"\{[\s\S]*\}", text)

    if match:
        return match.group(0)

    # fallback
    return '{"action_type":"READ_TICKET"}'


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_episode(
    api_key: str,
    server: str,
    task_id: str,
    model: str,
    verbose: bool = False,
) -> dict:

    # Reset environment
    resp = requests.post(f"{server}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    session_id = data["session_id"]
    obs = data["observation"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Task: {task_id}\n\n"
                f"Initial observation:\n{json.dumps(obs, indent=2)}\n\n"
                "Take your first action."
            ),
        },
    ]

    final_score = 0.0
    steps = 0
    total_reward = 0.0

    for step_num in range(100):

        try:
            raw_action = hf_generate(
                model=model,
                api_key=api_key,
                messages=messages
            ).strip()

        except Exception as e:
            print(f"LLM error: {e}")
            break

        # Remove markdown
        if raw_action.startswith("```"):
            raw_action = raw_action.split("```")[1]

        raw_action = raw_action.strip()

        try:
            action_dict = json.loads(raw_action)
        except json.JSONDecodeError:

            messages.append({"role": "assistant", "content": raw_action})
            messages.append({
                "role": "user",
                "content": "Your response must be valid JSON"
            })
            continue

        # Send action to environment
        step_resp = requests.post(
            f"{server}/step",
            json={"action": action_dict, "session_id": session_id},
            timeout=30,
        )

        step_data = step_resp.json()

        reward_val = step_data["reward"]["value"]
        done = step_data["done"]
        new_obs = step_data["observation"]

        steps += 1
        total_reward += reward_val

        if done:
            final_score = step_data.get("info", {}).get(
                "final_score", reward_val
            )
            break

        messages.append({"role": "assistant", "content": raw_action})
        messages.append({
            "role": "user",
            "content": json.dumps(new_obs, indent=2)
        })

    return {
        "task_id": task_id,
        "session_id": session_id,
        "final_score": final_score,
        "steps_taken": steps,
        "total_reward": round(total_reward, 4),
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():

    api_key = os.getenv("HF_API_KEY")

    if not api_key:
        print("[START] task=init", flush=True)
        print("[END] task=init score=0 steps=0", flush=True)
        return

    server = DEFAULT_SERVER
    model = DEFAULT_MODEL

    # Health Check
    try:
        requests.get(f"{server}/health", timeout=10)
    except Exception:
        print("[START] task=health", flush=True)
        print("[END] task=health score=0 steps=0", flush=True)
        return

    tasks = [
        "easy_triage",
        "medium_response",
        "hard_escalation"
    ]

    for task_id in tasks:

        print(f"[START] task={task_id}", flush=True)

        scores = []

        for step in range(2):

            result = run_episode(
                api_key=api_key,
                server=server,
                task_id=task_id,
                model=model
            )

            score = result["final_score"]
            scores.append(score)

            print(
                f"[STEP] task={task_id} step={step} reward={score}",
                flush=True
            )

            time.sleep(0.5)

        mean_score = round(statistics.mean(scores), 4)

        print(
            f"[END] task={task_id} score={mean_score} steps={len(scores)}",
            flush=True
        )


if __name__ == "__main__":
    main()