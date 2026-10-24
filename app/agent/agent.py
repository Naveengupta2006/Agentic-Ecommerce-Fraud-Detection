import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

from app.predict import predict_fraud
from app.agent.tools import get_user_history, check_velocity, explain_score

load_dotenv()

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ["GEMINI_API_KEY"],
    timeout=60,
    max_retries=0,
)
MODEL = os.environ["GEMINI_MODEL"]

LOW, HIGH = 0.15, 0.85          # iske bahar LLM ki zaroorat nahi
HIGH_VALUE = 1000               # isse upar agent akela block nahi karega
MAX_STEPS = 5
APPROVE_CAP = 0.5   # is score ya usse upar agent approve nahi kar sakta
BLOCK_MIN = 0.5
TOOL_NAMES = {"get_user_history", "check_velocity", "explain_score"}
ALLOWED = {"approve", "step_up_verification", "hold_for_review", "block"}

SYSTEM_PROMPT = """You are a fraud investigation agent for an e-commerce platform.
A machine-learning model has scored a transaction and the score is in the uncertain range, so you must investigate.

You can call these tools. You do not pass any arguments, they run on the current transaction:
- get_user_history: past transaction count, average amount, past confirmed frauds, account age for this user
- check_velocity: how many transactions this user made in the last 72 hours
- explain_score: the top reasons the ML model gave this score

Reply with ONLY one JSON object per turn, nothing else.
To call a tool: {"action": "get_user_history"}
To finish: {"action": "final", "decision": "...", "confidence": 0.0 to 1.0, "reasoning": "2-3 sentences", "evidence": ["short fact", "short fact"]}

Call explain_score first, then the other two tools, one per turn, then give your final answer.

DECISION RULES (these are the only rules; do not mention any other rule or policy):
1. If the ML score is below 0.5, choose approve or step_up_verification. Never block.
2. If the ML score is 0.5 or higher, never choose approve. Choose block, step_up_verification or hold_for_review.
3. Choose hold_for_review only when the amount is 1000 or more, or when you truly cannot decide.
4. When signals are mixed, lean towards step_up_verification. Failed AVS, CVV or 3D Secure, a country mismatch between country and bin_country, a past confirmed fraud and a long shipping distance raise the risk. A clean history, passed checks and an amount close to the customer's average lower it.
5. A past confirmed fraud alone never justifies a block.

Base every statement on tool results or the transaction fields. Never invent facts."""


def run_tool(name, txn, user_id):
    t = datetime.fromisoformat(txn["transaction_time"].replace("Z", "+00:00"))
    as_of = t.strftime("%Y-%m-%d %H:%M:%S")
    if name == "get_user_history":
        return get_user_history(user_id, as_of)
    if name == "check_velocity":
        return check_velocity(user_id, as_of)
    if name == "explain_score":
        return explain_score(txn)


def parse_json(text):
    decoder = json.JSONDecoder()
    found = None
    i = 0
    while True:
        i = text.find("{", i)
        if i == -1:
            break
        try:
            obj, end = decoder.raw_decode(text[i:])
            if isinstance(obj, dict) and "action" in obj:
                found = obj
            i += end
        except json.JSONDecodeError:
            i += 1
    return found


def _result(decision, confidence, score, reasoning, evidence, tool_calls, used_agent):
    return {
        "decision": decision,
        "confidence": confidence,
        "model_score": score,
        "reasoning": reasoning,
        "evidence": evidence,
        "tool_calls": tool_calls,
        "used_agent": used_agent,
    }

MIN_GAP = 6.0       # do LLM calls ke beech kam se kam itne second
_last_call = 0.0


def call_llm(messages, retries=4):
    global _last_call
    for attempt in range(retries):
        wait = MIN_GAP - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=messages, temperature=0, max_tokens=2048
            )
            if resp is not None and resp.choices:
                return resp.choices[0].message
            print("LLM returned no choices")
        except Exception as e:
            print("LLM error:", type(e).__name__, str(e)[-350:])
            if "429" in str(e):
                time.sleep(30 * (attempt + 1))   # per-minute limit 60 second mein reset hoti hai
                continue
        time.sleep(3 * (attempt + 1))
    return None

def review_transaction(txn, user_id, force_agent=False):
    score = predict_fraud(txn)["fraud_probability"]

    if not force_agent:
        if score < LOW:
            return _result("approve", 0.95, score, "Model score is very low.", [], [], False)
        if score > HIGH:
            decision = "hold_for_review" if txn["amount"] >= HIGH_VALUE else "block"
            return _result(decision, 0.95, score, "Model score is very high.", [], [], False)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Transaction: {json.dumps(txn)}\nML model fraud probability: {score}\nInvestigate and decide."},
    ]
    tool_calls = []

    for _ in range(MAX_STEPS):
        msg = call_llm(messages)
        if msg is None:
            return _result("hold_for_review", 0.0, score,
                           "Agent unavailable (LLM did not respond), sent to human review.",
                           [], tool_calls, True)

        text = msg.content or getattr(msg, "reasoning_content", None) or ""
        data = parse_json(text)
        if not data:
            break

        action = data.get("action")

        if action == "final":
            decision = data.get("decision")
            if decision not in ALLOWED:
                break
            reasoning = data.get("reasoning", "")
            if decision == "block" and txn["amount"] >= HIGH_VALUE:
                decision = "hold_for_review"
                reasoning += " (Guardrail: high-value block escalated to a human.)"
            if decision == "approve" and score >= APPROVE_CAP:
                decision = "step_up_verification"
                reasoning += " (Guardrail: model score is high, so approve is not allowed; escalated to step-up.)"
            if decision == "block" and score < BLOCK_MIN:
                decision = "step_up_verification"
                reasoning += " (Guardrail: block needs a model score of 0.5 or higher, so this became step-up.)"
            try:
                confidence = float(data.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            return _result(decision, confidence, score, reasoning,
                           data.get("evidence", []), tool_calls, True)

        if action in TOOL_NAMES:
            result = run_tool(action, txn, user_id)
            tool_calls.append(action)
            messages.append({"role": "assistant", "content": json.dumps(data)})
            messages.append({"role": "user", "content": f"Tool result for {action}: {json.dumps(result)}"})
        else:
            break

    return _result("hold_for_review", 0.0, score,
                   "Agent could not reach a clear decision, sent to human review.",
                   [], tool_calls, True)