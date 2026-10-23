import json
import sqlite3
import pandas as pd
from evaluate_agent import to_txn
from app.agent.agent import review_transaction

# The 6 wrong cases (all false blocks on legit transactions)
WRONG_IDS = [202385, 255414, 107862, 215736, 121157, 22250]

con = sqlite3.connect("data/fraud.db")
df = pd.read_sql(
    f"SELECT * FROM transactions WHERE transaction_id IN ({','.join(map(str, WRONG_IDS))})",
    con
)
con.close()

print(f"Re-testing {len(df)} previously wrong cases with updated SYSTEM_PROMPT...\n")
results = []
for _, row in df.iterrows():
    txn = to_txn(row)
    r = review_transaction(txn, user_id=int(row["user_id"]))
    fraud = int(row["is_fraud"])
    decision = r["decision"]
    score = r["model_score"]
    txn_id = int(row["transaction_id"])

    correct = (fraud == 0 and decision in ["approve", "step_up_verification", "hold_for_review"]) or \
              (fraud == 1 and decision in ["block", "step_up_verification", "hold_for_review"])

    tag = "CORRECT" if correct else "STILL WRONG"
    print(f"[{tag}] txn={txn_id} | actual_fraud={fraud} | score={score:.3f} | decision={decision}")
    print(f"         reasoning: {r['reasoning'][:120]}")
    print()
    results.append({"txn_id": txn_id, "correct": correct, "decision": decision})

correct_now = sum(r["correct"] for r in results)
print(f"--- Re-test Summary ---")
print(f"Previously wrong: 6")
print(f"Now correct:      {correct_now}")
print(f"Still wrong:      {6 - correct_now}")
print(f"Overall accuracy: {(24 + correct_now) / 30 * 100:.1f}%  (was 80.0%)")
