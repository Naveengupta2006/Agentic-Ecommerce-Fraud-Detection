import json
import os
import sqlite3
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app.predict import predict_fraud
from app.agent.agent import review_transaction, LOW, HIGH, HIGH_VALUE

# ---------------- settings ----------------
N_POOL = 3000          # test rows scanned to find gray-zone cases
N_CASES = 5            # pehle 5 se trial karo, phir 40 kar dena
SEED = 42
DB_PATH = "data/fraud.db"
RESULTS_CSV = "data/eval_results.csv"

# cost assumptions (apne hisaab se badal sakte ho)
REVIEW_COST = 5.0           # ek manual review ka kharcha
STEPUP_FRAUD_PASS = 0.2     # step-up ke baad bhi nikal jaane wale fraud ka hissa
STEPUP_LEGIT_COST = 0.05    # step-up se bhaag jaane wale asli customer ka nuksaan (amount ka hissa)
BLOCK_LEGIT_COST = 0.25     # galat block hone par asli customer ka nuksaan (amount ka hissa)
MODEL_ONLY_THRESHOLD = 0.3  # sirf-model baseline ka threshold

FIELDS = ["account_age_days", "total_transactions_user", "avg_amount_user", "amount",
          "channel", "merchant_category", "promo_used", "avs_match", "cvv_result",
          "three_ds_flag", "shipping_distance_km", "country", "bin_country"]


def py(v):
    return v.item() if hasattr(v, "item") else v


def to_txn(row):
    txn = {f: py(row[f]) for f in FIELDS}
    txn["transaction_time"] = str(row["transaction_time"]).replace(" ", "T") + "Z"
    return txn


def load_test_rows():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM transactions ORDER BY rowid", con)
    con.close()
    idx = np.arange(len(df))
    _, test_idx = train_test_split(idx, test_size=0.2, stratify=df["is_fraud"], random_state=42)
    return df.iloc[test_idx].copy()


def guardrail(decision, amount):
    return "hold_for_review" if decision == "block" and amount >= HIGH_VALUE else decision


def cost(decision, is_fraud, amount):
    if decision == "approve":
        return amount if is_fraud else 0.0
    if decision == "block":
        return 0.0 if is_fraud else BLOCK_LEGIT_COST * amount
    if decision == "step_up_verification":
        return STEPUP_FRAUD_PASS * amount if is_fraud else STEPUP_LEGIT_COST * amount
    return REVIEW_COST  # hold_for_review


def report(res):
    res["model_only"] = [
        guardrail("block" if s >= MODEL_ONLY_THRESHOLD else "approve", a)
        for s, a in zip(res["model_score"], res["amount"])
    ]
    res["stepup_all"] = "step_up_verification"

    print(f"\n=== RESULTS on {len(res)} gray-zone cases ({int(res['is_fraud'].sum())} actually fraud) ===")
    print(f"agent fallbacks (LLM failed, sent to human): {int((res['confidence'] == 0.0).sum())}\n")

    rows = {}
    for col, label in [("model_only", "Model only (block >= 0.3)"),
                       ("stepup_all", "Step-up everything"),
                       ("decision", "Agent")]:
        d, f, a = res[col], res["is_fraud"], res["amount"]
        total = sum(cost(x, y, z) for x, y, z in zip(d, f, a))
        rows[label] = {
            "missed_fraud": int(((d == "approve") & (f == 1)).sum()),
            "false_blocks": int(((d == "block") & (f == 0)).sum()),
            "step_ups": int((d == "step_up_verification").sum()),
            "reviews": int((d == "hold_for_review").sum()),
            "total_cost": round(total, 1),
        }
    print(pd.DataFrame(rows).T.to_string())

    print("\nAgent decision vs actual fraud (0 = legit, 1 = fraud):")
    print(pd.crosstab(res["decision"], res["is_fraud"]).to_string())


def main():
    test_df = load_test_rows()
    print(f"test rows: {len(test_df)} (expected 59939), fraud: {int(test_df['is_fraud'].sum())} (expected 1322)")

    pool = test_df.sample(n=N_POOL, random_state=SEED)
    gray = []
    print("scanning for gray-zone cases...")
    for i, (_, row) in enumerate(pool.iterrows(), 1):
        score = predict_fraud(to_txn(row))["fraud_probability"]
        if LOW <= score <= HIGH:
            gray.append(row)
            if len(gray) >= N_CASES:
                break
        if i % 500 == 0:
            print(f"  scanned {i}, gray so far {len(gray)}")

    print(f"gray-zone cases selected: {len(gray)}")
    case_ids = [int(r["transaction_id"]) for r in gray]

    done = set()
    if os.path.exists(RESULTS_CSV):
        done = set(pd.read_csv(RESULTS_CSV)["transaction_id"].astype(int))

    for n, row in enumerate(gray, 1):
        tid = int(row["transaction_id"])
        if tid in done:
            continue
        r = review_transaction(to_txn(row), user_id=int(row["user_id"]))
        rec = {
            "transaction_id": tid,
            "user_id": int(row["user_id"]),
            "amount": float(row["amount"]),
            "is_fraud": int(row["is_fraud"]),
            "model_score": r["model_score"],
            "decision": r["decision"],
            "confidence": r["confidence"],
            "tool_calls": json.dumps(r["tool_calls"]),
        }
        pd.DataFrame([rec]).to_csv(RESULTS_CSV, mode="a",
                                   header=not os.path.exists(RESULTS_CSV), index=False)
        print(f"[{n}/{len(gray)}] txn {tid} | score {r['model_score']} | "
              f"actual fraud={rec['is_fraud']} | agent: {r['decision']}")
        time.sleep(2)

    res = pd.read_csv(RESULTS_CSV)
    res = res[res["transaction_id"].isin(case_ids)].copy()
    report(res)


if __name__ == "__main__":
    main()