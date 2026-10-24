import pandas as pd

res = pd.read_csv("data/eval_results_v3.csv")
HIGH_VALUE = 1000
REVIEW_COST = 5.0
BLOCK_LEGIT = 0.25
STEPUP_LEGIT = 0.05


def model_only(score, amount):
    d = "block" if score >= 0.3 else "approve"
    return "hold_for_review" if d == "block" and amount >= HIGH_VALUE else d


res["model_only"] = [model_only(s, a) for s, a in zip(res["model_score"], res["amount"])]
res["stepup_all"] = "step_up_verification"


def cost(decision, fraud, amount, pass_rate, review_catch):
    if decision == "approve":
        return amount if fraud else 0.0
    if decision == "block":
        return 0.0 if fraud else BLOCK_LEGIT * amount
    if decision == "step_up_verification":
        return pass_rate * amount if fraud else STEPUP_LEGIT * amount
    # hold_for_review: insaani reviewer har fraud nahi pakadta
    return REVIEW_COST + ((1 - review_catch) * amount if fraud else 0.0)


rows = []
for pass_rate in [0.05, 0.2, 0.4]:
    for review_catch in [1.0, 0.8, 0.5]:
        row = {"stepup_fraud_pass": pass_rate, "reviewer_catch": review_catch}
        for col, label in [("model_only", "model_only"), ("stepup_all", "stepup_all"), ("decision", "agent")]:
            row[label] = round(sum(cost(d, f, a, pass_rate, review_catch)
                                   for d, f, a in zip(res[col], res["is_fraud"], res["amount"])), 1)
        row["cheapest"] = min(["model_only", "stepup_all", "agent"], key=lambda k: row[k])
        rows.append(row)

print(pd.DataFrame(rows).to_string(index=False))
print()
print(res.loc[res["decision"] == "hold_for_review",
              ["transaction_id", "amount", "is_fraud", "model_score"]].to_string(index=False))