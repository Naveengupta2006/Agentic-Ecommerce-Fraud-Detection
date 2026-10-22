import pandas as pd

res = pd.read_csv("data/eval_results_v2.csv")
HIGH_VALUE = 1000
REVIEW_COST = 5.0


def model_only(score, amount):
    d = "block" if score >= 0.3 else "approve"
    return "hold_for_review" if d == "block" and amount >= HIGH_VALUE else d


res["model_only"] = [model_only(s, a) for s, a in zip(res["model_score"], res["amount"])]
res["stepup_all"] = "step_up_verification"


def cost(decision, fraud, amount, pass_rate, block_legit):
    if decision == "approve":
        return amount if fraud else 0.0
    if decision == "block":
        return 0.0 if fraud else block_legit * amount
    if decision == "step_up_verification":
        return pass_rate * amount if fraud else 0.05 * amount
    return REVIEW_COST


rows = []
for pass_rate in [0.05, 0.1, 0.2, 0.4]:
    for block_legit in [0.1, 0.25, 0.5]:
        row = {"stepup_fraud_pass": pass_rate, "block_legit_cost": block_legit}
        for col, label in [("model_only", "model_only"), ("stepup_all", "stepup_all"), ("decision", "agent")]:
            row[label] = round(sum(cost(d, f, a, pass_rate, block_legit)
                                   for d, f, a in zip(res[col], res["is_fraud"], res["amount"])), 1)
        rows.append(row)

print(pd.DataFrame(rows).to_string(index=False))