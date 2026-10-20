from app.agent.agent import review_transaction

base = {
    "total_transactions_user": 47, "avg_amount_user": 170.57,
    "channel": "web", "merchant_category": "travel", "promo_used": 0,
    "transaction_time": "2024-01-20T03:15:00Z",
    "country": "FR", "bin_country": "US",
}

cases = {
    "A": {**base, "account_age_days": 30, "amount": 600.0, "avs_match": 0, "cvv_result": 0, "three_ds_flag": 0, "shipping_distance_km": 1500.0},
    "B": {**base, "account_age_days": 60, "amount": 500.0, "avs_match": 1, "cvv_result": 0, "three_ds_flag": 0, "shipping_distance_km": 1200.0},
    "C": {**base, "account_age_days": 141, "amount": 420.0, "avs_match": 0, "cvv_result": 0, "three_ds_flag": 0, "shipping_distance_km": 2000.0},
    "D": {**base, "account_age_days": 20, "amount": 300.0, "avs_match": 1, "cvv_result": 1, "three_ds_flag": 0, "shipping_distance_km": 1800.0},
}

for name, txn in cases.items():
    r = review_transaction(txn, user_id=1)
    print(name, "| score:", r["model_score"], "| agent used:", r["used_agent"],
          "| decision:", r["decision"], "| tools:", r["tool_calls"])
    print("   ", r["reasoning"])