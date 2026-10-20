import json
from app.agent.agent import review_transaction

txn = {
    "account_age_days": 141, "total_transactions_user": 47, "avg_amount_user": 170.57,
    "amount": 420.0, "channel": "web", "merchant_category": "travel",
    "promo_used": 0, "avs_match": 1, "cvv_result": 0, "three_ds_flag": 0,
    "transaction_time": "2024-01-20T03:15:00Z", "shipping_distance_km": 900.0,
    "country": "FR", "bin_country": "US",
}

result = review_transaction(txn, user_id=1, force_agent=True)
print(json.dumps(result, indent=2))