import json
from app.agent.agent import review_transaction

txn = {
    "total_transactions_user": 47, "avg_amount_user": 170.57,
    "channel": "web", "merchant_category": "travel", "promo_used": 0,
    "transaction_time": "2024-01-20T03:15:00Z",
    "country": "FR", "bin_country": "US",
    "account_age_days": 60, "amount": 500.0, "avs_match": 1,
    "cvv_result": 0, "three_ds_flag": 0, "shipping_distance_km": 1200.0,
}

r = review_transaction(txn, user_id=1)
print(json.dumps(r, indent=2))