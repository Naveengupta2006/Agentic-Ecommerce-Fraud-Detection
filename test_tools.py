from app.agent.tools import get_user_history, check_velocity, explain_score

print(get_user_history(1))
print(check_velocity(1, "2024-01-15 17:00:04"))
print(explain_score({
    "account_age_days": 5, "total_transactions_user": 1, "avg_amount_user": 300.0,
    "amount": 850.0, "channel": "web", "merchant_category": "travel",
    "promo_used": 0, "avs_match": 0, "cvv_result": 0, "three_ds_flag": 0,
    "transaction_time": "2024-01-06T03:15:00Z", "shipping_distance_km": 2500.0,
    "country": "FR", "bin_country": "US"
}))