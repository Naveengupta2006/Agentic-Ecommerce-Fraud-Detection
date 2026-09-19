from pydantic import BaseModel

class TransactionInput(BaseModel):
    account_age_days: int
    total_transactions_user: int
    avg_amount_user: float
    amount: float
    channel: str                # "web" / "app"
    merchant_category: str      # "travel", "fashion", etc.
    promo_used: int             # 0 or 1
    avs_match: int              # 0 or 1
    cvv_result: int             # 0 or 1
    three_ds_flag: int          # 0 or 1
    transaction_time: str       # ISO format, e.g. "2024-01-06T04:09:39Z"
    shipping_distance_km: float
    country: str                # e.g. "FR"
    bin_country: str            # e.g. "FR"

class FraudOutput(BaseModel):
    is_fraud: int
    fraud_probability: float
    risk_level: str