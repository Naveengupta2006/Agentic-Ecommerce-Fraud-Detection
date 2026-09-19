import joblib
import pandas as pd

model = joblib.load("models/fraud_model.pkl")       # calibrated model
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")

FRAUD_THRESHOLD = 0.3   # tuned threshold, not default 0.5

FEATURE_ORDER = [
    'account_age_days', 'total_transactions_user', 'avg_amount_user',
    'amount', 'channel', 'merchant_category', 'promo_used',
    'avs_match', 'cvv_result', 'three_ds_flag',
    'shipping_distance_km', 'country_mismatch', 'hour'
]

NUM_COLS = ['account_age_days', 'total_transactions_user', 'avg_amount_user',
            'amount', 'shipping_distance_km', 'hour']

def preprocess(input_dict):
    df = pd.DataFrame([input_dict])

    # derived features (same logic as notebook EDA/feature engineering)
    df['country_mismatch'] = (df['country'] != df['bin_country']).astype(int)
    df['hour'] = pd.to_datetime(df['transaction_time']).dt.hour

    df.drop(columns=['country', 'bin_country', 'transaction_time'], inplace=True)

    # categorical encoding
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    # numeric scaling
    df[NUM_COLS] = scaler.transform(df[NUM_COLS])

    # enforce exact column order the model was trained on
    df = df[FEATURE_ORDER]
    return df

def predict_fraud(input_dict):
    df = preprocess(input_dict)
    prob = model.predict_proba(df)[0][1]
    pred = int(prob >= FRAUD_THRESHOLD)
    risk = "High" if prob >= 0.6 else "Medium" if prob >= FRAUD_THRESHOLD else "Low"
    return {
        "is_fraud": pred,
        "fraud_probability": round(float(prob), 4),
        "risk_level": risk
    }