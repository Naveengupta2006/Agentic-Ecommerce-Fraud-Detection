import sqlite3
import shap
from app.predict import model, preprocess

DB_PATH = "data/fraud.db"

# Calibrated model ke andar jo fitted XGBoost hai, SHAP usi par chalta hai
_base_xgb = model.calibrated_classifiers_[0].estimator
_explainer = shap.TreeExplainer(_base_xgb)


def get_user_history(user_id: int, before_time: str = None) -> dict:
    query = """SELECT COUNT(*), AVG(amount), SUM(is_fraud), MAX(account_age_days)
               FROM transactions WHERE user_id = ?"""
    params = [user_id]
    if before_time:
        query += " AND transaction_time < ?"
        params.append(before_time)
    con = sqlite3.connect(DB_PATH)
    row = con.execute(query, params).fetchone()
    con.close()
    return {
        "total_transactions": row[0],
        "avg_amount": round(row[1] or 0, 2),
        "past_confirmed_frauds": row[2] or 0,
        "account_age_days": row[3],
    }


def check_velocity(user_id: int, as_of_time: str, window_hours: int = 72) -> dict:
    con = sqlite3.connect(DB_PATH)
    count = con.execute(
        """SELECT COUNT(*) FROM transactions
           WHERE user_id = ? AND transaction_time < ?
           AND transaction_time >= datetime(?, ?)""",
        (user_id, as_of_time, as_of_time, f"-{window_hours} hours"),
    ).fetchone()[0]
    con.close()
    return {"window_hours": window_hours, "transactions_in_window": count}

def explain_score(txn: dict, top_k: int = 4) -> list:
    X = preprocess(txn)
    values = _explainer.shap_values(X)[0]
    ranked = sorted(zip(X.columns, values), key=lambda p: abs(p[1]), reverse=True)[:top_k]
    return [
        {"feature": name,
         "effect": "raises risk" if v > 0 else "lowers risk",
         "strength": round(float(abs(v)), 3)}
        for name, v in ranked
    ]