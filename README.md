# 🛡️ E-Commerce Fraud Detection & Risk Intelligence System

An end-to-end Machine Learning and Risk Decisioning engine designed to identify anomalous and fraudulent e-commerce transactions in real time. Powered by an **XGBoost Classifier with Isotonic Calibration**, deployed via **FastAPI**, and accompanied by a real-time **Risk Analyst Dashboard**.

---

## 🚀 Key Highlights & Performance

- **Discrimination Capability**: **0.974 ROC-AUC** and **0.857 PR-AUC** on highly imbalanced real-world transaction distributions.
- **Calibrated Probability**: Employs Isotonic Probability Calibration ensuring risk scores reflect actual empirical likelihood of fraud.
- **Precision / Recall Balance**: Tuned operational threshold ($0.30$) achieving **86% Fraud Recall** and **95% Calibrated Alert Precision**.
- **Key Risk Multipliers Uncovered**:
  - **AVS Mismatch**: $12.7\times$ elevated fraud incidence ($0.76\%$ vs. $9.67\%$).
  - **Country Mismatch**: $7.9\times$ higher risk ($1.43\%$ vs. $11.28\%$).
  - **3D Secure Failure**: $7.0\times$ risk amplification ($0.96\%$ vs. $6.75\%$).
  - **CVV Mismatch**: $4.8\times$ risk increase ($0.97\%$ vs. $10.61\%$).

---

## 📁 Repository Structure

```text
├── app/
│   ├── main.py          # FastAPI application & API endpoints
│   ├── predict.py       # Inference pipeline, feature extraction, & scoring
│   └── schemas.py       # Pydantic request & response models
├── frontend/
│   ├── index.html       # Risk Analyst Dashboard (React 18 + Babel)
│   ├── style.css        # Cyber-fintech dark UI styling
│   └── script.js        # Standalone JS client utilities
├── notebooks/
│   └── 01_eda_and_model.ipynb   # Complete exploratory analysis, EDA & model training
├── models/              # Serialized artifacts (scaler.pkl, encoders.pkl, fraud_model.pkl)
├── data/                # Dataset directory (raw transactions CSV)
├── .gitignore           # Excludes data/ and large model weights
└── README.md
```

---

## ⚠️ Dataset & Model Regeneration

> **Note on Best Practices**: Large raw datasets (`data/`) and compiled model binary weights (`models/*.pkl`) are deliberately excluded from this Git repository via `.gitignore` to keep the repository lightweight and adhere to version control standards.

### How to Regenerate Models & Data:
1. Open and execute the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/01_eda_and_model.ipynb
   ```
2. Run all cells sequentially. The notebook will:
   - Perform exploratory data analysis and feature engineering.
   - Train the XGBoost model, perform hyperparameter tuning, and apply isotonic probability calibration.
   - Automatically export the fitted `fraud_model.pkl`, `scaler.pkl`, and `encoders.pkl` into the `models/` folder.

---

## 🛠️ Installation & Setup

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/Naveengupta2006/E-commerce-Fraud-Detection---Risk-Intelligence-System.git
cd "Ecommerce Fraud Detection"

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install fastapi uvicorn pydantic scikit-learn xgboost pandas numpy joblib
```

### 2. Launch FastAPI Inference Server
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
Interactive OpenAPI documentation will be accessible at:
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Check: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

### 3. Open the Risk Analyst Dashboard
Open `frontend/index.html` directly in any modern browser, or serve it using Python's static server:
```bash
# In a separate terminal:
python -m http.server 3000 --directory frontend
```
Then visit [http://localhost:3000](http://localhost:3000) in your web browser.

---

## 📡 API Endpoint Reference

### `POST /predict`
Evaluates transaction attributes and outputs risk metrics:

#### Sample Request:
```json
{
  "account_age_days": 900,
  "total_transactions_user": 85,
  "avg_amount_user": 120.0,
  "amount": 115.0,
  "channel": "app",
  "merchant_category": "grocery",
  "promo_used": 0,
  "avs_match": 1,
  "cvv_result": 1,
  "three_ds_flag": 1,
  "transaction_time": "2024-02-10T14:30:00Z",
  "shipping_distance_km": 12.0,
  "country": "FR",
  "bin_country": "FR"
}
```

#### Sample Response:
```json
{
  "is_fraud": 0,
  "fraud_probability": 0.0142,
  "risk_level": "Low"
}
```
- **Low Risk**: $P(\text{fraud}) < 0.30$ (Instant Approval)
- **Medium Risk**: $0.30 \le P(\text{fraud}) < 0.60$ (Step-Up Authentication / Manual Review)
- **High Risk**: $P(\text{fraud}) \ge 0.60$ (Automated Decline / Urgent Alert)

---

## 📄 License
MIT License. Built for enterprise transaction security and risk intelligence.
