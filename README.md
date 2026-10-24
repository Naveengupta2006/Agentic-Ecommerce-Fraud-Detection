# Agentic E-commerce Fraud Detection & Risk Intelligence System

A fraud-scoring model and an LLM investigation agent behind one FastAPI service.

An XGBoost classifier (with isotonic calibration) scores every transaction. Only the transactions the model is unsure about (score between 0.15 and 0.85, roughly 1% of the test sample) are passed to an agent. The agent looks up the customer's history and the reasons behind the model's score, then decides: approve, ask for extra verification, hold for human review, or block. Every decision is written to an audit log.

---

## How it works

```text
Transaction
    |
    v
XGBoost + isotonic calibration  ->  fraud score
    |
    |-- score < 0.15 ----->  approve  (model alone)
    |-- score > 0.85 ----->  block    (model alone; human review if amount >= $1000)
    |-- 0.15 to 0.85 ----->  LLM agent
                               |- explain_score      why the model gave this score (SHAP)
                               |- get_user_history   past transactions, past confirmed frauds
                               |- check_velocity     purchases in the last 72 hours
                               v
                             decision + reasoning + evidence
                               v
                             guardrails  ->  audit log (SQLite)
```

The model decides the clear cases quickly and cheaply. The LLM is only called where the score alone is not enough, which keeps latency and API cost low.

---

## Results

**Data:** 299,695 transactions, 17 columns, 2.21% fraud (95% CI 2.15% to 2.26%). Stratified 80/20 split: 59,939 test transactions, 1,322 of them fraud.

### Model

| Model | Threshold | Fraud precision | Fraud recall |
|---|---|---|---|
| XGBoost, class-weighted (`scale_pos_weight` 44.3) | 0.5 | 0.48 | 0.86 |
| + isotonic calibration | 0.5 | 0.95 | 0.74 |
| + isotonic calibration (**deployed**) | 0.3 | 0.89 | 0.78 |

- The class-weighted model reaches **ROC-AUC 0.974** and **PR-AUC 0.857**.
- Class weighting pushed the raw probabilities too high. Isotonic calibration fixed this: the calibration curve moved onto the diagonal and the Brier score fell from 0.0194 to 0.0057.
- The deployed threshold is 0.3. Risk levels: below 0.3 Low, 0.3 to 0.6 Medium, 0.6 and above High.

### What the data shows

Each check was tested with a chi-square test (p < 0.001 for all four). Transaction amount differs between fraud and legitimate transactions (Mann-Whitney U, p about 1e-14). Time of day has almost no effect (fraud rate stays between 1.9% and 2.5% across all hours).

| Signal | Fraud rate, check passed | Fraud rate, check failed | Ratio |
|---|---|---|---|
| AVS (address check) | 0.76% | 9.67% | 12.7x |
| CVV (security code) | 0.97% | 10.61% | 10.9x |
| 3D Secure | 0.96% | 6.75% | 7.0x |
| Card country differs from purchase country | 1.43% | 11.28% | 7.9x |

### Agent evaluation

The agent was evaluated on 40 transactions from the held-out test split whose model score fell between 0.15 and 0.85 (19 actual fraud, 21 legitimate). None of these 40 were used to write the prompt or the guardrails.

| Policy | Fraud approved | Legitimate blocked | Legitimate sent to extra verification |
|---|---|---|---|
| Model only (block at 0.3 or higher) | 5 | 9 | 0 |
| Step up everything | 0 | 0 | 21 |
| Agent | 0 | 0 | 15 |

- The agent approved 6 legitimate customers. None of the 6 was fraud.
- It sent the three largest frauds (all above $1,000) to human review.
- It never blocked a transaction. Inside its step-up group it did not separate fraud from legitimate: 16 fraud and 15 legitimate.
- In short, the agent works as a cautious triage step. It clears some clean cases and escalates the rest. It did not show that it catches fraud better than the model.
- Which policy is cheapest depends on two business numbers this project does not have: how much fraud step-up verification stops, and how much a human reviewer catches. Across 9 combinations of these assumptions (`sensitivity.py`) the agent was cheapest in 3, "step up everything" in 4 and the model-only policy in 2.

---

## Agent design

**Tools.** The agent calls them without arguments. They always run on the current transaction and its own user, so the agent cannot read another customer's data. Lookups only use transactions that happened before the one being reviewed.

**Decision rules** (in the system prompt):
1. Below 0.5 the agent chooses approve or step-up, never block.
2. At 0.5 or above it never approves.
3. Hold for review is for amounts of $1,000 or more, or when the agent cannot decide.
4. When signals are mixed it leans towards step-up.
5. A past confirmed fraud alone never justifies a block.

**Guardrails in code:**
- The agent cannot approve when the model score is 0.5 or higher.
- A block on an amount of $1,000 or more is sent to a human.
- If the LLM fails, times out or returns something unusable, the transaction goes to human review. It is never silently approved or blocked.
- Only the four allowed decisions are accepted. Temperature is 0. Failed calls are retried with backoff.

The `confidence` field is the LLM's own estimate. It is not a calibrated probability.

---

## Repository structure

```text
├── app/
│   ├── main.py            # FastAPI: /predict, /agent/review, /agent/decisions
│   ├── predict.py         # preprocessing and model scoring (threshold 0.3)
│   ├── schemas.py         # request and response models
│   └── agent/
│       ├── agent.py       # LLM loop, decision rules, guardrails
│       ├── tools.py       # explain_score, get_user_history, check_velocity
│       └── audit.py       # SQLite audit log
├── scripts/
│   └── build_db.py        # loads the dataset into data/fraud.db for the agent's tools
├── evaluate_agent.py      # agent vs baselines on held-out uncertain cases
├── sensitivity.py         # cost comparison under different assumptions
├── frontend/
│   ├── index.html         # analyst dashboard (/predict)
│   ├── agent.html         # agent console (/agent/review and audit log)
│   ├── style.css
│   └── script.js
├── notebooks/
│   └── 01_eda_and_model.ipynb   # EDA, statistical tests, training, calibration, threshold tuning
├── models/                # generated: fraud_model.pkl, scaler.pkl, encoders.pkl (not in repo)
├── data/                  # not in repo: raw dataset, fraud.db, audit.db, evaluation results
├── .env                   # not in repo: API key
└── README.md
```

---

## Setup

### 1. Install

```bash
git clone https://github.com/Naveengupta2006/E-commerce-Fraud-Detection---Risk-Intelligence-System.git
cd E-commerce-Fraud-Detection---Risk-Intelligence-System

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

pip install fastapi uvicorn pydantic scikit-learn xgboost pandas numpy joblib shap openai python-dotenv
```

If your copy of the dataset is an `.xlsx` file, also run `pip install openpyxl`.

### 2. Dataset and models

The dataset, the trained models and the databases are not in the repository.

1. Put the dataset at `data/raw/Ecommerce_Fraud_Detection.csv`.
2. Run every cell of `notebooks/01_eda_and_model.ipynb`. It trains the model, calibrates it and writes `fraud_model.pkl`, `scaler.pkl` and `encoders.pkl` into `models/`.
3. Build the database the agent's tools read from:
   ```bash
   python scripts/build_db.py
   ```

### 3. LLM key

The agent calls Gemini through its OpenAI-compatible endpoint. Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-3.5-flash-lite
```

Google retires model names from time to time. If you get a 404, pick a current Flash model in Google AI Studio. Never commit `.env`.

### 4. Run

```bash
uvicorn app.main:app --reload
```

- Swagger UI: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/

In a second terminal, serve the frontend:

```bash
python -m http.server 5500 --directory frontend
```

- Agent console: http://localhost:5500/agent.html
- Analyst dashboard: http://localhost:5500/index.html

Uncertain cases take 15 to 40 seconds because the agent makes several LLM calls.

### 5. Re-run the evaluation (optional)

```bash
python evaluate_agent.py
python sensitivity.py
```

`evaluate_agent.py` makes many LLM calls and takes 20 to 30 minutes. On a free API tier it may hit rate limits. Re-run it and it continues where it stopped.

---

## API reference

### `POST /predict`

Scores one transaction with the model only.

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

Response:

```json
{
  "is_fraud": 0,
  "fraud_probability": 0.0009,
  "risk_level": "Low"
}
```

### `POST /agent/review`

Same fields as `/predict` plus `user_id`. Transactions with a clear model score are decided by the model alone and `used_agent` is `false`. Uncertain ones are investigated by the agent.

```json
{
  "user_id": 97,
  "account_age_days": 351,
  "total_transactions_user": 49,
  "avg_amount_user": 121.04,
  "amount": 105.33,
  "channel": "web",
  "merchant_category": "travel",
  "promo_used": 0,
  "avs_match": 0,
  "cvv_result": 0,
  "three_ds_flag": 0,
  "transaction_time": "2024-08-28T17:45:08Z",
  "shipping_distance_km": 1171.64,
  "country": "TR",
  "bin_country": "PL"
}
```

Example response (the wording of `reasoning` varies between runs):

```json
{
  "decision": "step_up_verification",
  "confidence": 0.85,
  "model_score": 0.2615,
  "reasoning": "The ML score is below 0.5, so the transaction is not blocked. Failed AVS and CVV, a long shipping distance and a country mismatch are balanced by a modest amount and low velocity, so extra verification is the safest choice.",
  "evidence": [
    "ML score 0.2615",
    "failed AVS and CVV",
    "country mismatch TR/PL",
    "1 transaction in the last 72 hours"
  ],
  "tool_calls": ["explain_score", "get_user_history", "check_velocity"],
  "used_agent": true
}
```

Possible decisions: `approve`, `step_up_verification`, `hold_for_review`, `block`.

### `GET /agent/decisions?limit=20`

Returns the most recent decisions from the audit log, including the model score, the decision, the tools used and the reasoning.

---

## Limitations

- **Small evaluation.** 40 cases (19 fraud), one run. Three frauds above $1,000 dominate the cost totals. LLM output can differ between runs.
- **Not an independent test of the baselines.** The 0.3 threshold and the 0.15 to 0.85 band were chosen on the same dataset's test split, which slightly favours the model-only baseline.
- **Cost results rest on assumptions** about step-up effectiveness and reviewer accuracy, not on real business data.
- **Label timing.** The agent sees whether a user's earlier transactions were fraud. In a real system such labels arrive late (chargebacks take weeks), so the evaluation is optimistic on this point.
- **Cited evidence is not checked automatically** against the tool output. It was spot-checked by hand.
- **Approximate explanations.** `explain_score` uses SHAP on one of the five XGBoost models inside the calibrated ensemble.
- **Not production-ready.** The endpoints have no authentication or rate limiting, CORS is open, the audit log is a local SQLite file, and the free LLM tier has quota limits. When the LLM is unavailable, transactions go to human review.
- **Nothing is deployed yet.**

---

## License

MIT