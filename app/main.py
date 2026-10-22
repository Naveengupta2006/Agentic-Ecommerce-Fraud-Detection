from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import TransactionInput, FraudOutput, AgentInput, AgentOutput
from app.predict import predict_fraud
from app.agent.agent import review_transaction
from app.agent.audit import log_decision, recent_decisions

app = FastAPI(title="Agentic E-commerce Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/predict", response_model=FraudOutput)
def predict(transaction: TransactionInput):
    return predict_fraud(transaction.dict())

@app.post("/agent/review", response_model=AgentOutput)
def agent_review(req: AgentInput):
    data = req.dict()
    user_id = data.pop("user_id")
    result = review_transaction(data, user_id)
    log_decision(user_id, data, result)
    return result


@app.get("/agent/decisions")
def agent_decisions(limit: int = 20):
    return recent_decisions(limit)