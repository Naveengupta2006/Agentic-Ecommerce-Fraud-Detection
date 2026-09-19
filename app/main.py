from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import TransactionInput, FraudOutput
from app.predict import predict_fraud

app = FastAPI(title="E-commerce Fraud Detection API")

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
    result = predict_fraud(transaction.dict())
    return result