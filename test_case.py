import json
import sqlite3
import pandas as pd
from evaluate_agent import to_txn
from app.agent.agent import review_transaction

con = sqlite3.connect("data/fraud.db")
df = pd.read_sql("SELECT * FROM transactions WHERE transaction_id = 4837", con)
con.close()

row = df.iloc[0]
txn = to_txn(row)
print(json.dumps(txn, indent=2))

r = review_transaction(txn, user_id=int(row["user_id"]))
print(json.dumps(r, indent=2))