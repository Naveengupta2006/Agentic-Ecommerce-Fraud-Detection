import glob
import sqlite3
import pandas as pd

path = glob.glob("data/raw/Ecommerce_Fraud_Detection*")[0]
print("reading:", path)

df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
df["transaction_time"] = pd.to_datetime(df["transaction_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")

con = sqlite3.connect("data/fraud.db")
df.to_sql("transactions", con, if_exists="replace", index=False)
con.execute("CREATE INDEX idx_user_time ON transactions(user_id, transaction_time)")
con.commit()
con.close()

print("done, rows:", len(df))