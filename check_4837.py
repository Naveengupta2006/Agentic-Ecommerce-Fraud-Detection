import sqlite3
from app.agent.tools import get_user_history

con = sqlite3.connect("data/fraud.db")
row = con.execute(
    "SELECT user_id, transaction_time FROM transactions WHERE transaction_id = 4837"
).fetchone()
con.close()

print(row)
print(get_user_history(row[0], row[1]))