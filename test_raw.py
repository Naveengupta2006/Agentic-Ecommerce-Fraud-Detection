import time
from app.agent.agent import client, MODEL, SYSTEM_PROMPT

user_msg = (
    'Transaction: {"amount": 420.0, "channel": "web", "cvv_result": 0, "three_ds_flag": 0, '
    '"country": "FR", "bin_country": "US"}\n'
    "ML model fraud probability: 0.0202\nInvestigate and decide."
)

start = time.time()
resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ],
    temperature=0.2,
    max_tokens=2048,
)
print("time:", round(time.time() - start, 1), "sec")
print("finish_reason:", resp.choices[0].finish_reason)
print("CONTENT:", repr(resp.choices[0].message.content))