import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Use a vision-capable model — override here or change NVIDIA_MODEL in .env
VISION_MODEL = "meta/llama-3.2-11b-vision-instruct"

invoke_url = f"{os.environ['NVIDIA_BASE_URL']}/chat/completions"
stream = False

headers = {
    "Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}",
    "Accept": "text/event-stream" if stream else "application/json",
}

payload = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://assets.ngc.nvidia.com/products/api-catalog/phi-3-5-vision/example1b.jpg"
                    }
                }
            ]
        }
    ],
    "model": VISION_MODEL,
    "max_tokens": 1024,
    "stream": stream,
    "temperature": 0.7,
    "top_p": 0.95,
}

print(f"Calling model: {VISION_MODEL}")
print(f"Endpoint: {invoke_url}\n")

response = requests.post(invoke_url, headers=headers, json=payload, stream=stream)

if stream:
    for line in response.iter_lines():
        if line:
            print(line.decode("utf-8"))
else:
    if not response.ok or not response.text.strip():
        print(f"ERROR — HTTP {response.status_code}")
        print("Raw response:", response.text[:500] or "<empty body>")
    else:
        data = response.json()
        if "choices" in data:
            print("Response:", data["choices"][0]["message"]["content"])
        else:
            print("Full response:", data)

