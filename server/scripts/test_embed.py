"""验证 embedding 连接。"""
import httpx
import os

key = os.environ.get("EMBEDDING_API_KEY", "sk-your-key-here")
url = os.environ.get("EMBEDDING_BASE_URL", "http://localhost:3000/v1")

r = httpx.post(url + "/embeddings", json={"model": "bge-m3", "input": "hello"}, headers={
    "Authorization": "Bearer " + key,
    "Content-Type": "application/json"
}, timeout=15)
print("status:", r.status_code)
if r.status_code == 200:
    data = r.json()
    print("got", len(data.get("data", [])), "embeddings, dim:", len(data["data"][0]["embedding"]) if data.get("data") else 0)
else:
    print("error:", r.text[:300])