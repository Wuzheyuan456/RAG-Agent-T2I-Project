# reshaper.py
import re
import json
import uuid
from typing import List, Dict, Any

import fastapi
import pydantic
import uvicorn
import requests

app = fastapi.FastAPI()

# 配置你的真实后端
UPSTREAM_URL = r"http://127.0.0.1:8000/v1/chat/completions"


class Req(pydantic.BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    tools: List[Dict] = None
    temperature: float = 0


def extract_json_blocks(content: str) -> List[Dict[str, Any]]:
    # 1. 去掉 ```json 和 ```
    content = re.sub(r'```json\s*|\s*```', '', content)

    # 2. 用 JSONDecoder 多次解析
    decoder = json.JSONDecoder()
    idx = 0
    calls = []
    while idx < len(content):
        try:
            obj, end = decoder.raw_decode(content, idx)
            idx = end
            if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                calls.append({"name": obj["name"], "arguments": obj["arguments"]})
        except ValueError:
            idx += 1
    return calls


def build_tool_calls(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """转成 OpenAI 格式 tool_calls"""
    return [
        {
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": c["name"],
                "arguments": json.dumps(c["arguments"], ensure_ascii=False)
            }
        }
        for c in calls
    ]


@app.post("/v1/chat/completions")
def reshape(req: Req):
    # 1. 先走真实后端
    resp = requests.post(UPSTREAM_URL, json=req.dict()).json()
    choice = resp["choices"][0]
    content = choice["message"]["content"] or ""
    print(content)
    # 2. 提取假 json 块
    calls = extract_json_blocks(content)
    print(calls)
    if calls:
        choice["message"]["content"] = ""
        choice["message"]["tool_calls"] = build_tool_calls(calls)
        choice["finish_reason"] = "tool_calls"
        resp["choices"][0] = choice

    return resp


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)