# reshaper.py
import json
import  fastapi, uvicorn, pydantic

import re
import xml.etree.ElementTree as ET
import uuid
from typing import List

import fastapi
import pydantic
import requests

app = fastapi.FastAPI()
VLLM = "http://127.0.0.1:8000/v1/chat/completions"


class Req(pydantic.BaseModel):
    model: str
    messages: List[dict]
    tools: List[dict] = None
    temperature: float = 0


def xml2calls(content: str):
    """把嵌套 XML 转成 OpenAI tool_calls 格式"""
    try:
        # 去掉可能的 Markdown 包裹
        content = re.sub(r"```xml\s*|\s*```", "", content.strip(), flags=re.I)
        root = ET.fromstring(content)
        calls = []
        for tc in root.findall(".//tool_call"):
            name_node = tc.find("name")
            args_node = tc.find("arguments")
            if name_node is None or args_node is None:
                continue
            func = name_node.text.strip()
            args = args_node.text.strip()  # 这里是 JSON 字符串
            calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": func, "arguments": args}
            })
        return calls
    except Exception as e:
        print("XML 解析失败:", e)
        return []


@app.post("/v1/chat/completions")
def reshape(req: Req):
    raw = requests.post(VLLM, json=req.model_dump()).json()
    choice = raw["choices"][0]
    content = choice["message"]["content"] or ""

    calls = xml2calls(content)
    if calls:
        choice["message"]["content"] = ""
        choice["message"]["tool_calls"] = calls
        choice["finish_reason"] = "tool_calls"
        raw["choices"][0] = choice
    return raw

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)