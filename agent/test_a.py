from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="dummy")

tools = [{
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "计算表达式",
        "parameters": {"type": "object", "properties": {"expr": {"type": "string"}},"required": ["expr"]}
    }
}]

resp = client.chat.completions.create(
    model="qwen2_5_coder_7b",
    messages=[{"role": "user", "content": "计算 3*(7+2)"}],
    tools=tools,
    tool_choice="required",  # 强制调用
    # tool_choice="auto"
)
print(resp.choices[0].message.tool_calls)   # 应返回合法 tool_calls