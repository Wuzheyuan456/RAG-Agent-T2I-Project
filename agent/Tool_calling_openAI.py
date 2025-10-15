import os, json, re
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="dummy")

# ---------- 1. 工具定义 ----------
tools = [
    {
        "type": "function",
        "function": {
            "name": "weather_search",
            "description": "查询城市今日天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {"expr": {"type": "string"}},
                "required": ["expr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "image_generator",
            "description": "根据英文提示词返回一张图片 URL",
            "parameters": {
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
                "required": ["prompt"]
            }
        }
    }
]

# ---------- 2. 工具实现 ----------
def weather_search(city: str) -> str:
    import random
    w = random.choice(["晴", "小雨", "阴"])
    t = random.randint(20, 30)
    return f"{city}今日{w}，{t}°C"

def calculator(expr: str) -> float:
    # 简单保护，生产请用 ast.literal_eval 或专门 parser
    return eval(expr)

def image_generator(prompt: str) -> str:
    from hashlib import md5
    h = md5(prompt.encode()).hexdigest()[:8]
    return f"https://picsum.photos/800/600?random={h}"

tool_map = {
    "weather_search": weather_search,
    "calculator": calculator,
    "image_generator": image_generator
}

# ---------- 3. 对话 ----------
messages = [
    {"role": "user", "content": "帮我查今日 Beijing 天气并生成一张雨天海报"}
]

MAX_ROUND = 5
for _ in range(MAX_ROUND):
    response = client.chat.completions.create(
        model="qwen2_5_coder_7b",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0
    )

    choice = response.choices[0]
    msg = choice.message

    # ---- 情况 1：模型决定调用工具 ----
    if msg.tool_calls:
        # 先把助手这条“我要调工具”的消息存进去
        messages.append(msg)
        for tc in msg.tool_calls:
            func = tc.function.name
            args = json.loads(tc.function.arguments)
            obs = tool_map[func](**args)
            print(f"-- 调用 {func}({args}) -> {obs}")
            # 把工具返回结果以 role=tool 写回
            messages.append({
                "tool_call_id": tc.id,
                "role": "tool",
                "name": func,
                "content": str(obs)
            })
        continue   # 继续循环，让模型看到观测值

    # # ---- 情况 2：模型不调用工具，直接给答案 ----
    # print("Final Answer:", msg.content)
    # break
    # ↓↓↓ 手动再请求一次，让模型总结 ↓↓↓
    summary_resp = client.chat.completions.create(
        model="qwen2_5_coder_7b",
        messages=messages,
        tools=[],
        temperature=0
    )
    print("Final Answer:", summary_resp.choices[0].message.content)
    break
else:
    print("达到最大轮数，强制结束。")