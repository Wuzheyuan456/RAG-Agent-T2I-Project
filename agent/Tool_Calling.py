# pip install openai langchain-openai
import os
from openai import OpenAI
import re

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="dummy")

# 1. 定义工具列表（OpenAI 格式）
tools = [
    {
        "type": "function",
        "function": {
            "name": "weather_search",
            "description": "查询城市今日天气，输入：城市名（如 Beijing）",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如 Beijing"}
                },
                "required": ["city"]
            }
        }
    },
    # 下面是你已有的两个工具
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式，返回数字",
            "parameters": {
                "type": "object",
                "properties": {"expr": {"type": "string", "description": "数学表达式"}},
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
                "properties": {"prompt": {"type": "string", "description": "英文提示词"}},
                "required": ["prompt"]
            }
        }
    }
]
def weather_search(city: str) -> str:
    import random
    weather = random.choice(["晴", "小雨", "阴"])
    temp = random.randint(20, 30)
    return f"{city}今日{weather}，{temp}°C"
# 2. 本地 Python 实现（名字必须和上面 name 一致）
def calculator(expr: str) -> float:
    return eval(expr)

def image_generator(prompt: str) -> str:
    # 用 picsum 当免费图床
    from hashlib import md5
    h = md5(prompt.encode()).hexdigest()[:8]
    return f"https://picsum.photos/800/600?random={h}"
def exec_json_calls(content: str) -> bool:
    try:
        # (?s) 让 . 匹配换行；取最内层 ```json ... ```
        block = re.search(r'(?s)```json\s*(\[.*?\])\s*```', content).group(1)
        calls = json.loads(block)
        for call in calls:
            func_name = call["name"]
            args = call["arguments"]
            obs = globals()[func_name](**args)
            print(f">>> 执行 {func_name}({args}) → {obs}")
        return True
    except Exception as e:
        print(">>> 正则/JSON 失败:", e)
        return False
# 3. 对话循环（ReAct 由 OpenAI 内部完成）
# messages = [{"role": "user", "content": "计算 3*(7+2) 并生成对应数量的小猫图片"}]
messages = [
   # {"role": "system", "content": "以 JSON 格式调用，不要输出 XML。"},
    {"role": "user", "content": "帮我查今日beijing天气并生成一张雨天海报"}
]
import json
MAX_ROUND = 4
for rnd in range(1, MAX_ROUND + 1):
    resp = client.chat.completions.create(
        model="qwen2_5_coder_7b",
        messages=messages,
        tools=tools,
        #tool_choice="required",   # ① 强制至少调一次；想口嗨时改 auto
        tool_choice="auto",   # ① 强制至少调一次；想口嗨时改 auto
        temperature=0
    )

    choice = resp.choices[0]
    content = choice.message.content or ""

    # 1. 协议层 tool_calls → 执行并写回
    if choice.message.tool_calls:
        messages.append(choice.message)
        for tc in choice.message.tool_calls:
            args = json.loads(tc.function.arguments)
            obs = globals()[tc.function.name](**args)
            print(f">>> 执行 {tc.function.name}({tc.function.arguments}) → {obs}")
            messages.append({
                "tool_call_id": tc.id,
                "role": "tool",
                "name": tc.function.name,
                "content": str(obs)
            })
        continue

    # 2. 口嗨 JSON → 客户端兜底执行
    if exec_json_calls(content):
        # 清空 tools，防止模型再口嗨
        print("Final Answer:", resp.choices[0].message.content)
        break

    # 3. 真正自然语言结束
    print("Final Answer:", content)
    break
else:
    print("Final Answer: 达到最大轮数，结果已执行完成。")