#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReAct Demo: 查天气 → 生成雨天海报
运行前设置环境变量：
export OPENAI_API_KEY="sk-你的key"
"""

import os
import random
import requests
from typing import Optional

from langchain_openai import ChatOpenAI          # 聊天接口
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import BaseTool            # 新基类，兼容 Pydantic v2
import hashlib, datetime as dt
# -------------------- 1. 自定义工具 -------------------- #

class WeatherTool(BaseTool):
    name: str = "weather_search"
    description: str = "查询城市今日天气，输入：城市名（如 Beijing）"

    def _run(self, city: str) -> str:
        # Mock：随机返回晴/雨/阴
        weather = random.choice(["晴", "小雨", "阴"])
        temp = random.randint(20, 30)
        return f"{city}今日{weather}，{temp}°C"

    async def _arun(self, city: str):
        raise NotImplementedError


class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "计算数学表达式，例如 5+3*2"

    def _run(self, expr: str) -> str:
        try:
            return str(eval(expr))
        except Exception as e:
            return f"计算错误：{e}"

    async def _arun(self, expr: str):
        raise NotImplementedError


class ImageGenTool(BaseTool):
    name: str = "image_generator"
    description: str = "根据英文提示词生成图片，返回图片URL"

    # def _run(self, prompt: str) -> str:
    #     # 用免费 Unsplash Source 做 Mock：直接返回一张关键字图
    #     # 真实场景可换成 DALL·E / Stable Diffusion API
    #     url = f"https://source.unsplash.com/800x600/?{prompt.replace(' ', ',')}"
    #     # 访问一次拿到跳转后的真实地址
    #     r = requests.head(url, allow_redirects=True)
    #     return r.url
    def _run(self, prompt: str) -> str:
        h = hashlib.md5(prompt.encode()).hexdigest()[:8]
        url = f"https://picsum.photos/800/600?random={h}"
        r = requests.head(url, allow_redirects=True, verify=False, timeout=10)
        return r.url
    async def _arun(self, prompt: str):
        raise NotImplementedError


# -------------------- 2. 构建 Agent -------------------- #

tools = [WeatherTool(), CalculatorTool(), ImageGenTool()]

# llm = OpenAI(
#     temperature=0,
#     openai_api_key=os.getenv("OPENAI_API_KEY"),
#     model_name="gpt-3.5-turbo-instruct"
# )
llm = ChatOpenAI(
    openai_api_base="http://127.0.0.1:8000/v1",  # 1. 去掉空格
    model_name="chatglm3-6b",
    temperature=0.7,
    max_tokens=1024,              # 3. 本地模型上下文短，给足长度
    openai_api_key="dummy"
)
from string import Formatter

class SafeFormatter(Formatter):
    def get_value(self, key, args, kwargs):
        # 找不到变量就原样输出
        return kwargs.get(key, "{%s}" % key)

safe_fmt = SafeFormatter()
examples = (
    "Human: 北京天气怎样，再给我一张雨天海报\n"
    "Thought: 我需要先查天气\n"
    "Action: weather_search\n"
    "Action Input: Beijing\n"
    "Observation: Beijing今日小雨，23°C\n"
    "Thought: 天气已拿到，现在生成海报\n"
    "Action: image_generator\n"
    "Action Input: rainy day poster\n"
    "Observation: https://images.unsplash.com/xxx.jpg\n"
    "Thought: 信息齐全，可以回答用户了\n"
    "Final Answer: 北京今天小雨，23°C，海报已生成： https://images.unsplash.com/xxx.jpg\n\n"
)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    agent_kwargs={
        "prefix": examples,          # 关键：给样例
        # "format_instructions": (
        #     "你必须使用如下格式：\n\n"
        #     "Thought: 思考下一步应该做什么\n"
        #     "Action:\n"
        #     "```json\n"
        #     '{{{{"action": "工具名", "action_input": "工具参数"}}}}\n'
        #     "```\n"
        #     "Observation: 工具返回的结果\n"
        #     "（重复上面三步）\n"
        #     "Final Answer: 对原始问题的最终回答"
        # )
    }
)

# agent = initialize_agent(
#     tools=tools,
#     llm=llm,
#     agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,   # 换这个
#     verbose=True,
#     handle_parsing_errors=True,
#     max_iterations=5,
#     # agent_kwargs={...}  也不用再给 format_instructions
# )
# agent = initialize_agent(
#     tools=tools,
#     llm=llm,
#     agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
#     handle_parsing_errors=True,
#     max_iterations=5,
#     agent_kwargs={
#         "prefix": "",
#         "format_instructions": format_instructions
#     }
# )

# agent = initialize_agent(
#     tools=tools,
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
#     handle_parsing_errors=True,
# )

# -------------------- 3. 运行一次 -------------------- #

if __name__ == "__main__":
    city = "Beijing"
    result = agent.invoke({"input": f"帮我查今日{city}天气并生成一张雨天海报"})

    print("\n=== 最终结果 ===")
    print(result["output"])
