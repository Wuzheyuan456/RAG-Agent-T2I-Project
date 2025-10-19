# orchestrator.py
import json, inspect, time
from types import SimpleNamespace
from tools import *
from openai import OpenAI

LLM = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="dummy")

functions = {
    "llm_agent": llm_agent,
    "sd_agent": sd_agent,
    "rag_agent": rag_agent,
    "chart_agent": chart_agent,
    "pdf_agent": pdf_agent
}
# 在文件开头加一个映射表
RESULT_KEY_MAP = {
    "rag_agent": "rag",
    "llm_agent": "llm",
    "chart_agent": "chart",
    "sd_agent": "sd",
    "pdf_agent": "pdf"
}
# ---------- 工具定义 ----------
tools = []
for name, func in functions.items():
    sig = inspect.signature(func)
    props = {k: {"type": "string"} for k in sig.parameters}
    tools.append({
        "type": "function",
        "function": {
            "name": name,
            "description": func.__doc__.strip(),
            "parameters": {"type": "object", "properties": props}
        }
})
import json
print("=== 当前注册工具 ===")
for t in tools:
    print(json.dumps(t, ensure_ascii=False, indent=2))

# SYSTEM_PROMPT = (
#     "你是一次只调一个函数的调度器，必须按以下顺序执行：\n"
#     "1. llm_agent → 得到 copy\n"
#     "2. rag_agent → 得到 data\n"
#     "3. chart_agent → 用 data 生成 chart_path\n"
#     "4. sd_agent → 用 copy 生成海报图片路径 img_path\n"
#     "5. pdf_agent → 合并 copy/img_path/chart_path\n"
#     "除函数调用外不要返回任何文字！"
# )
# SYSTEM_PROMPT = """
# 你是一个严格的多步骤工作流调度器，负责生成汉服社招新海报的PDF。你必须严格按照以下五步顺序执行，不能跳过任何步骤。
#
# 当前任务：
# 为“汉服社”生成国风风格的招新海报PDF，并整合历史招新数据。
#
# 执行流程（必须严格遵守）：
# 1.  文案生成 (llm_agent): 调用 `llm_agent` 生成海报宣传文案 (copy)。
# 2.  数据检索 (rag_agent): 调用 `rag_agent` 获取“汉服社”的历史招新数据 (data)。
# 3.  图表生成 (chart_agent): 使用上一步获得的 `data` 调用 `chart_agent` 生成统计图表图片路径 (chart_path)。
# 4.  海报设计 (sd_agent): 使用第一步获得的 `copy` 调用 `sd_agent` 生成国风海报图片路径 (img_path)。
# 5.  PDF合成 (pdf_agent): 将 `copy`, `img_path`, 和 `chart_path` 提供给 `pdf_agent`，生成最终的PDF文件。
#
# 你的输出规则（极其重要）：
# - 在每一步，你必须先用一句话说明你即将执行的操作（例如：“正在调用 rag_agent 获取历史数据。”）。
# - 紧接着，你必须且只能输出一个有效的函数调用（tool call）来执行该步骤。
# - 禁止仅输出文字说明而不进行函数调用。
# - 你的响应应该看起来像这样：
#   > 正在调用 llm_agent 生成海报文案。
#   > {"name": "llm_agent", "arguments": {"product": "汉服社", "style": "国风"}}
#
#
# 现在，请一步一步执行，后续将把已经执行的内容告知给你。
# """

# ---------- 重试封装 ----------
def call_with_retry(messages, tools, func_name: str, max_retry=3):
    for attempt in range(1, max_retry + 1):
        resp = LLM.chat.completions.create(
            model="qwen2_5_coder_7b",
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": func_name}},
            temperature=0,
        )
        msg = resp.choices[0].message
        # ➜ 新增：打印模型真实返回
        print(f">>>【{func_name} 第{attempt}次】返回：{msg}")
        if msg.tool_calls:
            print(f"  ↳ 成功调用：{msg.tool_calls[0].function.name} 参数：{msg.tool_calls[0].function.arguments}")
            return msg
        # ➕ 只有当没有 tool_calls 时才追加消息（推进对话）
        if msg.content.strip():  # 如果有内容，先放 assistant 的回复
            messages.append({"role": "assistant", "content": msg.content})
        else:
            messages.append({"role": "assistant", "content": "（无内容）"})

        # 👉 明确指令
        messages.append({
            "role": "user",
            "content": f"你必须调用函数 `pdf_agent` 来生成最终海报。不要说任何话，只调用函数。"
        })
        # 没拿到，催一下
        # messages.append({"role": "assistant", "content": msg.content or ""})
        # messages.append({"role": "user", "content": "请继续调用下一个函数。"})
        # print(len(messages))
        time.sleep(0.5)
    raise RuntimeError(f"步骤 {func_name} 重试 {max_retry} 次仍无 tool_calls，中断")
# ---------- 主流程 ----------
# 替换原来的 SYSTEM_PROMPT
SYSTEM_PROMPT = """
你是一个函数调用执行器。你必须根据用户的指令，调用指定的函数并提供正确的参数。
你只能输出一个函数调用（tool call），不要输出任何其他文字。
"""
# ---------- 主流程：重构版 ----------
def run(user_query: str) -> str:
    """
    执行汉服社招新海报生成的五步工作流：
    1. llm_agent      → 生成文案 (copy)
    2. rag_agent      → 获取历史数据 (data)
    3. chart_agent    → 用 data 生成图表路径 (chart_path)
    4. sd_agent       → 用 copy 生成海报图片路径 (img_path)
    5. pdf_agent      → 合并所有内容生成 PDF 路径
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    results = {}

    # 👉 第0步：用 LLM 解析用户输入，提取 product 和 style
    parse_prompt = f"""
       请从以下用户请求中提取两个信息：
       - product: 社团名称（如“摄影社”、“舞蹈社”、“机器人社”等）
       - style: 宣传风格（如“国风”、“科技感”、“小清新”、“赛博朋克”等）

       如果未明确说明风格，默认为“国风”。

       用户请求：{user_query}

       请以 JSON 格式输出，如：
       {{"product": "摄影社", "style": "科技感"}}
       """

    try:
        response = LLM.chat.completions.create(
            model="qwen2_5_coder_7b",
            messages=[{"role": "user", "content": parse_prompt}],
            response_format={"type": "json_object"}
        )
        parsed = json.loads(response.choices[0].message.content)

        product = parsed["product"]
        style = parsed["style"]
    except:
        # 备用方案：默认值
        product = "汉服社"
        style = "国风"

    # 把解析结果存入 results，供后续占位符替换
    results["product"] = product
    results["style"] = style



    # 定义执行流程：函数名、用户指令、参数模板（支持占位符）
    workflow = [
        {
            "func": "llm_agent",
            "instruction": "正在调用 llm_agent 生成汉服社海报文案。",
            "args": {"product": "<your_product>", "style": "<your_style>"}
        },
        {
            "func": "rag_agent",
            "instruction": "正在调用 rag_agent 获取历史招新数据。",
            "args": {"product": "<your_product>"}
        },
        {
            "func": "chart_agent",
            "instruction": "正在调用 chart_agent 生成统计图表。",
            "args": {"data": "<your_rag_agent_output>"}
        },
        {
            "func": "sd_agent",
            "instruction": "正在调用 sd_agent 生成国风风格的海报图片。",
            "args": {"product": "<your_product>", "copy": "<your_llm_agent_output>", "style": "<your_style>"}
        },
        {
            "func": "pdf_agent",
            "instruction": "正在调用 pdf_agent 生成最终PDF。",
            "args": {
                "copy": "<your_llm_agent_output>",
                "img_path": "<your_sd_agent_output>",
                "chart_path": "<your_chart_agent_output>"
            }
        }
    ]

    for step in workflow:
        print(f"\n--- 执行步骤: {step['func']} ---")
        print(step["instruction"])

        # 构建当前步骤的消息
        current_messages = messages + [
            {"role": "user", "content": step["instruction"]}
        ]

        # 强制调用指定函数
        try:
            msg = call_with_retry(
                messages=current_messages,
                tools=tools,
                func_name=step["func"],
                max_retry=3
            )
        except RuntimeError as e:
            print(f"❌ 步骤失败: {e}")
            return ""

        # 解析函数调用
        tool_call = msg.tool_calls[0]
        # args = json.loads(tool_call.function.arguments)
        #
        # # 替换占位符（如 <your_llm_agent_output>）
        # args = replace_placeholder(args, results)
        args = step["args"].copy()  # ← 直接用 workflow 里的硬编码参数
        args = replace_placeholder(args, results)  # 替换 <your_xxx_output>
        # 执行本地函数
        print(f"  🔧 执行 {tool_call.function.name} 参数: {args}")
        result = functions[tool_call.function.name](**args)
        store_key = RESULT_KEY_MAP.get(tool_call.function.name, tool_call.function.name)
        results[store_key] = result

        # results[tool_call.function.name] = result

        # 打印返回结果
        print(f"  ✅ 返回: {result}")
        if isinstance(result, str):
            print(f"     长度: {len(result)} 字符")

        # 更新全局 messages（用于 trace，可选）
        messages.append(msg)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": json.dumps(result)
        })

    # 返回最终 PDF 路径
    final_pdf = results.get("pdf")
    print(f"\n🎉 任务完成！生成的PDF路径: {final_pdf}")
    return final_pdf
# ---------- 主流程 ----------
# def run(user_query: str) -> str:
#     messages = [
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": user_query}
#     ]
#     order = ["llm_agent", "rag_agent", "chart_agent", "sd_agent", "pdf_agent"]
#     results = {}
#     max_step = 5
#     step_cnt = 0          # 已真正执行的步数
#     count = 0
#     for func in order:
#         if step_cnt >= max_step:
#             break
#         msg = call_with_retry(messages, tools, func)
#         step_cnt += 1     # 真正调用成功才计数
#
#         call = msg.tool_calls[0]
#         args = json.loads(call.function.arguments)
#         args = replace_placeholder(args, results)
#         ret = functions[call.function.name](**args)
#         results[call.function.name] = ret
#         # ➜ 新增：立即打印
#         print(f">>>【{call.function.name}】真实返回：{ret}")
#         print(f"  ↳ 长度：{len(json.dumps(ret))} 字符")
#         for m in messages:
#             if isinstance(m, dict):
#                 print(f"消息内容长度: {len(m['content'])}")
#                 count = count + len(m['content'])
#             else:
#                 print(f"消息内容长度: {len(m.content)}")
#                 count = count + len(m.content)
#
#
#         messages.append(msg)
#         messages.append({
#             "role": "tool",
#             "tool_call_id": call.id,
#             "name": call.function.name,
#             "content": json.dumps(ret)
#         })
#         print(f"消息总长{count}")
#         # 在调用模型之后打印当前 messages 列表
#         print(f"调用 {func} 后 messages 列表:")
#         for m in messages:
#             print(m)
#     return results.get("pdf_agent", "")

# ---------- 占位符替换 ----------
import re



# def replace_placeholder(args: dict, upstream: dict) -> dict:
#     args = json.loads(json.dumps(args))
#     # ✅ 正确正则：匹配 <your_ 和 _agent_output> 之间的任意字符（非贪婪）
#     pattern = r"^<your_(.+)_agent_output>$"
#     for k, v in args.items():
#         if isinstance(v, str):
#             match = re.match(pattern, v)
#             if match:
#                 key = match.group(1)  # 提取中间部分
#                 print(f"🔍 发现占位符: {v} -> 查找 upstream[{key}]")
#                 if key in upstream:
#                     args[k] = upstream[key]
#                     print(f"✅ 替换成功: {v} -> {upstream[key]}")
#                 else:
#                     print(f"❌ {key} 不存在于 upstream 中！使用空字符串")
#                     args[k] = ""
#     return args
def replace_placeholder(args: dict, upstream: dict) -> dict:
    import re
    import json

    # 先深拷贝一份，避免污染原始输入
    args = json.loads(json.dumps(args))

    # 🔴 第一步：先处理新的简单占位符 <your_product>, <your_style>
    simple_pattern = r"<your_(product|style)>"
    for k, v in list(args.items()):
        if isinstance(v, str):
            matches = re.findall(simple_pattern, v)
            for match in matches:
                print(f"🔍 发现占位符: <your_{match}> -> 查找 upstream[{match}]")
                if match in upstream:
                    v = v.replace(f"<your_{match}>", str(upstream[match]))
                    args[k] = v
                    print(f"✅ 替换成功: <your_{match}> -> {upstream[match]}")
                else:
                    print(f"❌ {match} 不存在于 upstream 中！替换为空字符串")
                    args[k] = v.replace(f"<your_{match}>", "")

    # ✅ 第二步：完全保留你原有的逻辑，不做任何修改
    pattern = r"^<your_(.+)_agent_output>$"
    for k, v in args.items():
        if isinstance(v, str):
            match = re.match(pattern, v)
            if match:
                key = match.group(1)  # 提取中间部分
                print(f"🔍 发现占位符: {v} -> 查找 upstream[{key}]")
                if key in upstream:
                    args[k] = upstream[key]
                    print(f"✅ 替换成功: {v} -> {upstream[key]}")
                else:
                    print(f"❌ {key} 不存在于 upstream 中！使用空字符串")
                    args[k] = ""

    return args