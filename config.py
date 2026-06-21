"""全局配置：DeepSeek LLM 工厂与常量。

DeepSeek 兼容 OpenAI 接口，故复用 langchain-openai 的 ChatOpenAI，
仅替换 base_url 与 model。
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# 预算上限（元）：超过则触发人工审批
BUDGET_LIMIT = 8000.0
# 反思循环最大次数
MAX_ITERATIONS = 3
# SQLite 检查点文件
SQLITE_PATH = "checkpoints.sqlite"


def make_llm(temperature: float = 0.3) -> ChatOpenAI:
    """构造 DeepSeek ChatOpenAI 实例。"""
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        api_key=DEEPSEEK_API_KEY,
        temperature=temperature,
    )
