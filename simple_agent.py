"""
간단한 LangGraph ReAct Agent 예시
- OpenAI를 LLM으로 사용
- 계산기, 날짜 조회 도구 포함
- 사용자 질문에 도구를 활용해 답변
"""

import os
from datetime import datetime
from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict


# ── 도구 정의 ──────────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """수식을 계산합니다. 예: '2 + 3 * 4', '100 / 5'"""
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"{expression} = {result}"
    except Exception as e:
        return f"계산 오류: {e}"


@tool
def get_current_date() -> str:
    """오늘 날짜와 현재 시각을 반환합니다."""
    now = datetime.now()
    return now.strftime("%Y년 %m월 %d일 %H:%M:%S")


# ── State 정의 ─────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── Agent 생성 ─────────────────────────────────────────────────

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError(
        "OPENAI_API_KEY 환경변수가 필요합니다.\n"
        "실행 전에: export OPENAI_API_KEY='sk-...'"
    )

tools = [calculator, get_current_date]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

tool_node = ToolNode(tools)


def call_llm(state: State) -> State:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: State) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ── Graph 구성 ─────────────────────────────────────────────────

graph = (
    StateGraph(State)
    .add_node("llm", call_llm)
    .add_node("tools", tool_node)
    .set_entry_point("llm")
    .add_conditional_edges("llm", should_continue)
    .add_edge("tools", "llm")
    .compile()
)


# ── 실행 ───────────────────────────────────────────────────────

def run(query: str):
    print(f"\n[사용자] {query}")
    print("-" * 40)

    result = graph.invoke({"messages": [{"role": "user", "content": query}]})

    for msg in result["messages"]:
        role = msg.__class__.__name__.replace("Message", "")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"[{role}] 도구 호출: {tc['name']}({tc['args']})")
        elif isinstance(msg, ToolMessage):
            print(f"[Tool:{msg.name}] {msg.content}")
        else:
            content = msg.content
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            if content:
                print(f"[{role}] {content}")

    print()


if __name__ == "__main__":
    run("오늘 날짜를 알려줘")
    run("(123 + 456) * 2 를 계산해줘")
    run("오늘이 몇 월이고, 그 달의 일수를 곱하면 얼마야?")
