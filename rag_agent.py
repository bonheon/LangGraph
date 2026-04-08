"""
RAG + Tool 통합 LangGraph Agent 예시

구조:
  사용자 질문
      │
   [LLM] ── 도구 선택
      │
  ┌───┴────────────────────┐
  │                        │
[rag_search]        [calculator]
(벡터DB 검색)       (수식 계산)
  │                        │
  └───────┬────────────────┘
       [LLM] ── 최종 답변
"""

import os
from datetime import datetime
from typing import Annotated

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
import faiss


# ── API 키 확인 ────────────────────────────────────────────────

if not os.environ.get("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY 환경변수가 필요합니다.")


# ── RAG 벡터 DB 구성 ───────────────────────────────────────────

# 예시 문서 (실제로는 PDF, 웹페이지 등을 로드)
SAMPLE_DOCS = [
    "LangGraph는 LLM 기반 애플리케이션을 상태 머신(State Machine)으로 구성하는 프레임워크다. "
    "노드(Node)와 엣지(Edge)로 워크플로우를 정의하며, 순환(cycle) 구조도 지원한다.",

    "RAG(Retrieval-Augmented Generation)는 LLM이 외부 지식을 검색해 답변을 생성하는 기법이다. "
    "벡터 DB에서 관련 문서를 검색한 뒤 LLM의 컨텍스트에 포함시켜 정확도를 높인다.",

    "FAISS는 Facebook AI Research에서 만든 고속 벡터 유사도 검색 라이브러리다. "
    "수백만 개의 벡터를 밀리초 단위로 검색할 수 있어 RAG 시스템에 자주 사용된다.",

    "LangChain은 LLM 애플리케이션 개발을 위한 파이썬/자바스크립트 프레임워크다. "
    "체인, 에이전트, 메모리 등의 추상화를 제공하며 LangGraph와 함께 사용된다.",

    "OpenAI의 text-embedding-ada-002 모델은 텍스트를 1536차원 벡터로 변환한다. "
    "의미적으로 유사한 텍스트는 벡터 공간에서 가까운 위치에 배치된다.",
]

print("벡터 DB 구성 중...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_texts(SAMPLE_DOCS, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
print("벡터 DB 준비 완료\n")


# ── 도구 정의 ──────────────────────────────────────────────────

@tool
def rag_search(query: str) -> str:
    """
    내부 지식 베이스에서 관련 정보를 검색합니다.
    LangGraph, RAG, LangChain, FAISS, 임베딩 등 AI 기술에 관한 질문에 사용하세요.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "관련 문서를 찾지 못했습니다."
    return "\n\n".join(f"[문서 {i+1}] {doc.page_content}" for i, doc in enumerate(docs))


@tool
def calculator(expression: str) -> str:
    """수식을 계산합니다. 예: '2 + 3 * 4', '1536 * 4'"""
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"{expression} = {result}"
    except Exception as e:
        return f"계산 오류: {e}"


@tool
def get_current_date() -> str:
    """오늘 날짜와 현재 시각을 반환합니다."""
    return datetime.now().strftime("%Y년 %m월 %d일 %H:%M:%S")


# ── Graph 구성 ─────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


tools = [rag_search, calculator, get_current_date]
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
    print(f"[사용자] {query}")
    print("-" * 50)

    result = graph.invoke({"messages": [{"role": "user", "content": query}]})

    for msg in result["messages"]:
        role = msg.__class__.__name__.replace("Message", "")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  → 도구 호출: {tc['name']}({tc['args']})")
        elif isinstance(msg, ToolMessage):
            preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            print(f"  ← [{msg.name}] {preview}")
        else:
            content = msg.content
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            if content:
                print(f"[AI] {content}")

    print()


if __name__ == "__main__":
    # RAG만 사용
    run("LangGraph가 뭔지 설명해줘")

    # Tool만 사용
    run("1536 곱하기 4는 얼마야?")

    # RAG + 계산기 복합 사용
    run("FAISS가 뭔지 알려주고, text-embedding-ada-002의 벡터 차원 수에 8을 곱하면 얼마야?")
