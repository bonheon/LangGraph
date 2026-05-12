# workflow_notebook_toolcalling.py
# workflow_notebook_toolcalling.ipynb 의 모든 코드 셀을 하나로 합친 파일
#
# [Part 1] Tool Calling 방식 (LLM이 직접 도구 선택)
#   run_tool_calling(query) → LLM → @tool 실행 → 결과
#
# [Part 2] LangGraph 방식 (classify → fetch → retrieve → generate_response)
#   classify (regex/API로 의도 판별)
#     → lot/eqp/fab/oper : fetch → retrieve(skip) → generate_response
#     → general          : retrieve(RAG)           → generate_response


# ── 1. 임포트 ────────────────────────────────────────────────────────────────
import re
import json
import os
import functools
from typing import List, Optional, Literal, Dict, Any, Annotated, TypedDict
from collections import defaultdict
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END


# ── 2. API 키 & LLM 초기화 ──────────────────────────────────────────────────
_base = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(_base, "..", ".env"))
if not os.getenv("OPENAI_API_KEY"):
    load_dotenv(dotenv_path=os.path.join(_base, ".env"))

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError(
        "OPENAI_API_KEY가 설정되지 않았습니다.\n"
        "프로젝트 루트의 .env 파일에 OPENAI_API_KEY=sk-... 를 입력하세요."
    )

# ▶ 사내 Private LLM 사용 시:
# from gaia_agent_manager.core import config
# llm = ChatOpenAI(model=config.PRIVATE_LLM_MODEL_NAME,
#                  base_url=config.PRIVATE_LLM_ENDPOINT,
#                  api_key=config.PRIVATE_LLM_API_KEY, temperature=0)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)


# ── 3. 샘플 데이터 & Mock API ─────────────────────────────────────────────────
SAMPLE_EQP_M15 = [
    {"EQP_ID": "M15A001", "FAC_ID": "M15", "DET_FAC_ID": "M15A",
     "EQP_MODEL_CD": "LPCVD_A", "EQ_GROUP": "CVD", "VENDOR_NM": "AMAT",
     "MGMT_AREA_ID": "CVD", "SECTION_GRP_NM": "M15NAND",
     "MES_STAT_TYP": "Up", "EQP_STAT_CD": "RUN",
     "BAY_NM": "BAY-01", "LAST_EVENT_TM": "2026-05-07 08:00:00",
     "PORT_INFO": "P1l진행중lRun,P2l대기lIdle"},
    {"EQP_ID": "M15A002", "FAC_ID": "M15", "DET_FAC_ID": "M15A",
     "EQP_MODEL_CD": "LPCVD_A", "EQ_GROUP": "CVD", "VENDOR_NM": "AMAT",
     "MGMT_AREA_ID": "CVD", "SECTION_GRP_NM": "M15NAND",
     "MES_STAT_TYP": "Down", "EQP_STAT_CD": "PM",
     "BAY_NM": "BAY-01", "LAST_EVENT_TM": "2026-05-06 14:30:00", "PORT_INFO": ""},
    {"EQP_ID": "M15B001", "FAC_ID": "M15", "DET_FAC_ID": "M15B",
     "EQP_MODEL_CD": "CMP_B", "EQ_GROUP": "CMP", "VENDOR_NM": "KLA",
     "MGMT_AREA_ID": "CMP", "SECTION_GRP_NM": "M15NAND",
     "MES_STAT_TYP": "Up", "EQP_STAT_CD": "RUN",
     "BAY_NM": "BAY-02", "LAST_EVENT_TM": "2026-05-07 09:15:00", "PORT_INFO": ""},
    {"EQP_ID": "M15A001_CH1", "FAC_ID": "M15", "DET_FAC_ID": "M15A",
     "EQP_MODEL_CD": "LPCVD_A", "EQ_GROUP": "CVD", "VENDOR_NM": "AMAT",
     "MGMT_AREA_ID": "CVD", "SECTION_GRP_NM": "M15NAND",
     "MES_STAT_TYP": "Up", "EQP_STAT_CD": "RUN",
     "BAY_NM": "BAY-01", "LAST_EVENT_TM": "2026-05-07 08:00:00", "PORT_INFO": ""},
    {"EQP_ID": "M15A001_CH2", "FAC_ID": "M15", "DET_FAC_ID": "M15A",
     "EQP_MODEL_CD": "LPCVD_A", "EQ_GROUP": "CVD", "VENDOR_NM": "AMAT",
     "MGMT_AREA_ID": "CVD", "SECTION_GRP_NM": "M15NAND",
     "MES_STAT_TYP": "Down", "EQP_STAT_CD": "FAULT",
     "BAY_NM": "BAY-01", "LAST_EVENT_TM": "2026-05-06 20:00:00", "PORT_INFO": ""},
]
SAMPLE_EQP_M16 = [
    {"EQP_ID": "M16C001", "FAC_ID": "M16", "DET_FAC_ID": "M16C",
     "EQP_MODEL_CD": "ALD_C", "EQ_GROUP": "ALD", "VENDOR_NM": "TEL",
     "MGMT_AREA_ID": "ALD", "SECTION_GRP_NM": "M16DRAM",
     "MES_STAT_TYP": "Up", "EQP_STAT_CD": "RUN",
     "BAY_NM": "BAY-03", "LAST_EVENT_TM": "2026-05-07 07:00:00", "PORT_INFO": ""},
    {"EQP_ID": "M16C002", "FAC_ID": "M16", "DET_FAC_ID": "M16C",
     "EQP_MODEL_CD": "ALD_C", "EQ_GROUP": "ALD", "VENDOR_NM": "TEL",
     "MGMT_AREA_ID": "ALD", "SECTION_GRP_NM": "M16DRAM",
     "MES_STAT_TYP": "Down", "EQP_STAT_CD": "PM",
     "BAY_NM": "BAY-03", "LAST_EVENT_TM": "2026-05-06 18:00:00", "PORT_INFO": ""},
]
SAMPLE_LOT_DATA = {
    "AB123456": [{"LOT_ID": "AB123456", "PROD_ID": "NAND_128G", "STATUS": "ACTIVE"}],
    "CD789012": [{"LOT_ID": "CD789012", "PROD_ID": "DRAM_16G",  "STATUS": "ACTIVE"}],
}
SAMPLE_SETMO = [
    {"LOT_ID": "AB123456", "ACT_NM": "SetMonitor",
     "OPER_DESC": "CVD 증착", "ACT_DESC": "두께 측정 필요",
     "MEAS_SLOT_NM": "Slot 1,3,5", "ENGR_USER_NM": "김엔지니어"},
    {"LOT_ID": "AB123456", "ACT_NM": "SetMonitor",
     "OPER_DESC": "CMP 연마", "ACT_DESC": "평탄도 확인",
     "MEAS_SLOT_NM": "Slot 2,4", "ENGR_USER_NM": "이엔지니어"},
    {"LOT_ID": "AB123456", "ACT_NM": "makeOnHold",
     "OPER_DESC": "식각 공정", "ACT_DESC": "이상 감지 시 홀드",
     "MEAS_SLOT_NM": "-", "ENGR_USER_NM": "박엔지니어"},
]
SAMPLE_SLOT = [
    {"POSITION_VAL": 1, "WF_ID": "WF001", "FAB": "M15", "OPER_DESC": "CVD 증착",  "EVENT_TM": "2026-05-07 08:00:00"},
    {"POSITION_VAL": 2, "WF_ID": "WF002", "FAB": "M15", "OPER_DESC": "CVD 증착",  "EVENT_TM": "2026-05-07 08:01:00"},
    {"POSITION_VAL": 3, "WF_ID": "WF003", "FAB": "M15", "OPER_DESC": "CMP 연마",  "EVENT_TM": "2026-05-07 08:02:00"},
    {"POSITION_VAL": 5, "WF_ID": "WF005", "FAB": "M15", "OPER_DESC": "검사",      "EVENT_TM": "2026-05-07 08:03:00"},
]
SAMPLE_LOTHIS = [
    {"TIMEKEY": "20260507090000", "EVENT_CD": "TRACK_IN",  "OPER_ID": "CVD001",   "CTN_DESC": "CVD 증착",        "WF_QTY": 25, "PROD_ID": "NAND_128G"},
    {"TIMEKEY": "20260507080000", "EVENT_CD": "TRACK_OUT", "OPER_ID": "PHOTO001", "CTN_DESC": "포토 리소그래피", "WF_QTY": 25, "PROD_ID": "NAND_128G"},
    {"TIMEKEY": "20260507060000", "EVENT_CD": "TRACK_IN",  "OPER_ID": "PHOTO001", "CTN_DESC": "포토 리소그래피", "WF_QTY": 25, "PROD_ID": "NAND_128G"},
    {"TIMEKEY": "20260506200000", "EVENT_CD": "TRACK_OUT", "OPER_ID": "CMP001",   "CTN_DESC": "CMP 연마",        "WF_QTY": 25, "PROD_ID": "NAND_128G"},
]
SAMPLE_OPERHIS = [
    {"LOT_ID": "AB1001", "OPERATIONDESC": "CVD 증착",        "OPERLEVEL": 300, "WF_QTY": 25, "CTN_DESC": "CVD 증착",        "MES_PROC_STAT_CD": "COMPLETE", "LAST_EVENT_TM": "2026-05-07 08:00", "FLOW_ID": "FLOW_A"},
    {"LOT_ID": "AB1002", "OPERATIONDESC": "CVD 증착",        "OPERLEVEL": 290, "WF_QTY": 24, "CTN_DESC": "CVD 증착",        "MES_PROC_STAT_CD": "COMPLETE", "LAST_EVENT_TM": "2026-05-06 20:00", "FLOW_ID": "FLOW_A"},
    {"LOT_ID": "AB1001", "OPERATIONDESC": "포토 리소그래피", "OPERLEVEL": 280, "WF_QTY": 25, "CTN_DESC": "포토 리소그래피", "MES_PROC_STAT_CD": "COMPLETE", "LAST_EVENT_TM": "2026-05-07 06:00", "FLOW_ID": "FLOW_A"},
    {"LOT_ID": "AB1001", "OPERATIONDESC": "CMP 연마",        "OPERLEVEL": 270, "WF_QTY": 25, "CTN_DESC": "CMP 연마",        "MES_PROC_STAT_CD": "COMPLETE", "LAST_EVENT_TM": "2026-05-06 18:00", "FLOW_ID": "FLOW_A"},
]

def call_lotid(lot_id):        return SAMPLE_LOT_DATA.get(lot_id.upper(), [])
def call_setmo_check(lot_id):  return [d for d in SAMPLE_SETMO if d["LOT_ID"] == lot_id.upper()]
def call_slot_info(lot_id):    return SAMPLE_SLOT
def call_lothis(lot_id):       return SAMPLE_LOTHIS
def call_eq(eqp_id):           return [d for d in SAMPLE_EQP_M15 + SAMPLE_EQP_M16 if d["EQP_ID"] == eqp_id.upper()]
def call_eqpm15(eqp_id="*"):  return SAMPLE_EQP_M15 if eqp_id == "*" else [d for d in SAMPLE_EQP_M15 if d["EQP_ID"] == eqp_id]
def call_eqpm16(eqp_id="*"):  return SAMPLE_EQP_M16 if eqp_id == "*" else [d for d in SAMPLE_EQP_M16 if d["EQP_ID"] == eqp_id]
def call_eqpm10(eqp_id="*"):  return []
def call_eqpm11(eqp_id="*"):  return []
def call_eqpm14(eqp_id="*"):  return []
def call_operhis_api(lot_cd):  return SAMPLE_OPERHIS


# ── 4. GraphState (Pydantic) & 공통 헬퍼 함수 ───────────────────────────────
class GraphState(BaseModel):
    query: str = ""
    answer: Optional[str] = None
    intent: Optional[str] = None
    is_eqp: bool = False
    is_lot: bool = False
    eqp_id: Optional[str] = None
    lot_id: Optional[str] = None
    fab: Optional[str] = None
    skip_rag: bool = False
    tool_calls_log: List[str] = Field(default_factory=list)
    class Config:
        arbitrary_types_allowed = True


@functools.lru_cache(maxsize=32)
def get_eqp_by_fab(fab_input: str, eqp_id: str = "*") -> tuple:
    fab_key = fab_input.upper().replace("X", "")
    api_map = {"M10": call_eqpm10, "M11": call_eqpm11, "M14": call_eqpm14,
               "M15": call_eqpm15, "M16": call_eqpm16}
    result = api_map.get(fab_key, lambda x: [])(eqp_id) or []
    return tuple(json.dumps(d) for d in result)

def get_eqp_list(fab_input: str, eqp_id: str = "*") -> List[Dict]:
    return [json.loads(s) for s in get_eqp_by_fab(fab_input, eqp_id)]


def format_setmo_data(data, lot_id, filter_act=None):
    if not data: return f"{lot_id} SETMO 정보 없음"
    if filter_act: data = [d for d in data if d.get("ACT_NM") == filter_act]
    if not data: return f"{lot_id} 해당 SETMO 정보 없음"
    if filter_act == "makeOnHold":
        lines = [f"##{lot_id} - Future Hold\n", "| 공정 | Comment | 엔지니어 |", "| --- | --- | --- |"]
        for d in data[:20]:
            lines.append(f"| {d.get('OPER_DESC','-')} | {d.get('ACT_DESC','-')} | {d.get('ENGR_USER_NM','-')} |")
    else:
        lines = [f"##{lot_id} - SetMonitor\n", "| 공정 | Comment | 측정 Slot | 엔지니어 |", "| --- | --- | --- | --- |"]
        for d in data[:20]:
            lines.append(f"| {d.get('OPER_DESC','-')} | {d.get('ACT_DESC','-')} | {d.get('MEAS_SLOT_NM','-')} | {d.get('ENGR_USER_NM','-')} |")
    return "\n".join(lines)

def format_slot_data(data, lot_id):
    if not data: return f"{lot_id} 슬롯 정보 없음"
    latest = {}
    for d in data:
        p = d.get("POSITION_VAL")
        if p is not None:
            p = int(p)
            if p not in latest or d.get("EVENT_TM", "") > latest[p].get("EVENT_TM", ""):
                latest[p] = d
    lines = [f"##{lot_id} - Slot 정보\n", "| Slot | WF_ID | 공정 (FAB) | 시간 |", "| --- | --- | --- | --- |"]
    for i in range(1, 26):
        if i in latest:
            d = latest[i]
            lines.append(f"| {i} | {d.get('WF_ID','-')} | {d.get('OPER_DESC','-')} ({d.get('FAB','-')}) | {str(d.get('EVENT_TM','-'))[:19]} |")
        else:
            lines.append(f"| {i} | *Empty* | - | - |")
    return "\n".join(lines)

def format_lothis_data(data, lot_id):
    if not data: return f"{lot_id} 이력 없음"
    sorted_data = sorted(data, key=lambda x: x.get("TIMEKEY", ""), reverse=True)[:10]
    lines = [f"##{lot_id} - 이력정보\n", "| 순번 | 시간 | 이벤트 | 공정 ID | 공정명 | 수량 | 제품 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for i, d in enumerate(sorted_data, 1):
        t = str(d.get("TIMEKEY", "-"))
        if len(t) >= 14 and t.isdigit():
            t = f"{t[:4]}-{t[4:6]}-{t[6:8]} {t[8:10]}:{t[10:12]}:{t[12:14]}"
        lines.append(f"| {i} | {t} | {d.get('EVENT_CD','-')} | {d.get('OPER_ID','-')} | {d.get('CTN_DESC','-')} | {d.get('WF_QTY','-')} | {d.get('PROD_ID','-')} |")
    return "\n".join(lines)

def analyze_eqp_data(data, eqp_id):
    if not data: return {}
    main_eqp = next((d for d in data if d.get("EQP_ID") == eqp_id), data[0])
    chambers = [d for d in data if "_" in d.get("EQP_ID", "") or
                (d.get("EQP_ID", "").startswith(eqp_id) and d.get("EQP_ID") != eqp_id)]
    down_ch = [c for c in chambers if c.get("MES_STAT_TYP") == "Down"]
    port_list = []
    for seg in (main_eqp.get("PORT_INFO", "") or "").split(","):
        parts = seg.strip().split("l")
        if len(parts) >= 2:
            port_list.append({"port": parts[0],
                               "transfer": parts[1] if len(parts) > 1 else "-",
                               "status":   parts[2] if len(parts) > 2 else "-"})
    return {"EQP_ID": main_eqp.get("EQP_ID", eqp_id), "EQ_GROUP": main_eqp.get("EQ_GROUP", "-"),
            "MES_STAT_TYP": main_eqp.get("MES_STAT_TYP", "-"), "EQP_STAT_CD": main_eqp.get("EQP_STAT_CD", "-"),
            "FAB": main_eqp.get("FAC_ID", "-"), "장비사": main_eqp.get("VENDOR_NM", "-"),
            "모델": main_eqp.get("EQP_MODEL_CD", "-"),
            "CHAMBER 수": len(chambers), "DOWN 수": len(down_ch),
            "chamber_details": chambers, "port_list": port_list}

def format_eqp_table(analyzed, show_chamber=True, show_port=True):
    lines = ["## 장비 상태\n", "| 항목 | 값 |", "| --- | --- |"]
    for k in ["EQP_ID", "EQ_GROUP", "MES_STAT_TYP", "EQP_STAT_CD", "CHAMBER 수", "DOWN 수"]:
        lines.append(f"| {k} | {analyzed.get(k, '-')} |")
    if show_chamber and analyzed.get("chamber_details"):
        lines += ["\n| 챔버 ID | 상태 | EQP_STAT | LAST_EVENT |", "| --- | --- | --- | --- |"]
        for ch in sorted(analyzed["chamber_details"], key=lambda x: x.get("EQP_ID", "")):
            lines.append(f"| {ch.get('EQP_ID','-')} | {ch.get('MES_STAT_TYP','-')} | {ch.get('EQP_STAT_CD','-')} | {ch.get('LAST_EVENT_TM','-')} |")
    if show_port and analyzed.get("port_list"):
        lines += ["\n### PORT 정보", "| PORT | Transfer | 상태 |", "| --- | --- | --- |"]
        for p in analyzed["port_list"]:
            lines.append(f"| {p['port']} | {p['transfer']} | {p['status']} |")
    return "\n".join(lines)

def format_fab_summary(data, fab, query):
    main = [e for e in data if "_" not in e.get("EQP_ID", "") and
            not any(s in e.get("EQP_ID", "") for s in ["CH", "SPIN"])]
    ch = [e for e in data if e not in main]
    models: Dict[str, Dict] = defaultdict(lambda: {"TOTAL": 0, "UP": 0, "DOWN": 0})
    for e in main:
        m = e.get("EQP_MODEL_CD", "Unknown")
        models[m]["TOTAL"] += 1
        models[m]["UP" if e.get("MES_STAT_TYP") == "Up" else "DOWN"] += 1
    lines = [f"## {fab} 장비 현황 ({query})\n",
             f"- 메인 장비 총: {len(main)}대 / 챔버: {len(ch)}개", "",
             "| MODEL | TOTAL | UP | DOWN |", "| --- | --- | --- | --- |"]
    for k, v in sorted(models.items()):
        lines.append(f"| {k} | {v['TOTAL']} | {v['UP']} | {v['DOWN']} |")
    return "\n".join(lines)

def format_operhis_data(data, ctn_desc):
    groups: Dict[str, list] = defaultdict(list)
    for d in data:
        if d.get("LOT_ID"): groups[d["LOT_ID"]].append(d)
    selected = sorted([max(v, key=lambda x: x.get("OPERLEVEL", 0)) for v in groups.values()],
                      key=lambda x: x.get("OPERLEVEL", 0), reverse=True)[:10]
    lines = [f"## {ctn_desc} 공정 이력\n",
             "| LOT_ID | WF_QTY | CTN_DESC | 상태 | 시간 | FLOW_ID |",
             "| --- | --- | --- | --- | --- | --- |"]
    for d in selected:
        lines.append(f"| {d.get('LOT_ID','-')} | {d.get('WF_QTY','-')} | {d.get('CTN_DESC','-')} | {d.get('MES_PROC_STAT_CD','-')} | {d.get('LAST_EVENT_TM','-')} | {d.get('FLOW_ID','-')} |")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — Tool Calling 방식
# ══════════════════════════════════════════════════════════════════════════════

# ── 5. @tool 정의 (8개) ──────────────────────────────────────────────────────
@tool
def get_lot_slot_info(lot_id: str) -> str:
    """LOT의 현재 슬롯별 위치와 공정 정보를 조회합니다.
    사용자가 LOT이 어디 있는지, 슬롯 정보, 현재 공정 위치를 물어볼 때 사용합니다.
    예: 'AB123456 슬롯 정보', 'AB123456 어디 있어', 'AB123456 현재 공정'
    """
    return format_slot_data(call_slot_info(lot_id), lot_id)


@tool
def get_lot_setmonitor(lot_id: str) -> str:
    """LOT의 Set Monitoring(셋모) 계측 지시 정보를 조회합니다.
    사용자가 셋모, setmo, 계측, S/M, set monitoring을 언급할 때 사용합니다.
    예: 'AB123456 셋모 확인', 'AB123456 setmonitor 있어?'
    """
    return format_setmo_data(call_setmo_check(lot_id), lot_id, "SetMonitor")


@tool
def get_lot_future_hold(lot_id: str) -> str:
    """LOT의 Future Hold(퓨처홀드) 정보를 조회합니다.
    사용자가 F/H, future hold, 퓨처홀드, makeOnHold를 언급할 때 사용합니다.
    예: 'AB123456 future hold 있어?', 'AB123456 F/H 확인'
    """
    return format_setmo_data(call_setmo_check(lot_id), lot_id, "makeOnHold")


@tool
def get_lot_history(lot_id: str) -> str:
    """LOT의 공정 이력(이동 이력)을 시간 순으로 조회합니다.
    사용자가 LOT 이력, 처리 이력, 어떤 공정을 거쳤는지 물어볼 때 사용합니다.
    예: 'AB123456 이력', 'AB123456 처리 이력 알려줘'
    """
    return format_lothis_data(call_lothis(lot_id), lot_id)


@tool
def get_equipment_status(eqp_id: str, fab: str) -> str:
    """특정 장비 ID의 현재 상태, 챔버 정보, PORT 정보를 조회합니다.
    사용자가 특정 장비 ID를 언급하며 상태, 다운여부, 정보를 물어볼 때 사용합니다.
    예: 'M15A001 상태', 'M15A001 장비 정보', 'M15A002 다운이야?'
    """
    all_data = get_eqp_list(fab, "*")
    eqp_group = [d for d in all_data if d["EQP_ID"].startswith(eqp_id)]
    if not eqp_group:
        return f"{eqp_id} 장비를 찾을 수 없습니다."
    analyzed = analyze_eqp_data(eqp_group, eqp_id)
    is_cmp = any("CMP" in str(d.get("MGMT_AREA_ID", "")).upper() for d in eqp_group)
    return format_eqp_table(analyzed, show_chamber=not is_cmp, show_port=True)


@tool
def get_fab_equipment_summary(fab: str, model: Optional[str] = None, area: Optional[str] = None) -> str:
    """FAB(공장) 전체 또는 특정 모델/구역의 장비 현황(대수, UP/DOWN)을 집계합니다.
    사용자가 FAB 번호(M10~M16)를 언급하며 장비 현황, 댓수, DOWN 현황을 물어볼 때 사용합니다.
    예: 'M15 장비 현황', 'M16 ALD 다운 몇 대야?', 'M15 LPCVD_A 상태'
    """
    data = get_eqp_list(fab, "*")
    if model:
        data = [d for d in data if d.get("EQP_MODEL_CD") == model]
    if area:
        data = [d for d in data if d.get("MGMT_AREA_ID") == area or d.get("EQ_GROUP") == area]
    if not data:
        return f"{fab} 조건에 맞는 장비 없음"
    return format_fab_summary(data, fab, f"{model or ''} {area or ''}")


@tool
def get_operation_history(lot_prefix: str, operation_name: str, is_previous: bool = False) -> str:
    """특정 공정(operation)을 거친 LOT들의 이력을 조회합니다.
    사용자가 특정 공정명을 언급하거나 이전 공정을 물어볼 때 사용합니다.
    예: 'AB1 CVD 증착 이력', 'AB1 포토 공정 이전 단계', 'CVD 증착 진행 LOT 현황'
    """
    raw = call_operhis_api(lot_prefix) or []
    target = operation_name.strip().upper()
    matched = [d for d in raw if str(d.get("OPERATIONDESC", "")).strip().upper() == target]
    if not matched:
        return f"'{operation_name}' 공정 데이터 없음"
    if is_previous:
        max_level = max(d.get("OPERLEVEL", 0) for d in matched)
        raw = [d for d in raw if d.get("OPERLEVEL", 0) < max_level and str(d.get("LOT_ID", "")).startswith(lot_prefix)]
        raw = sorted(raw, key=lambda x: x.get("OPERLEVEL", 0), reverse=True)[:10]
        return format_operhis_data(raw, f"{operation_name} 이전 공정")
    return format_operhis_data(matched, operation_name)


@tool
def answer_general_question(query: str) -> str:
    """위의 어떤 도구도 해당하지 않는 일반적인 질문에 사용합니다.
    공정 지식, 장비 원리, 트러블슈팅 방법 등 RAG 검색이 필요한 질문에 사용합니다.
    예: 'CVD 공정에서 두께 편차 원인은?', 'CMP 스크래치 발생 원인'
    """
    return f"[RAG 검색 필요] '{query}' → 지식 베이스 검색 후 LLM 답변 생성 단계로 진행"


TOOLS = [
    get_lot_slot_info,
    get_lot_setmonitor,
    get_lot_future_hold,
    get_lot_history,
    get_equipment_status,
    get_fab_equipment_summary,
    get_operation_history,
    answer_general_question,
]

llm_with_tools = llm.bind_tools(TOOLS)


# ── 6. Tool Calling 실행기 ────────────────────────────────────────────────────
TOOL_MAP: Dict[str, Any] = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = """당신은 반도체 FAB의 MES 시스템 AI 어시스턴트입니다.
사용자의 질문을 분석하여 적절한 도구를 선택하고 호출하세요.

도구 선택 원칙:
- LOT ID(영문+숫자 혼합, 예: AB123456)가 포함된 경우 → LOT 관련 도구
- 장비 ID(예: M15A001, M16C002)가 포함된 경우 → get_equipment_status
- FAB 번호(M10~M16)만 있고 장비 ID가 없는 경우 → get_fab_equipment_summary
- 특정 공정명과 LOT 접두사가 있는 경우 → get_operation_history
- 위 모두 해당 없으면 → answer_general_question
"""


def run_tool_calling(query: str, verbose: bool = True) -> GraphState:
    """Tool Calling 방식으로 질문을 처리하는 메인 함수"""
    state = GraphState(query=query)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    if verbose:
        print(f"[질문] {query}")
        print("-" * 60)

    ai_response = llm_with_tools.invoke(messages)

    if not ai_response.tool_calls:
        if verbose:
            print("[결과] 도구 없이 직접 답변:")
            print(ai_response.content)
        state.answer = ai_response.content
        state.intent = "direct_answer"
        return state

    for tc in ai_response.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]

        if verbose:
            print(f"[LLM 선택 도구] {tool_name}")
            print(f"[인자] {tool_args}")

        state.tool_calls_log.append(f"{tool_name}({tool_args})")
        state.intent = tool_name

        if tool_name not in TOOL_MAP:
            state.answer = f"알 수 없는 도구: {tool_name}"
            continue

        tool_result = TOOL_MAP[tool_name].invoke(tool_args)

        if verbose:
            print(f"[실행 결과 미리보기]")
            print(tool_result[:200] + "..." if len(tool_result) > 200 else tool_result)

        state.answer = tool_result
        state.skip_rag = (tool_name != "answer_general_question")

    return state


# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — LangGraph 방식
# ══════════════════════════════════════════════════════════════════════════════

# ── 7. WorkflowState (TypedDict) & 파서 헬퍼 ────────────────────────────────
class WorkflowState(TypedDict, total=False):
    query: str
    answer: str
    group: str          # lot / eqp / fab / oper / general
    intent: str         # 그룹 내 세부 의도
    # LOT
    is_lot: bool
    lot_id: str
    # EQP
    is_eqp: bool
    eqp_id: str
    eqp_data: list
    fab: str
    # FAB 필터
    fab_model: str
    fab_area: str
    # OPER 파라미터
    parsed_ctn_desc: str
    parsed_lot_cd: str
    parsed_is_prev: bool
    # RAG 제어
    skip_rag: bool
    retrieved_docs: list


def extract_candidate(message: str) -> Optional[str]:
    patterns = [
        r'\b([A-Z]{2}\d{6,})\b',
        r'\b([A-Z]{2}\d{4})\b',
        r'\b([A-Z]{3,}\d{2,})\b',
    ]
    for p in patterns:
        m = re.search(p, message.upper())
        if m:
            return m.group(1)
    return None

def extract_fab(message: str) -> Optional[str]:
    m = re.search(r'\b(M1[0-6])\b', message.upper())
    return m.group(1) if m else None

def parse_fab_model_query(query: str) -> Optional[dict]:
    fab = extract_fab(query)
    if not fab:
        return None
    fab_data = get_eqp_list(fab)
    if not fab_data:
        return None
    all_models = {e.get("EQP_MODEL_CD", "") for e in fab_data if e.get("EQP_MODEL_CD")}
    all_groups = {e.get("EQ_GROUP", "")     for e in fab_data if e.get("EQ_GROUP")}
    words = sorted(set(re.findall(r'\b[A-Z][A-Z0-9_]*\b', query.upper())), key=len, reverse=True)
    model, area = None, None
    for w in words:
        if w == fab: continue
        if not model and w in all_models: model = w
        if not area  and w in all_groups: area = w
    if "CMP"   in query.upper(): area = "CMP"
    if "CLEAN" in query.upper(): area = "CLEAN"
    return {"fab": fab, "model": model, "area": area}

def parse_operation_query(query: str) -> Optional[dict]:
    if not query: return None
    query_upper = query.upper().strip()
    m = re.match(r"^([A-Z0-9]{3})", query_upper)
    lot_cd = m.group(1) if m else None
    operhis = call_operhis_api(lot_cd or "")
    ctn_set = {d.get("OPERATIONDESC", "") for d in operhis if d.get("OPERATIONDESC")}
    matched = [c for c in sorted(ctn_set, key=len, reverse=True) if c and c in query_upper]
    if not matched: return None
    is_prev = any(k in query.lower() for k in ["이전", "전에", "before", "prior", "앞", "전 공정"])
    return {"lot_cd": lot_cd, "ctn_desc": matched[0], "is_prev": is_prev}


# ── 8. classify 노드 ──────────────────────────────────────────────────────────
def classify(state: WorkflowState) -> dict:
    """LLM 없이 regex + API 호출로 그룹(lot/eqp/fab/oper/general) 판별"""
    query = state["query"]
    print(f"\n[classify] 질문: {query}")

    candidate = extract_candidate(query)

    if candidate and call_lotid(candidate):
        print(f"[classify] → lot ({candidate})")
        return {"is_lot": True, "lot_id": candidate, "group": "lot", "skip_rag": True}

    if candidate:
        eq_data = call_eq(candidate)
        if eq_data:
            fab = eq_data[0].get("FAC_ID", "M15")
            all_fab = get_eqp_list(fab)
            eqp_detail = [d for d in all_fab
                          if d["EQP_ID"] == candidate or d["EQP_ID"].startswith(candidate + "_")]
            print(f"[classify] → eqp ({candidate}, FAB={fab})")
            return {"is_eqp": True, "eqp_id": candidate, "eqp_data": eqp_detail,
                    "fab": fab, "group": "eqp", "skip_rag": True}

    fab_info = parse_fab_model_query(query)
    if fab_info:
        print(f"[classify] → fab {fab_info}")
        return {"fab": fab_info["fab"], "fab_model": fab_info.get("model"),
                "fab_area": fab_info.get("area"), "group": "fab", "skip_rag": True}

    oper_info = parse_operation_query(query)
    if oper_info:
        print(f"[classify] → oper {oper_info}")
        return {"parsed_lot_cd": oper_info["lot_cd"], "parsed_ctn_desc": oper_info["ctn_desc"],
                "parsed_is_prev": oper_info["is_prev"], "group": "oper", "skip_rag": True}

    print("[classify] → general (RAG)")
    return {"group": "general", "skip_rag": False}


def route_by_group(state: WorkflowState) -> str:
    return {"lot": "classify_lot", "eqp": "classify_eqp",
            "fab": "fetch_fab", "oper": "fetch_oper"}.get(state.get("group", ""), "retrieve")


# ── 9. LOT / EQP 세부 의도 분류 ──────────────────────────────────────────────
def classify_lot(state: WorkflowState) -> dict:
    """LOT 세부 의도 키워드 분류"""
    q = state["query"].lower()
    if   any(k in q for k in ["셋모", "setmo", "계측", "s/m", "set monitoring"]):
        intent = "setmonitoring"
    elif any(k in q for k in ["f/h", "퓨처홀드", "future hold", "makeonhold"]):
        intent = "future_action"
    elif any(k in q for k in ["슬롯", "slot", "위치", "공정", "어디", "상태", "정보"]):
        intent = "lot_info"
    else:
        intent = "lot_his"
    print(f"[classify_lot] intent={intent}")
    return {"intent": intent}

def route_lot_intent(state: WorkflowState) -> str:
    return {"setmonitoring": "fetch_lot_setmo", "future_action": "fetch_lot_fhold",
            "lot_info": "fetch_lot_slot", "lot_his": "fetch_lot_history"}.get(
            state.get("intent", ""), "fetch_lot_history")

def classify_eqp(state: WorkflowState) -> dict:
    """EQP 세부 의도 분류"""
    q = state["query"].lower()
    intent = "eqp_info" if any(k in q for k in ["정보", "info"]) else "eqp_status"
    print(f"[classify_eqp] intent={intent}")
    return {"intent": intent}

def route_eqp_intent(state: WorkflowState) -> str:
    return "fetch_eqp_info" if state.get("intent") == "eqp_info" else "fetch_eqp_status"


# ── 10. fetch 노드 (8개) ──────────────────────────────────────────────────────
def fetch_lot_setmo(state: WorkflowState) -> dict:
    return {"answer": format_setmo_data(call_setmo_check(state["lot_id"]), state["lot_id"], "SetMonitor")}

def fetch_lot_fhold(state: WorkflowState) -> dict:
    return {"answer": format_setmo_data(call_setmo_check(state["lot_id"]), state["lot_id"], "makeOnHold")}

def fetch_lot_slot(state: WorkflowState) -> dict:
    return {"answer": format_slot_data(call_slot_info(state["lot_id"]), state["lot_id"])}

def fetch_lot_history(state: WorkflowState) -> dict:
    return {"answer": format_lothis_data(call_lothis(state["lot_id"]), state["lot_id"])}

def fetch_eqp_info(state: WorkflowState) -> dict:
    analyzed = analyze_eqp_data(state["eqp_data"], state["eqp_id"])
    return {"answer": format_eqp_table(analyzed, show_chamber=True, show_port=True)}

def fetch_eqp_status(state: WorkflowState) -> dict:
    is_cmp = any("CMP" in str(d.get("MGMT_AREA_ID", "")).upper() for d in state["eqp_data"])
    analyzed = analyze_eqp_data(state["eqp_data"], state["eqp_id"])
    return {"answer": format_eqp_table(analyzed, show_chamber=not is_cmp, show_port=True)}

def fetch_fab(state: WorkflowState) -> dict:
    fab = state["fab"]
    data = get_eqp_list(fab)
    if state.get("fab_model"):
        data = [d for d in data if d.get("EQP_MODEL_CD") == state["fab_model"]]
    if state.get("fab_area"):
        data = [d for d in data if d.get("EQ_GROUP") == state["fab_area"]
                                or d.get("MGMT_AREA_ID") == state["fab_area"]]
    label = " ".join(filter(None, [state.get("fab_model"), state.get("fab_area")])) or "전체"
    return {"answer": format_fab_summary(data, fab, label) if data else f"{fab} 조건에 맞는 장비 없음"}

def fetch_oper(state: WorkflowState) -> dict:
    raw = call_operhis_api(state.get("parsed_lot_cd", "")) or []
    target = (state.get("parsed_ctn_desc") or "").strip().upper()
    matched = [d for d in raw if str(d.get("OPERATIONDESC", "")).strip().upper() == target]
    if not matched:
        return {"answer": f"'{state.get('parsed_ctn_desc')}' 공정 데이터 없음"}
    if state.get("parsed_is_prev"):
        max_lv = max(d.get("OPERLEVEL", 0) for d in matched)
        prev = sorted([d for d in raw if d.get("OPERLEVEL", 0) < max_lv],
                      key=lambda x: x.get("OPERLEVEL", 0), reverse=True)[:10]
        return {"answer": format_operhis_data(prev, f"{state['parsed_ctn_desc']} 이전 공정")}
    return {"answer": format_operhis_data(matched, state.get("parsed_ctn_desc", ""))}


# ── 11. FAISS 벡터스토어 ──────────────────────────────────────────────────────
SAMPLE_KNOWLEDGE_DOCS = [
    Document(page_content="CVD(Chemical Vapor Deposition) 공정에서 두께 편차의 주요 원인은 가스 유량 불균형, 챔버 온도 불균일, 압력 변동입니다. LPCVD 장비에서 ±2°C 이상 온도 편차 발생 시 두께 균일도에 직접 영향을 줍니다. 가스 유량 비율(SiH4/N2) 조정 및 히터 점검을 우선 권장합니다.", metadata={"source": "CVD공정_가이드", "category": "CVD"}),
    Document(page_content="CMP(Chemical Mechanical Planarization) 스크래치 발생 원인: 슬러리 입자 오염, 패드 컨디셔닝 불량, 과도한 가압, 드레서 마모. 조치: 슬러리 필터 점검, 패드 교체 주기 확인(통상 2000웨이퍼), 다운포스 압력 재조정.", metadata={"source": "CMP공정_가이드", "category": "CMP"}),
    Document(page_content="LPCVD 장비 PM(Preventive Maintenance) 주기는 500 사이클 또는 3개월입니다. PM 항목: 챔버 세정(HF wet clean), 히터 균일도 측정, 가스 라인 누기 점검, 쿼츠 보트 교체. PM 완료 후 더미 런(dummy run) 3회 이상 필수.", metadata={"source": "장비PM_매뉴얼", "category": "PM"}),
    Document(page_content="ALD(Atomic Layer Deposition) 막두께 균일도 불량 원인: 전구체 펄스 시간 오류, 퍼지 불완전으로 인한 CVD 모드 전환, 챔버 온도 불균일. TEL 장비에서는 전구체 공급 라인 온도(60~80°C) 유지가 핵심입니다.", metadata={"source": "ALD공정_가이드", "category": "ALD"}),
    Document(page_content="장비 FAULT 발생 시 초기 대응 절차: 1) 알람 코드 확인 후 MES 장비 상태 Down 변경, 2) 담당 엔지니어 즉시 연락, 3) 진행 중 LOT 상태 확인(Track-in 중이면 Track-out 처리), 4) 원인 분석 리포트 작성.", metadata={"source": "장비장애_대응", "category": "EQP"}),
    Document(page_content="포토(Photo) 리소그래피 패턴 불량 원인: 포커스 오류(Z축 편차), 노광량 이상(dose error), 레지스트 코팅 불균일, 현상 불량. CD(Critical Dimension) 편차 ±5% 초과 시 재작업(rework) 검토.", metadata={"source": "Photo공정_가이드", "category": "PHOTO"}),
    Document(page_content="식각(Etch) 공정 균일도 불량: 가스 유량 불균형, RF 파워 이상, 챔버 오염(폴리머 누적). 건식 식각(Dry Etch) 식각 속도 편차 기준은 5% 이내. 챔버 클리닝 주기(통상 50 런)와 엔드포인트 검출 정확도 확인 필수.", metadata={"source": "Etch공정_가이드", "category": "ETCH"}),
    Document(page_content="SetMonitor(셋모): 특정 공정 후 계측 지시를 자동 부여하는 기능. 계측 슬롯 지정 및 담당 엔지니어 설정 필요. 계측 결과 이상 시 자동 홀드(makeOnHold) 연계 가능.", metadata={"source": "MES_가이드", "category": "MES"}),
    Document(page_content="Future Hold(퓨처홀드/F/H): 특정 공정 도달 시 LOT을 자동 홀드로 전환. 설정: MES > LOT 관리 > makeOnHold. 홀드 사유, 대상 공정, 담당 엔지니어 필수 입력.", metadata={"source": "MES_가이드", "category": "MES"}),
    Document(page_content="M15 FAB NAND 공정: LPCVD(CVD 그룹) 및 CMP 장비 중심 구성. BAY-01 구역에 LPCVD_A 모델 집중 배치. 챔버 구성은 메인 장비 1대당 2~4개 챔버.", metadata={"source": "M15_운영가이드", "category": "FAB"}),
]

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
vectorstore = FAISS.from_documents(SAMPLE_KNOWLEDGE_DOCS, embedding_model)


# ── 12. retrieve & generate_response 노드 ────────────────────────────────────
SIMILARITY_THRESHOLD = 0.30

def retrieve(state: WorkflowState) -> dict:
    """skip_rag=True : 벡터 검색 스킵 / False : FAISS 검색 + 키워드 재랭킹"""
    if state.get("skip_rag", True):
        print("[retrieve] skip_rag=True → 스킵")
        return {"retrieved_docs": []}

    query = state["query"]
    docs_with_score = vectorstore.similarity_search_with_score(query, k=5)

    filtered = []
    for doc, score in docs_with_score:
        sim = max(0.0, 1 - score)
        if sim >= SIMILARITY_THRESHOLD:
            doc.metadata["similarity"] = round(sim, 4)
            filtered.append(doc)

    if filtered:
        stop_words = {"은", "는", "의", "가", "을", "를", "에", "와", "과", "로", "이"}
        keywords = [w for w in query.split() if len(w) > 1 and w not in stop_words]
        for doc in filtered:
            doc.metadata["keyword_score"] = sum(1 for k in keywords if k.lower() in doc.page_content.lower())
        max_kw = max(doc.metadata["keyword_score"] for doc in filtered) or 1
        for doc in filtered:
            norm = doc.metadata["keyword_score"] / max_kw
            doc.metadata["combined_score"] = doc.metadata["similarity"] * 0.7 + norm * 0.3
        filtered = sorted(filtered, key=lambda x: x.metadata["combined_score"], reverse=True)

    print(f"[retrieve] {len(docs_with_score)}건 검색 → {len(filtered)}건 채택")
    for doc in filtered:
        print(f"  [{doc.metadata['source']}] sim={doc.metadata.get('similarity'):.3f} combined={doc.metadata.get('combined_score', 0):.3f}")
    return {"retrieved_docs": filtered}


_RAG_PROMPT = """당신은 반도체 FAB MES 시스템 AI 어시스턴트입니다.
아래 정보를 바탕으로 질문에 답변하세요.

{context}"""

def generate_response(state: WorkflowState) -> dict:
    """구조적 데이터(answer) + retrieved_docs → LLM 최종 답변"""
    parts = []
    if state.get("answer"):
        parts.append(f"[구조적 데이터]\n{state['answer']}")
    if state.get("retrieved_docs"):
        rag_ctx = "\n\n".join(
            f"[출처: {d.metadata.get('source')}]\n{d.page_content}"
            for d in state["retrieved_docs"]
        )
        parts.append(f"[참고 문서]\n{rag_ctx}")

    context = "\n\n".join(parts) or "관련 데이터 없음"
    response = llm.invoke([
        SystemMessage(content=_RAG_PROMPT.format(context=context)),
        HumanMessage(content=state["query"]),
    ])
    src = f"참고문서 {len(state.get('retrieved_docs', []))}건" if state.get("retrieved_docs") else "구조적 데이터"
    print(f"[generate_response] 완료 ({src})")
    return {"answer": response.content}


# ── 13. LangGraph 그래프 조립 ─────────────────────────────────────────────────
def build_graph() -> StateGraph:
    wb = StateGraph(WorkflowState)

    wb.add_node("classify",          classify)
    wb.add_node("classify_lot",      classify_lot)
    wb.add_node("classify_eqp",      classify_eqp)
    wb.add_node("fetch_lot_setmo",   fetch_lot_setmo)
    wb.add_node("fetch_lot_fhold",   fetch_lot_fhold)
    wb.add_node("fetch_lot_slot",    fetch_lot_slot)
    wb.add_node("fetch_lot_history", fetch_lot_history)
    wb.add_node("fetch_eqp_info",    fetch_eqp_info)
    wb.add_node("fetch_eqp_status",  fetch_eqp_status)
    wb.add_node("fetch_fab",         fetch_fab)
    wb.add_node("fetch_oper",        fetch_oper)
    wb.add_node("retrieve",          retrieve)
    wb.add_node("generate_response", generate_response)

    wb.add_edge(START, "classify")

    wb.add_conditional_edges("classify", route_by_group, {
        "classify_lot": "classify_lot", "classify_eqp": "classify_eqp",
        "fetch_fab":    "fetch_fab",    "fetch_oper":   "fetch_oper",
        "retrieve":     "retrieve",
    })
    wb.add_conditional_edges("classify_lot", route_lot_intent, {
        "fetch_lot_setmo":  "fetch_lot_setmo",  "fetch_lot_fhold":   "fetch_lot_fhold",
        "fetch_lot_slot":   "fetch_lot_slot",   "fetch_lot_history": "fetch_lot_history",
    })
    wb.add_conditional_edges("classify_eqp", route_eqp_intent, {
        "fetch_eqp_info": "fetch_eqp_info", "fetch_eqp_status": "fetch_eqp_status",
    })

    for node in ["fetch_lot_setmo", "fetch_lot_fhold", "fetch_lot_slot", "fetch_lot_history",
                 "fetch_eqp_info", "fetch_eqp_status", "fetch_fab", "fetch_oper"]:
        wb.add_edge(node, "retrieve")

    wb.add_edge("retrieve",          "generate_response")
    wb.add_edge("generate_response", END)

    return wb.compile()


workflow_app = build_graph()


# ── 14. 실행 헬퍼 ─────────────────────────────────────────────────────────────
def run_workflow(query: str) -> dict:
    print(f"\n{'='*60}\n[질문] {query}")
    result = workflow_app.invoke({"query": query})
    print(f"\n[최종 답변]\n{result.get('answer', '(없음)')}")
    return result


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Part 1: Tool Calling 방식")
    print("="*60)
    run_tool_calling("AB123456 슬롯 정보 알려줘")
    print()
    run_tool_calling("AB123456 계측 지시 확인해줘")
    print()
    run_tool_calling("AB123456 퓨처홀드 걸려있어?")
    print()
    run_tool_calling("M15A001 지금 돌아가?")
    print()
    run_tool_calling("M15에서 CVD 장비 몇 대 다운이야?")
    print()
    run_tool_calling("AB1 CVD 증착 이전 공정 뭐야?")
    print()
    run_tool_calling("CVD 공정에서 두께 편차가 생기는 원인이 뭐야?")

    print("\n" + "="*60)
    print("  Part 2: LangGraph 방식")
    print("="*60)
    run_workflow("AB123456 슬롯 정보 알려줘")
    run_workflow("AB123456 셋모 잡혀있어?")
    run_workflow("M15A001 지금 돌아가?")
    run_workflow("M15 LPCVD_A 현황 알려줘")
    run_workflow("CVD 공정에서 두께 편차 원인이 뭐야?")
