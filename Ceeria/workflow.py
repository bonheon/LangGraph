import logging
import os
import re
import functools
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Literal, Dict, Any
from pydantic import Field
from datetime import datetime
from collections import defaultdict

#GaiA
from gaia_agent_manager.core import AgentManager, config
from gaia_agent_manager.core.constants import GAIA_STANDARD_OUTPUT_NODE
from gaia_agent_manager.core.schemas.agent_manager import GaiaGraphState, Doc
from langchain_milvus.vectorstores.milvus import Milvus
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pymilvus import connections
from langchain_core.messages import HumanMessage
from scr.datasource.knowhow.knowhow_adapter import get_knowhow_by_query
from scr.datasource.knowhow.knowhow_adapter import get_knowhow_by_query_with_score

#Datahub
from scr.datasource.datahub_client import (
    call_setmo_check,
    call_slot_info,
    call_eqpm10, call_eqpm11, call_eqpm14, call_eqpm15, call_eqpm16,
    call_lotid,
    call_eq,
    call_lothis,
    call_operhis as call_operhis_api
)

logger = logging.getLogger(__name__)

#state 정의

class GraphState(GaiaGraphState):
    foo: Optional[int] = Field(default=0)
    bar: Optional[str] = Field(default=None)
    chat_upload_retrieved_docs: Optional[List] = Field(default=[])
    retrieved_docs: Optional[List] = Field(default=[])

    intent: Optional[str] = Field(default=None)
    query_type: Optional[str] = Field(default=None)

    is_eqp: bool = False
    is_lot: bool = False
    eqp_id: Optional[str] = None
    eqp_data: Optional[List[dict]] = None
    lot_id: Optional[str] = None

    fab: Optional[str] = None
    section_grp_nm: Optional[str] = None
    fab_model: Optional[str] = None
    fab_proc_model: Optional[str] = None
    fab_vendor: Optional[str] = None
    fab_area: Optional[str] = None
    fab_keyword: Optional[List[str]] = Field(default=[])
    fab_group_by: Optional[str] = None
    fab_aggregate: Optional[Literal["count", "list"]] = "list"
    fab_chamber_only: bool = False

    parsed_ctn_desc: Optional[str] = None
    parsed_lot_cd: Optional[str] = None
    parsed_is_prev: bool = False
    sql_result: Optional[List[dict]] = None
    skip_rag: bool = False

#환경 인프라
VERSION = os.path.basename(os.path.dirname(os.path.abspath(__file__))).replace("_", ".")

manager = AgentManager(
    service_id=config.SERVICE_ID,
    author="2065728",
    workflow_name="example workflow",
    description="workflow manager를 사용한 work flow 구현 예시",
    state_schema=GraphState,
    workflow_version=VERSION,
    main_model=config.PRIVATE_LLM_MODEL_NAME,
)

client_setting = {
    "model": config.PRIVATE_LLM_MODEL_NAME,
    "base_url": config.PRIVATE_LLM_ENDPOINT,
    "api_key": config.PRIVATE_LLM_API_KEY,
}
llm = ChatOpenAI(**client_setting)

embedding_function = OpenAIEmbeddings(
    model=config.DOCU_EMBEDDING_MODEL_NAME,
    openai_api_base=config.DOCU_EMBEDDING_ENDPOINT,
    embedding_ctx_length=config.DOCU_EMBEDDING_CTX_LENGTH,
    openai_api_key=config.DOCU_EMBEDDING_API_KEY,
    tiktoken_enabled=False,
    check_embedding_ctx_length=False,
)

connection_args = {
    "url": config.DOCU_VECTOR_DB_URI,
    "db_name": config.DOCU_VECTOR_DB_NAME,
    "user": config.DOCU_VECTOR_DB_USER,
    "password": config.DOCU_VECTOR_DB_PASSWORD
}

milvus_vectorstore = Milvus(
    embedding_function=embedding_function,
    collection_name=config.DOCU_USER_COLLECTION_NAME,
    connection_args=connection_args,
    vector_field='dense_vector',
    text_field='text',
    enable_dynamic_field=True
)

#helper 함수

def extract_candidate(message: str) -> Optional[str]:
    #lot/장비 추출
    patterns = [
        r'\b([A-Z]{2}\d{6,})\b',
        r'\b([A-Z]\d{7,})\b',
        r'\b([A-Z]{2}\d{4})\b',
        r'\b([A-Z]+[A-Z_0-9]+)\b',
        r'\b(CMP[A-Z0-9]+)\b',
        r'\b(KCC[A-Z0-9]+)\b',
        r'\b(\d[A-Z0-9]{3,7})\b',
        r'\b([A-Z]{3,}\d{2,})\b',
        r'\b([A-Z0-9]{3,})호기',
    ]
    for p in patterns:
        match = re.search(p, message.upper())
        if match:
            candidate = match.group(1)
            candidate = candidate.replace('호기', '')
            print(f"[DEBUG] extract_candidate 매칭 : {candidate} (패턴 :{p})")
            return candidate
    print(f"[DEBUG] extract_candidate 매칭실패 {message}")
    return None

def extract_fab(message: str) -> Optional[str]:
    """FAB 추출"""
    match = re.search(r'\b(M1[0-6])\b', message.upper())
    return match.group(1) if match else None

def parse_fab_model_query(query: str) -> Optional[Dict[str, Any]]:
    """FAB+MODEL 파싱"""
    if not query or not query.strip():
        return None

    query_upper = query.upper().strip()

    fab = None
    section_grp_nm = None

    if "M15X" in query_upper:
        fab = "M15"
        section_grp_nm = "M15DRAM"

    elif "M15NAND" in query_upper:
        fab = "M15"
        section_grp_nm = "M15NAND"

    elif "M15" in query_upper:
        fab = "M15"
        section_grp_nm = "M15NAND"

    else:
        fab = extract_fab(query_upper)

    if not fab:
        return None

    try:
        fab_data = get_eqp_by_fab(fab, "*")
        if not fab_data:
            return None

    except:
        return None

    all_models = set(e.get("EQP_MODEL_CD", "") for e in fab_data if e.get("EQP_MODEL_CD"))
    all_groups = set(e.get("EQ_GROUP", "") for e in fab_data if e.get("EQ_GROUP"))
    all_vendors = set(e.get("VENDOR_NM", "") for e in fab_data if e.get("VENDOR_NM"))

    words = re.findall(r'\b[A-Z][A-Z0-9_]*\b', query_upper)
    words = sorted(set(words), key=len, reverse=True)

    target_model = None
    target_proc_model = None
    target_vendor = None

    for word in words:
        if word == fab:
            continue
        if not target_model and word in all_models:
            target_model = word
        elif not target_proc_model and word in all_groups:
            target_proc_model = word
        elif not target_vendor and word in all_vendors:
            target_vendor = word

    target_area = None
    if "CLEAN" in query_upper:
        target_area = "CLEAN"
    if "CMP" in query_upper:
        target_area = "CMP"

    has_chamber = any(k in query for k in ['CHAMBER', '챔버'])
    has_summary = any(k in query for k in ['현황', '총', '댓수', '개수', '상태', 'DOWN', '다운'])

    group_by = None
    if '장비기준' in query or '모델별' in query:
        group_by = "model"
    elif '공정기준' in query or 'EQ_GROUP' in query:
        group_by = "eq_group"

    matched = {fab, target_model, target_proc_model, target_vendor, target_area}
    keywords = [w for w in words if w not in matched]
    return {
        "fab": fab,
        "model": target_model,
        "proc_model": target_proc_model,
        "vendor": target_vendor,
        "area": target_area,
        "section_grp_nm": section_grp_nm,
        "status_filter": None,
        "keywords": keywords,
        "chamber_only": has_chamber,
        "group_by": group_by,
        "aggregate": "count" if has_summary else "list",
        "search_all_fab": False,
    } if (fab or keywords) else None

def classify_equipment(data: List[Dict]) -> tuple[List[Dict], List[Dict]]:
    '''장비 챔버분리'''

    main_eqp = []
    chamber_eqp = []

    for item in data:
        eqp_id = item.get("EQP_ID", "")
        is_chamber = "_" in eqp_id or any(suffix in eqp_id for suffix in ["CH", "SPIN"])

        if is_chamber:
            chamber_eqp.append(item)
        else:
            main_eqp.append(item)

    return main_eqp, chamber_eqp

def parse_operation_query(query: str) -> Optional[Dict[str, Any]]:

    print(query)
    if not query:
        return None

    query_upper = query.upper().strip()
    lot_cd = None
    match_lot = re.match(r"^([A-Z0-9]{3})", query_upper)

    if match_lot:
        lot_cd = match_lot.group(1)
        query_upper = query_upper[3:].strip()

    operhis_data = call_operhis_api(lot_cd)
    ctn_desc = set(item.get("OPERATIONDESC", "") for item in operhis_data if item.get("OPERATIONDESC"))

    matches = []
    for ctn in sorted(ctn_desc, key=len, reverse=True):
        if ctn and ctn in query_upper:
            matches.append(ctn)

    if not lot_cd and not ctn_desc:
        return None

    prev_keywords = ["전에", "이전", "before", "prior", "previous", "앞", "이전 공정", "이전단계", "전", "전 공정", "BEFORE", "PRIOR", "PREVIOUS"]
    is_prev = any(kw in query for kw in prev_keywords) or \
              any(kw in query.lower() for kw in prev_keywords)

    return {
        "lot_cd": lot_cd,
        "ctn_desc_candidate": matches[0] if matches else None,
        "is_prev": is_prev
    }

@functools.lru_cache(maxsize=32)
def get_eqp_by_fab(fab_input: str, eqp_id: str = "*", section_grp: str = None) -> List[Dict]:
    """
    FAB 기준 와일드 카드 조회
    """

    fab_key = fab_input.upper().replace('X', '')

    fab_api_map = {
        'M10': call_eqpm10,
        'M11': call_eqpm11,
        'M14': call_eqpm14,
        'M15': call_eqpm15,
        'M16': call_eqpm16,
    }

    if fab_key not in fab_api_map:
        print(f"[DEBUG] 유효하지 않은 FAB 입력 : {fab_input} (매핑된 키: {fab_key})")

    api = fab_api_map[fab_key]

    print(f"[DEBUG] {fab_key} API 호출 : {api} ('{eqp_id}')")
    raw_result = api(eqp_id) or []
    print(f"[DEBUG] {fab_key} API 원본 결과 : {len(raw_result)}건")

    if section_grp:
        filtered_result = [
            item for item in raw_result
            if item.get('SECTION_GRP_NM') == section_grp
        ]
        print(f"[DEBUG] SECTION_GRP_NM '{section_grp}' 필터링 후 : {len(filtered_result)}건")

        if filtered_result:
            print(f"[DEBUG] 필터링 샘플 EQP_ID: {[d.get('EQP_ID') for d in filtered_result[:5]]}")
            print(f"[DEBUG] 필터링 샘플 SECTION_GRP_NM : {filtered_result[0].get('SECTION_GRP_NM')}")

        return filtered_result

    return raw_result

def format_fab_aggregated_result(data: List[Dict], query_info: Dict) -> str:
    model_filter = " ".join([f["value"] for f in query_info.get("filters", [])]) or "전체"
    status_filter = query_info.get("status_filter")

    total = len(data)
    down_list = [d for d in data if d.get("MES_STAT_TYP") == "Down"]
    down_count = len(down_list)
    up_count = total - down_count

    fab_stats = {}
    for d in data:
        fab = d.get("FAC_ID") or d.get("SRC") or d.get("_FAB") or "Unknown"
        if fab not in fab_stats:
            fab_stats[fab] = {"total": 0, "down": 0, "up": 0}
        fab_stats[fab]["total"] += 1
        if d.get("MES_STAT_TYP") == "Down":
            fab_stats[fab]["down"] += 1
        else:
            fab_stats[fab]["up"] += 1

    sorted_fabs = sorted(fab_stats.items(), key=lambda x: x[1]["total"], reverse=True)

    lines = [f"## {model_filter} 전체현황\n"]
    lines.extend([
        f"",
        f"**총 {total}대** / Down {down_count}대 / 가동{up_count}대",
        f"",
        f"### FAB 별 집계",
        f"",
        f"| FAB | 총 | Down | 가동 |",
        f"| ------ | ----- | ------ | ------ |",
    ])

    for fab, stats in sorted_fabs:
        lines.append(f"| {fab} | {stats['total']} | {stats['down']} | {stats['up']} |")

    if down_list:
        lines.extend([
            f"",
            f"### Down 상세 ({down_count}대)",
            f"",
            f"| EQP_ID | FAB | 모델 | 상태 | 코드 |",
            f"| ------- | ---- | ----- | ----- | ------ |",
        ])
        for d in sorted(down_list, key=lambda x: x.get("FAC_ID") or x.get("SRC", ""))[:30]:
            fab = d.get("FAC_ID") or d.get("SRC") or d.get("_FAB", "-")
            lines.append(
                f"| {d.get('EQP_ID', '-')} | {fab} |"
                f" {d.get('EQP_MODEL_CD', '-')} | {d.get('MES_STAT_TYP', '-')} |"
                f" {d.get('EQP_STAT_CD', '-')} |"
            )
        if len(down_list) > 30:
            lines.append(f"\n*외 {len(down_list)-30}건 생략*")

    return '\n'.join(lines)

def format_setmo_data(data: List[dict], lot_id: str, filter_act: str = None) -> str:
    """SETMO 데이터 포멧팅"""

    if not data:
        return f"{lot_id} SETMO 정보 없음"

    if filter_act:
        data = [d for d in data if d.get('ACT_NM') == filter_act]

    if not data:
        return f"{lot_id} Setmonitoring 정보 없음"

    if filter_act == "SetMonitor":
        title = f"##{lot_id} - Setmonitor\n"
        lines = [title, ""]
        lines.append("| 공정 | Comment | 측정 Slot | 엔지니어 |")
        lines.append("| ----- | --------- | --------- | --------- |")
        for item in data[:20]:
            lines.append(f"| {item.get('OPER_DESC', '-')} | {item.get('ACT_DESC', '-')} | {item.get('MEAS_SLOT_NM', '-')} | {item.get('ENGR_USER_NM', '-')} |")

    elif filter_act == "makeOnHold":
        title = f"##{lot_id} - Future Hold \n"
        lines = [title, ""]
        lines.append("| 공정 | Comment | 엔지니어 |")
        lines.append("| ----- | -------- | ---------- |")
        for item in data[:20]:
            lines.append(f"| {item.get('OPER_DESC', '-')} | {item.get('ACT_DESC', '-')} | {item.get('ENGR_USER_NM', '-')} |")

    else:
        title = f"##{lot_id} - ALL \n"
        lines = [title, ""]
        lines.append("| 공정 | Comment | 측정 Slot | 엔지니어 |")
        lines.append("| ----- | --------- | --------- | --------- |")
        for item in data[:20]:
            lines.append(f"| {item.get('OPER_DESC', '-')} | {item.get('ACT_DESC', '-')} | {item.get('MEAS_SLOT_NM', '-')} | {item.get('ENGR_USER_NM', '-')} |")

    if len(data) > 20:
        lines.append(f"\n*외 {len(data)-20}건 생략*")

    return '\n'.join(lines)

def format_slot_data(data: List[Dict[str, Any]], lot_id: str) -> str:

    if not data:
        return f"{lot_id} 슬롯 정보 없음"

    latest_slots: Dict[int, Dict[str, Any]] = {}

    for d in data:
        pos_val = d.get("POSITION_VAL")
        if pos_val is None:
            continue

        try:
            pos_int = int(pos_val)
        except (ValueError, TypeError):
            continue

        event_tm = d.get('EVENT_TM', '')

        if pos_int not in latest_slots:
            latest_slots[pos_int] = d
        else:
            existing_tm = latest_slots[pos_int].get('EVENT_TM', "")
            if event_tm > existing_tm:
                latest_slots[pos_int] = d

    lines = [f"##{lot_id} - Slot 정보(최신 기준)\n", ""]
    lines.append(" | Slot | WF_ID | 공정 (FAB) | 시간 |")
    lines.append(" | :----: | :----- | :---------- | :-----: |")

    for i in range(1, 26):
        if i in latest_slots:
            item = latest_slots[i]
            wf_id = item.get('WF_ID', '-')
            fab = item.get('FAB', '-')
            oper_desc = item.get('OPER_DESC', '-')
            event_tm = item.get('EVENT_TM', '-')[:19] #날짜 부분만 잘라서 표현
            lines.append(f" | {i} | {wf_id} | {oper_desc} ({fab}) | {event_tm} |")
        else:
            lines.append(f" | {i} | *Empty* | - | - |")

    return '\n'.join(lines)

def format_lothis_data(data: List[Dict[str, Any]], lot_id: str) -> str:

    if not data:
        return f"{lot_id} 이력정보 없음"

    try:
        sorted_data = sorted(data, key=lambda x: x.get('TIMEKEY', ''), reverse=True)
    except Exception:
        sorted_data = data

    recent_5 = sorted_data[:10]
    lines = [f"##{lot_id} - 이력정보 (최근 10건)\n", ""]
    lines.append(" | 순번 | 시간(timekey) | 이벤트(event_cd) | 공정 ID (OPER_ID) | 공정명 (CTN_DESC) | 수량 (WF_QTY) | 제품 ID(PROD_ID) |")
    lines.append("| :----: | :-------------: | :--------------------: | :----------------------: | :--------------------------------: | :------------------------------: | :-----------------: |")

    for idx, item in enumerate(recent_5, start=1):

        raw_time = str(item.get('TIMEKEY', '-'))
        if len(raw_time) >= 14 and raw_time.isdigit():
            formatted_time = f"{raw_time[:4]}-{raw_time[4:6]}-{raw_time[6:8]} {raw_time[8:10]}:{raw_time[10:12]}:{raw_time[12:14]}"
        else:
            formatted_time = raw_time

        event_cd = item.get('EVENT_CD', '-') or '-'
        oper_id = item.get('OPER_ID', '-') or '-'
        ctn_desc = item.get('CTN_DESC', '-') or '-'

        wf_qty = item.get('WF_QTY')
        wf_qty_str = str(wf_qty) if wf_qty is not None else '-'
        prod_id = item.get('PROD_ID', '-') or '-'

        lines.append(f" | {idx} | {formatted_time} | {event_cd} | {oper_id} | {ctn_desc} | {wf_qty_str} | {prod_id} |")
    return '\n'.join(lines)

def get_all_fab_data(filters: List[Dict] = None, status_filter: str = None) -> List[Dict]:

    all_data = []
    for fab in ['M10', 'M11', 'M14', 'M15', 'M16']:
        try:
            data = get_eqp_by_fab(fab, "*")
            for d in data:
                if not d.get('FAC_ID') and not d.get('SRC'):
                    d['_FAB'] = fab
            all_data.extend(data)
            print(f"[DEBUG] {fab} : {len(data)}건")
        except Exception as e:
            print(f"[DEBUG] {fab} 조회 실패: {e}")

    print(f"[DEBUG] 전 FAB 합계 : {len(all_data)}건")

    result = all_data
    if filters:
        for f in filters:
            field, value = f["field"], f["value"]
            result = [d for d in result if value.upper() in str(d.get(field, "")).upper()]
    if status_filter:
        result = [d for d in result if d.get("MES_STAT_TYP") == status_filter]
    print(f"[DEBUG] 필터 후 : {len(result)}건")
    return result

def apply_filters(data: List[Dict], query_info: Dict) -> List[Dict]:
    if not data:
        return []

    filtered = data
    target_model = query_info.get("model")
    target_proc_model = query_info.get("proc_model")
    target_vendor = query_info.get("vendor")
    target_area = query_info.get("area")
    target_section = query_info.get("section_grp_nm")

    if target_model:
        filtered = [item for item in filtered if item.get("EQP_MODEL_CD") == target_model]

    if target_proc_model:
        filtered = [item for item in filtered if item.get("EQ_GROUP") == target_proc_model]

    if target_vendor:
        filtered = [item for item in filtered if item.get("VENDOR_NM") == target_vendor]

    if target_area:
        filtered = [item for item in filtered if item.get("MGMT_AREA_ID") == target_area]

    if target_section:
        filtered = [item for item in filtered if item.get("SECTION_GRP_NM") == target_section]
    return filtered

def analyze_eqp_data(data: List[Dict], eqp_id: str) -> Dict[str, Any]:
    if not data:
        return {}
    main_eqp = None
    chambers = []

    for d in data:
        current_id = d.get("EQP_ID", "")
        if current_id == eqp_id or (eqp_id in current_id and "_" not in current_id):
            main_eqp = d
        elif "_" in current_id or current_id.startswith(eqp_id):
            chambers.append(d)

    if not main_eqp and data:
        main_eqp = data[0]

    down_chambers = [c for c in chambers if c.get("MES_STAT_TYP") == "Down"]
    port_list = []
    port_info = main_eqp.get("PORT_INFO", "") if main_eqp else ""
    if port_info and port_info not in ("-", "", "N/A"):
        for segment in port_info.split(","):
            parts = segment.strip().split("l") ## 추후 확인 필요
            if len(parts) >= 2:
                port_list.append({
                    "port": parts[0],
                    "transfer": parts[1] if len(parts) > 1 else "-",
                    "status": parts[2] if len(parts) > 2 else "-"})

    return {
        "EQP_ID": main_eqp.get("EQP_ID", eqp_id) if main_eqp else eqp_id,
        "EQ_GROUP": main_eqp.get("EQ_GROUP", "-") if main_eqp else "-",
        "SDPT_NM": main_eqp.get("SDPT_NM", "-") if main_eqp else "-",
        "FAB": main_eqp.get("FAC_ID", "-") if main_eqp else "-",
        "DFAB": main_eqp.get("DET_FAC_ID", "-") if main_eqp else "-",
        "장비사": main_eqp.get("VENDOR_NM", "-") if main_eqp else "-",
        "모델": main_eqp.get("EQP_MODEL_CD", "-") if main_eqp else "-",
        "BAY": main_eqp.get("BAY_NM", "-") if main_eqp else "-",
        "MES_STAT_TYP": main_eqp.get("MES_STAT_TYP", "-") if main_eqp else "-",
        "EQP_STAT_CD": main_eqp.get("EQP_STAT_CD", "-") if main_eqp else "-",
        "CHAMBER 수": len(chambers),
        "DOWN 수": len(down_chambers),
        "chamber_details": chambers,
        "port_list": port_list
    }

def format_eqp_info_table(analyzed: Dict, show_chamber: bool = True, show_port: bool = True) -> str:
    lines = ["## 장비 상태 \n"]
    lines.append(" | 항목 | 값 |")
    lines.append(" | ------ | ----- |")
    lines.append(f" | EQP_ID | {analyzed.get('EQP_ID', '-')} |")
    lines.append(f" | EQ_GROUP | {analyzed.get('EQ_GROUP', '-')} |")
    lines.append(f" | 메인 상태 | {analyzed.get('MES_STAT_TYP', '-')} |")
    lines.append(f" | EQP 상태 | {analyzed.get('EQP_STAT_CD', '-')} |")
    lines.append(f" | CHAMBER 수 | {analyzed.get('CHAMBER 수', '0')} |")
    lines.append(f" | DOWN 수 | {analyzed.get('DOWN 수', '0')} |")

    if show_chamber and analyzed.get("chamber_details"):
        sorted_chamber = sorted(analyzed["chamber_details"], key=lambda x: (x.get('EQP_ID') or ''))

        lines.append("\n | 챔버 ID | 상태 | EQP_STAT | LAST_EVENT |")
        lines.append("| --------- | ------ | -------- | ----------------- |")

        for ch in sorted_chamber:
            lines.append(f"| {ch.get('EQP_ID', '-')} | {ch.get('MES_STAT_TYP', '-')} | {ch.get('EQP_STAT_CD', '-')} | {ch.get('LAST_EVENT_TM', '-')} |")

    if show_port and analyzed.get("port_list"):
        lines.extend(["\n### PORT 정보", ""])
        lines.append(" | PORT | Transfer 상태 | 현재 상태 |")
        lines.append("| ------ | ---------------- | --------- |")
        for p in analyzed["port_list"]:
            lines.append(f"| {p['port']} | {p['transfer']} | {p['status']} |")
    return '\n'.join(lines)


def format_eqp_status_table(analyzed: Dict, show_chamber: bool = True, show_port: bool = True) -> str:
    lines = ["## 장비 상태\n"]
    lines.append("| 항목 | 값 |")
    lines.append("| ----- | ----- |")
    lines.append(f"| EQP_ID | {analyzed.get('EQP_ID', '-')} |")
    lines.append(f"| EQ_GROUP | {analyzed.get('EQ_GROUP', '-')} |")
    lines.append(f"| 메인 상태 | {analyzed.get('MES_STAT_TYP', '-')} |")
    lines.append(f"| EQP 상태 | {analyzed.get('EQP_STAT_CD', '-')} |")
    lines.append(f"| CHAMBER 수 | {analyzed.get('CHAMBER 수', '-')} |")
    lines.append(f"| DOWN 수 | {analyzed.get('DOWN 수', '-')} |")

    if show_chamber and analyzed.get("chamber_details"):
        sorted_chambers = sorted(analyzed["chamber_details"], key=lambda x: x.get('EQP_ID') or '')

        lines.append("\n | 챔버 ID | 상태 | EQP_STAT | LAST_EVENT |")
        lines.append("| --------- | ------ | ------------ | ------------ |")

        for ch in sorted_chambers:
            lines.append(f"| {ch.get('EQP_ID', '-')} | {ch.get('MES_STAT_TYP', '-')} | {ch.get('EQP_STAT_CD', '-')} | {ch.get('LAST_EVENT_TM', '-')} |")

    if show_port and analyzed.get("port_list"):
        lines.extend(["\n### PORT 정보", ""])
        lines.append(" | PORT | Transfer 상태 | 현재 상태 |")
        lines.append("| ------ | ------------------ | ------------- |")
        for p in analyzed["port_list"]:
            lines.append(f"| {p['port']} | {p['transfer']} | {p['status']} |")

    return '\n'.join(lines)

def format_fab_model_result(data: List[Dict], query_info: Dict, query: str) -> str:
    from collections import defaultdict
    fab = query_info["fab"]
    chamber_only = query_info.get("chamber_only", False)
    group_by = query_info.get("group_by")
    specific_model = next((e.get("EQP_MODEL_CD") for e in data), None)
    all_models = set(e.get("EQP_MODEL_CD") for e in data)

    main_eqp = [e for e in data
                if "_" not in e.get("EQP_ID", "") and not any(suffix in e.get("EQP_ID", "") for suffix in ["CH", "SPIN"])]
    chamber_eqp = [e for e in data
                   if "_" in e.get("EQP_ID", "") or any(suffix in e.get("EQP_ID", "") for suffix in ["CH", "SPIN"])]

    if chamber_only:
        main_eqp = []

    if chamber_only and not group_by:
        all_chambers = chamber_eqp
        ch_total = len(all_chambers)
        ch_down = sum(1 for d in all_chambers if d.get("MES_STAT_TYP") == "Down")
        ch_up = ch_total - ch_down

        lines = [f"## {fab} {query} 현황 \n"]
        lines.extend([
            f"",
            f"### Chamber",
            f"- 총: {ch_total}개 / down: {ch_down}개 / 가동: {ch_up}개",])

        ch_down_list = [d for d in all_chambers if d.get("MES_STAT_TYP") == "Down"]
        if ch_down_list:
            lines.extend([
                f"",
                f"### Down CHAMBER 상세 ({len(ch_down_list)}개 중 상위 10개)",
                f"",
                f" | CHAMBER_ID | 메인장비 | 상태 | 코드 |",
                f" | ------------------ | ---------- | -------- | ---------- |",])
            sorted_chambers = sorted(ch_down_list, key=lambda x: x.get("EQP_ID", ""))
            for c in sorted_chambers[:10]:
                main_id = str(c.get("EQP_ID", "-")).split("_")[0]
                lines.append(
                    f" | {c.get('EQP_ID', '-')} | {main_id} |"
                    f" {c.get('MES_STAT_TYP', '-')} | {c.get('EQP_STAT_CD', '-')} |")

            if len(ch_down_list) > 10:
                lines.append(f"\n*외 {len(ch_down_list)-10}건 생략*")

        return '\n'.join(lines)

    if len(all_models) == 1 and specific_model and (main_eqp or chamber_only):
        lines = [f"##{query}\n"]

        if chamber_only and not main_eqp and chamber_eqp:
            chamber_total = len(chamber_eqp)
            chamber_up = sum(1 for e in chamber_eqp if e.get("MES_STAT_TYP") == "Up")
            chamber_down = chamber_total - chamber_up

            lines.extend([
                f"",
                f"### {specific_model} CHAMBER",
                f"",
                f" | CHAMBER | TOTAL | UP | DOWN |",
                f" | ------------- | ------------- | ------- | ------- |",
                f"| {specific_model} | {chamber_total} | {chamber_up} | {chamber_down} |",])
        else:
            main_total = len(main_eqp)
            main_up = sum(1 for e in main_eqp if e.get("MES_STAT_TYP") == "Up")
            main_down = main_total - main_up

            chamber_total = len(chamber_eqp)
            chamber_up = sum(1 for e in chamber_eqp if e.get("MES_STAT_TYP") == "Up")
            chamber_down = chamber_total - chamber_up

            lines.extend([
                f"",
                f"### {specific_model} DOWN 상태",
                f"",
                f" | MODEL | TOTAL | UP | DOWN |",
                f" | ------------- | ------------- | ------- | ------- |",
                f"| {specific_model} | {main_total} | {main_up} | {main_down} |",])

            if chamber_total > 0:
                lines.append(f" | {specific_model}(CHAMBER) | {chamber_total} | {chamber_up} | {chamber_down} |")

            if main_eqp:
                main_down_eqp = [e for e in main_eqp if e.get("MES_STAT_TYP") == "Down"][:10]
                if main_down_eqp:
                    lines.extend([
                        f"",
                        f"### MAIN 장비 DOWN (상위 10개)",
                        f"",
                        f" | EQP_ID | 상태 | 마지막 이벤트 |",
                        f" | ----------- | ------- | ------------------ |",
                    ])
                    for e in main_down_eqp:
                        lines.append(f"| {e.get('EQP_ID')} | {e.get('EQP_STAT_CD')} | {e.get('LAST_EVENT_TM', '')} |")

        return '\n'.join(lines)

    else:
        lines = [f"##{query} \n"]
        models = defaultdict(lambda: {"TOTAL": 0, "UP": 0, "DOWN": 0})
        groups = defaultdict(lambda: {"TOTAL": 0, "UP": 0, "DOWN": 0})

        eqp_for_stats = chamber_eqp if chamber_only else main_eqp

        for e in eqp_for_stats:
            model = e.get("EQP_MODEL_CD", "Unknown")
            group = e.get("EQ_GROUP", "Unknown")
            stat = e.get("MES_STAT_TYP")

            models[model]["TOTAL"] += 1
            groups[group]["TOTAL"] += 1

            if stat == "Up":
                models[model]["UP"] += 1
                groups[group]["UP"] += 1
            else:
                models[model]["DOWN"] += 1
                groups[group]["DOWN"] += 1

        lines.extend([
            f"",
            f"### 장비 현황(MODEL 기준)",
            f"",
            f" | MODEL | TOTAL | UP | DOWN |",
            f" | ------- | ------- | ----- | ------- |",
        ])
        for k, v in sorted(models.items()):
            lines.append(f"| {k} | {v['TOTAL']} | {v['UP']} | {v['DOWN']} |")

        lines.extend([
            f"",
            f"### 장비 현황(EQP_GROUP 기준)",
            f"",
            f" | EQ_GROUP | TOTAL | UP | DOWN |",
            f" | ------- | ------- | ----- | ------- |",
        ])
        for k, v in sorted(groups.items()):
            lines.append(f"| {k} | {v['TOTAL']} | {v['UP']} | {v['DOWN']} |")

        return '\n'.join(lines)

def get_operhis_data(ctn_desc: str, lot_cd: Optional[str] = None, prev: bool = False, limit: int = 10) -> List[Dict]:

    if not lot_cd:
        print("[DEBUG] LOT_CD가 없어 조회 불가")
        return []

    try:
        print(f"[DEBUG] 공정 이력 API 호출 (LOT_CD : {lot_cd})")
        raw_result = call_operhis_api(lot_cd)

        if raw_result is None:
            print("[DEBUG] API 호출 결과 : None")
            return []
        all_data = raw_result
        print(f"[DEBUG] API 호출 성공 : {len(all_data)}건")

    except Exception as e:
        print(f"[DEBUG] 공정 이력 API 호출 실패: {e}")
        return []

    target_ctn = ctn_desc.strip().upper()

    ctn_items = []
    for item in all_data:
        raw_val = item.get("OPERATIONDESC")

        if raw_val is None:
            continue

        if str(raw_val).strip().upper() == target_ctn:
            ctn_items.append(item)

    if not ctn_items:
        print(f"[DEBUG] {lot_cd}에서 '{ctn_desc}' 공정 없음 (찾은 항목 : {len(ctn_items)})")
        return []

    if prev:
        valid_levels = []
        for item in ctn_items:
            lvl = item.get("OPERLEVEL")
            if lvl is not None:
                valid_levels.append(lvl)

        if not valid_levels:
            print("[DEBUG] 유효한 OPERLEVEL이 없습니다.")
            return []

        target_operlevel = max(valid_levels)
        print(f"[DEBUG] 기준 OPERLEVEL : {target_operlevel}")

        prev_items = []
        for item in all_data:
            lot_id = item.get("LOT_ID")
            if lot_id is None:
                continue
            if not str(lot_id).startswith(str(lot_cd)):
                continue
            op_lvl = item.get("OPERLEVEL")
            if op_lvl is None:
                continue
            if op_lvl < target_operlevel:
                prev_items.append(item)

        sorted_prev = sorted(prev_items, key=lambda x: x.get("OPERLEVEL", 0), reverse=True)
        return sorted_prev[:limit]

    else:
        sorted_ctn = sorted(ctn_items, key=lambda x: x.get("OPERLEVEL", 0), reverse=True)
        return sorted_ctn[:limit]

def format_operhis_data(data: List[Dict], ctn_desc: str) -> str:
    from collections import defaultdict

    lot_groups = defaultdict(list)
    for item in data:
        lot_id = item.get("LOT_ID")
        if lot_id:
            lot_groups[lot_id].append(item)

    selected = []
    for lot_id, items in lot_groups.items():
        max_item = max(items, key=lambda x: x.get("OPERLEVEL", 0))
        selected.append(max_item)

    selected = sorted(selected, key=lambda x: x.get("OPERLEVEL", 0), reverse=True)[:10]

    lines = [f"##{ctn_desc} 공정 이력\n"]
    lines.extend([
        "",
        "| LOT_ID | WF_QTY | CTN_DESC | MES_PROC_STAT_CD | LAST_EVENT_TM | FLOW_ID |",
        "| -------- | -------- | ---------- | ------------------------- | --------------------- | --------------- |",
    ])

    for item in selected:
        lines.append(
            f" | {item.get('LOT_ID', '-')} | {item.get('WF_QTY', '-')} | "
            f"{item.get('CTN_DESC', '-')} | {item.get('MES_PROC_STAT_CD', '-')} | "
            f"{item.get('LAST_EVENT_TM', '-')} | {item.get('FLOW_ID', '-')} |"
        )

    if len(lot_groups) > 10:
        lines.append(f"\n*외 {len(lot_groups)-10}건 LOT 생략*")

    return '\n'.join(lines)


def format_table(headers: List[str], rows: List[Dict]) -> str:

    if not rows:
        return "데이터가 없습니다"

    lines = []

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        values = [str(row.get(h, "-")) for h in headers]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)

def chat_upload_retrieve_documents(state: GraphState) -> GraphState:
    user_id = state.user_id
    session_id = state.session_id

    connection_args = {
        "uri": config.DOCU_VECTOR_DB_URI,
        "db_name": config.DOCU_VECTOR_DB_NAME,
        "user": config.DOCU_VECTOR_DB_USER,
        "password": config.DOCU_VECTOR_DB_PASSWORD
    }

    store = Milvus(
        embedding_function=embedding_function,
        collection_name=config.DOCU_USER_COLLECTION_NAME,
        connection_args=connection_args,
        vector_field='dense_vector',
        text_field='text',
        enable_dynamic_field=True
    )

    docs = store.similarity_search(
        query=state.query, k=50, fetch_k=50,
        param={"metric_type": "COSINE", "params": {"M": 8, "efConstruction": 64, "efSearch": 50}},
    )
    state.chat_upload_retrieved_docs = docs
    return state


# ============================================================
# 노드 1: 질문 분류 (의도 파악만 담당 - API 호출 최소화)
# ============================================================

def classify(state: GraphState) -> GraphState:
    """질문을 분석해 intent와 필요한 파라미터를 state에 저장한다."""
    query = state.query
    query_lower = query.lower()
    query_upper = query.upper()

    print(f"\n[DEBUG] ============")
    print(f"[DEBUG] 입력질문 : {query}")
    print(f"\n[DEBUG] ============")

    candidate = extract_candidate(query)
    print(f"[DEBUG] 추출된 후보 : {candidate}")

    # 1단계: LOT 판별
    if candidate:
        print(f"[DEBUG] call_lotid 호출 : {candidate}")
        lot_data = call_lotid(candidate)
        print(f"[DEBUG] call_lotid 결과 : {lot_data}")

        if lot_data:
            state.is_lot = True
            state.lot_id = candidate
            state.query_type = "sql_api"
            print(f"[DEBUG] LOT 확인 됨 : {candidate}")

            setmo_keywords = ['셋모', 'setmo', 'setmonitor', 'set monitoring', 'set-monitoring', '계측', 's/m', 's.m', 'set monitoring']
            hold_keywords = ['f/h', 'f.h', 'fh', 'future action', 'future hold', '퓨처홀드', '퓨처 홀드', '퓨처홀드']
            slot_keywords = ['슬롯', 'slot', '랏정보', 'lot 정보', '정보', '상태', '어디', '위치', '공정']
            his_keywords = ['이력']

            if any(k in query_lower for k in setmo_keywords):
                print("[DEBUG] -> SETMO Monitor 의도 확인")
                state.intent = "setmonitoring"
            elif any(k in query_lower for k in hold_keywords):
                print("[DEBUG] -> Future Hold 의도 확인")
                state.intent = "future_action"
            elif any(k in query_lower for k in slot_keywords):
                print("[DEBUG] -> Slot Info 의도 확인")
                state.intent = "lot_info"
            elif any(k in query_lower for k in his_keywords):
                print("[DEBUG] -> LOT 이력 조회 의도 확인")
                state.intent = "lot_his"
            else:
                print("[DEBUG] -> 키워드 없음, 기본 LOT 이력 조회")
                state.intent = "lot_his"

            return state
        else:
            print(f"[DEBUG] LOT 아님: {candidate}")

    # 2단계: 장비 판별
    print("[DEBUG] LOT 아님 -> 2단계 진행")
    eqp_candidate = candidate
    print(f"[DEBUG] 장비 후보 : {eqp_candidate}, FAB 힌트 : {extract_fab(query_upper) or '없음'}")

    if eqp_candidate:
        try:
            eq_check = call_eq(eqp_candidate)
        except Exception as e:
            print(f"[DEBUG] call eq 실패: {e}")
            eq_check = None

        if eq_check and len(eq_check) > 0:
            actual_fab = eq_check[0].get("FAC_ID") or eq_check[0].get("SRC")
            print(f"[DEBUG] call_eq 결과 : {eqp_candidate} in {actual_fab}")

            print(f"[DEBUG] {actual_fab} 상세 API 조회")
            eqp_data = get_eqp_by_fab(actual_fab, eqp_candidate)

            if eqp_data:
                print(f"[DEBUG] 상세 조회 성공: {len(eqp_data)}건")
                state.is_eqp = True
                state.eqp_id = eqp_candidate
                state.eqp_data = eqp_data
                state.fab = actual_fab
                state.query_type = "sql_api"

                info_keywords = ['장비정보', '장비 정보', 'equipment info', 'eqp info', '정보']
                if any(k in query_lower for k in info_keywords):
                    print("[DEBUG] -> 장비 정보 의도 확인")
                    state.intent = "eqp_info"
                else:
                    print("[DEBUG] -> 장비 상태 의도 확인 (기본값)")
                    state.intent = "eqp_status"

                return state
            else:
                print(f"[DEBUG] 상세 조회 실패 : {actual_fab} {eqp_candidate}")
        else:
            print(f"[DEBUG] 장비 아님: {eqp_candidate}")

    # 3단계: FAB+MODEL 판별
    print("[DEBUG] 3 단계: FAB+MODEL 판별 시도")
    fab_model_query = parse_fab_model_query(query)
    print(f"[DEBUG] parse_fab_model_query 결과 : {fab_model_query}")

    if fab_model_query:
        state.fab = fab_model_query["fab"]
        state.section_grp_nm = fab_model_query.get("section_grp_nm")
        state.fab_model = fab_model_query.get("model")
        state.fab_proc_model = fab_model_query.get("proc_model")
        state.fab_vendor = fab_model_query.get("vendor")
        state.fab_area = fab_model_query.get("area")
        state.fab_keyword = fab_model_query.get("keywords", [])
        state.fab_group_by = fab_model_query.get("group_by")
        state.fab_aggregate = fab_model_query.get("aggregate", "list")
        state.fab_chamber_only = fab_model_query.get("chamber_only", False)
        state.intent = "fab_model_query"
        state.query_type = "sql_api"
        return state

    # 4단계: OPER_STEP 판별
    print("[DEBUG] 3.5 단계: OPER_STEP 판별 시도 (구조화된 파싱)")
    oper_query_info = parse_operation_query(query)

    if oper_query_info:
        lot_cd = oper_query_info.get("lot_cd")
        ctn_candidate = oper_query_info.get("ctn_desc_candidate")
        is_prev = oper_query_info.get("is_prev", False)
        print(f"[DEBUG] 파싱 결과 - LOT_CD : {lot_cd}, CTN_CANDIDATE: {ctn_candidate}, PREV: {is_prev}")

        if ctn_candidate:
            state.parsed_lot_cd = lot_cd
            state.parsed_ctn_desc = ctn_candidate
            state.parsed_is_prev = is_prev
            state.intent = "oper_step"
            state.query_type = "sql_api"
            return state
        else:
            print(f"[DEBUG] 조건에 맞는 데이터 없음")

    # 5단계: 일반 질문 (RAG)
    print("[DEBUG] 5단계 : 일반 질문(RAG)")
    state.intent = "general_qa"
    state.query_type = "general"
    state.skip_rag = False
    return state


# ============================================================
# 라우팅 함수: intent → 다음 노드 결정
# ============================================================

def route_by_intent(state: GraphState) -> str:
    """classify 이후 intent에 따라 실행할 fetch 노드를 반환한다."""
    intent_map = {
        "setmonitoring":  "fetch_lot_setmo",
        "future_action":  "fetch_lot_future_hold",
        "lot_info":       "fetch_lot_slot",
        "lot_his":        "fetch_lot_history",
        "eqp_info":       "fetch_eqp_info",
        "eqp_status":     "fetch_eqp_status",
        "fab_model_query":"fetch_fab_model",
        "oper_step":      "fetch_oper_step",
    }
    return intent_map.get(state.intent, "chat_upload_retrieve_documents")


# ============================================================
# 노드 2~5: LOT 관련 fetch 노드
# ============================================================

def fetch_lot_setmo(state: GraphState) -> GraphState:
    """LOT SetMonitor 데이터 조회 및 포맷"""
    data = call_setmo_check(state.lot_id)
    if data and any(d.get('ACT_NM') == 'Monitor' for d in data):
        state.answer = format_setmo_data(data, state.lot_id, "Monitor")
    else:
        state.answer = f"{state.lot_id} setmonitor 데이터 없음"
    state.skip_rag = True
    print(f"[DEBUG] 응답완료 (skip_rag=True)")
    return state


def fetch_lot_future_hold(state: GraphState) -> GraphState:
    """LOT Future Hold 데이터 조회 및 포맷"""
    data = call_setmo_check(state.lot_id)
    if data and any(d.get('ACT_NM') == 'makeOnHold' for d in data):
        state.answer = format_setmo_data(data, state.lot_id, "makeOnHold")
    else:
        state.answer = f"{state.lot_id} Future Hold 데이터 없음"
    state.skip_rag = True
    print(f"[DEBUG] 응답완료 (skip_rag=True)")
    return state


def fetch_lot_slot(state: GraphState) -> GraphState:
    """LOT 슬롯 정보 조회 및 포맷"""
    data = call_slot_info(state.lot_id)
    state.answer = format_slot_data(data, state.lot_id) if data else f"{state.lot_id} 슬롯 정보 없음"
    state.skip_rag = True
    print(f"[DEBUG] 응답 완료 (skip_rag=True)")
    return state


def fetch_lot_history(state: GraphState) -> GraphState:
    """LOT 이력 조회 및 포맷"""
    data = call_lothis(state.lot_id)
    state.answer = format_lothis_data(data, state.lot_id) if data else f"{state.lot_id} 이력 정보 없음"
    state.skip_rag = True
    print(f"[DEBUG] 응답완료 (skip_rag=True)")
    return state


# ============================================================
# 노드 6~7: 장비(EQP) 관련 fetch 노드
# ============================================================

def fetch_eqp_info(state: GraphState) -> GraphState:
    """장비 정보 분석 및 포맷"""
    analyzed = analyze_eqp_data(state.eqp_data, state.eqp_id)
    state.answer = format_eqp_info_table(analyzed)
    state.skip_rag = True
    print(f"[DEBUG] 응답 완료 (skip_rag=True)")
    return state


def fetch_eqp_status(state: GraphState) -> GraphState:
    """장비 상태 분석 및 포맷"""
    is_cmp = any("CMP" in str(d.get("MGMT_AREA_ID", "")).upper() for d in state.eqp_data)
    analyzed = analyze_eqp_data(state.eqp_data, state.eqp_id)
    state.answer = format_eqp_status_table(analyzed, show_chamber=not is_cmp, show_port=True)
    state.skip_rag = True
    print(f"[DEBUG] 응답 완료 (skip_rag=True)")
    return state


# ============================================================
# 노드 8: FAB+MODEL 현황 fetch 노드
# ============================================================

def fetch_fab_model(state: GraphState) -> GraphState:
    """FAB+MODEL 기준 장비 현황 조회 및 포맷"""
    fab = state.fab
    raw_data = get_eqp_by_fab(fab, "*")

    if not raw_data:
        state.answer = f"{fab} 펩에 등록된 장비 데이터가 없습니다."
        state.skip_rag = True
        return state

    query_info = {
        "fab":           state.fab,
        "model":         state.fab_model,
        "proc_model":    state.fab_proc_model,
        "vendor":        state.fab_vendor,
        "area":          state.fab_area,
        "section_grp_nm":state.section_grp_nm,
        "keywords":      state.fab_keyword or [],
        "chamber_only":  state.fab_chamber_only,
        "group_by":      state.fab_group_by,
        "aggregate":     state.fab_aggregate,
    }

    filtered_data = apply_filters(raw_data, query_info)

    if not filtered_data:
        conditions = []
        if state.fab_model:     conditions.append(f"모델:{state.fab_model}")
        if state.fab_area:      conditions.append(f"구역:{state.fab_area}")
        if state.section_grp_nm: conditions.append(f"섹션:{state.section_grp_nm}")
        state.answer = f"{fab} 펩에서 {', '.join(conditions)} 조건에 맞는 장비를 찾을 수 없습니다."
    else:
        filtered_data.sort(key=lambda x: x.get('EQP_ID', ''))
        state.answer = format_fab_model_result(filtered_data, query_info, state.query)

    state.sql_result = raw_data
    state.skip_rag = True
    print(f"[DEBUG] 응답 완료 (skip_rag=True)")
    return state


# ============================================================
# 노드 9: 공정 이력(OPER_STEP) fetch 노드
# ============================================================

def fetch_oper_step(state: GraphState) -> GraphState:
    """공정 단계 이력 조회 및 포맷"""
    oper_data = get_operhis_data(
        ctn_desc=state.parsed_ctn_desc,
        lot_cd=state.parsed_lot_cd,
        prev=state.parsed_is_prev,
    )
    if oper_data:
        print(f"[DEBUG] OPER_STEP 데이터 확인됨: {len(oper_data)}건")
        state.answer = format_operhis_data(oper_data, state.parsed_ctn_desc)
    else:
        print(f"[DEBUG] 조건에 맞는 데이터 없음")
        state.answer = f"'{state.parsed_ctn_desc}' 공정 데이터 없음"
    state.skip_rag = True
    print(f"[DEBUG] 응답 완료 (skip_rag=True)")
    return state


# ============================================================
# 노드 10: 조건부 문서 검색 (RAG)
# ============================================================

def conditional_retrieve(state: GraphState) -> GraphState:
    if state.skip_rag:
        state.retrieved_docs = []
        return state

    query = state.query
    docs = milvus_vectorstore.similarity_search_with_score(
        query=query, k=50, fetch_k=50,
        param={"metric_type": "COSINE", "params": {"M": 8, "efConstruction": 64, "efSearch": 100}},
    )

    threshold = 0.10
    filtered = []
    for doc, score in docs:
        sim = max(0.0, score)
        if sim >= threshold:
            if 'title' in doc.metadata:
                del doc.metadata['title']
            doc.metadata["similarity"] = sim
            filtered.append(doc)

    if filtered:
        stop_words = {"은", "는", "의", "가", "을", "를", "에", "와", "과", "로"}
        keywords = [w for w in query.split() if len(w) > 1 and w not in stop_words]

        for doc in filtered:
            content = doc.page_content.lower()
            doc.metadata["keyword_score"] = sum(1 for w in keywords if w.lower() in content)

        max_kw = max(doc.metadata["keyword_score"] for doc in filtered)
        for doc in filtered:
            norm = doc.metadata["keyword_score"] / max_kw if max_kw > 0 else 0
            doc.metadata["combined_score"] = doc.metadata["similarity"] * 0.7 + norm * 0.3

        filtered = sorted(filtered, key=lambda x: x.metadata["combined_score"], reverse=True)

    state.retrieved_docs = filtered
    return state


# ============================================================
# 노드 11: 최종 답변 생성
# ============================================================

@manager.main_model_type()
def generate_response(state: GraphState) -> GraphState:
    생략

    return state


# ============================================================
# 워크플로우 등록
# ============================================================

manager.add_node(name="classify",                       description="질문 분류 및 의도 파악",        func=classify)
manager.add_node(name="fetch_lot_setmo",                description="LOT SetMonitor 조회",          func=fetch_lot_setmo)
manager.add_node(name="fetch_lot_future_hold",          description="LOT Future Hold 조회",         func=fetch_lot_future_hold)
manager.add_node(name="fetch_lot_slot",                 description="LOT 슬롯 정보 조회",            func=fetch_lot_slot)
manager.add_node(name="fetch_lot_history",              description="LOT 이력 조회",                func=fetch_lot_history)
manager.add_node(name="fetch_eqp_info",                 description="장비 정보 조회",               func=fetch_eqp_info)
manager.add_node(name="fetch_eqp_status",               description="장비 상태 조회",               func=fetch_eqp_status)
manager.add_node(name="fetch_fab_model",                description="FAB+MODEL 현황 조회",          func=fetch_fab_model)
manager.add_node(name="fetch_oper_step",                description="공정 이력 조회",               func=fetch_oper_step)
manager.add_node(name="chat_upload_retrieve_documents", description="채팅 업로드 문서 검색",         func=chat_upload_retrieve_documents)
manager.add_node(name="conditional_retrieve",           description="조건부 문서 검색",             func=conditional_retrieve)
manager.add_node(name="generate_response",              description="최종 답변 생성",               func=generate_response)

manager.set_entry_point("classify")

manager.add_conditional_edges(
    "classify",
    route_by_intent,
    {
        "fetch_lot_setmo":              "fetch_lot_setmo",
        "fetch_lot_future_hold":        "fetch_lot_future_hold",
        "fetch_lot_slot":               "fetch_lot_slot",
        "fetch_lot_history":            "fetch_lot_history",
        "fetch_eqp_info":               "fetch_eqp_info",
        "fetch_eqp_status":             "fetch_eqp_status",
        "fetch_fab_model":              "fetch_fab_model",
        "fetch_oper_step":              "fetch_oper_step",
        "chat_upload_retrieve_documents": "chat_upload_retrieve_documents",
    }
)

for _fetch_node in [
    "fetch_lot_setmo", "fetch_lot_future_hold", "fetch_lot_slot", "fetch_lot_history",
    "fetch_eqp_info", "fetch_eqp_status", "fetch_fab_model", "fetch_oper_step",
]:
    manager.add_edge(_fetch_node, "chat_upload_retrieve_documents")

manager.add_edge("chat_upload_retrieve_documents", "conditional_retrieve")
manager.add_edge("conditional_retrieve", "generate_response")
manager.add_edge("generate_response", GAIA_STANDARD_OUTPUT_NODE)
manager.compile()
print(f"[{VERSION}] 워크플로우 컴파일 완료")
