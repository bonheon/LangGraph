"""
RAGAS 평가 파이프라인 — Ceeria RAG 품질 측정 (ragas 0.4.x 호환)

실행 전 설치:
    pip install ragas datasets

평가 대상: general 경로 (skip_rag=False) 쿼리만
           LOT/EQP/FAB/OPER 구조적 경로는 벡터 검색이 없어 RAGAS 대상 외
"""

import os
import json
from datetime import datetime
from typing import List, Dict

from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings as LCEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# ── 환경변수 & RAGAS용 LLM / Embeddings 초기화 ───────────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
_api_key   = os.getenv("OPENAI_API_KEY")
_lc_llm    = ChatOpenAI(model="gpt-4o-mini", api_key=_api_key)
_lc_emb    = LCEmbeddings(model="text-embedding-3-small", api_key=_api_key)
_ragas_llm = LangchainLLMWrapper(_lc_llm)
_ragas_emb = LangchainEmbeddingsWrapper(_lc_emb)

# ragas 인스턴스에 LLM/Embeddings 주입 (0.4.x 방식)
faithfulness.llm        = _ragas_llm
answer_relevancy.llm    = _ragas_llm
answer_relevancy.embeddings = _ragas_emb
context_precision.llm   = _ragas_llm
context_recall.llm      = _ragas_llm

# ── 워크플로우 임포트 ─────────────────────────────────────────────────────
from workflow_notebook_toolcalling import run_workflow


# ── 골든셋 정의 ──────────────────────────────────────────────────────────
# question    : 평가할 질문 (general 경로로 가는 것만)
# ground_truth: 정답 요약 (ContextPrecision/Recall 측정에 사용)
#               없으면 Faithfulness + AnswerRelevancy만 측정 가능
GOLDEN_SET: List[Dict] = [
    {
        "question": "CVD 공정에서 두께 편차가 생기는 원인이 뭐야?",
        "ground_truth": "가스 유량 불균형, 챔버 온도 불균일, 압력 변동이 주요 원인이다. "
                        "LPCVD에서 ±2°C 이상 온도 편차가 발생하면 두께 균일도에 영향을 준다.",
    },
    {
        "question": "CMP 공정에서 스크래치가 발생하면 어떻게 조치해?",
        "ground_truth": "슬러리 필터 점검, 패드 교체 주기 확인(2000웨이퍼), 다운포스 압력 재조정이 필요하다. "
                        "스크래치 원인은 슬러리 입자 오염, 패드 컨디셔닝 불량, 드레서 마모 등이다.",
    },
    {
        "question": "LPCVD 장비 PM 주기와 항목을 알려줘",
        "ground_truth": "PM 주기는 500 사이클 또는 3개월이다. "
                        "PM 항목은 챔버 세정(HF wet clean), 히터 균일도 측정, 가스 라인 누기 점검, 쿼츠 보트 교체이며 "
                        "완료 후 더미 런 3회 이상 필수다.",
    },
    {
        "question": "장비 FAULT 발생 시 초기 대응 절차는?",
        "ground_truth": "알람 코드 확인 후 MES 장비 상태를 Down으로 변경하고 담당 엔지니어에게 즉시 연락한다. "
                        "진행 중인 LOT 상태를 확인하고 Track-out 처리 후 원인 분석 리포트를 작성한다.",
    },
    {
        "question": "ALD 막두께 균일도 불량의 원인이 뭐야?",
        "ground_truth": "전구체 펄스 시간 오류, 퍼지 불완전으로 인한 CVD 모드 전환, 챔버 온도 불균일이 원인이다. "
                        "TEL 장비에서는 전구체 공급 라인 온도 60~80°C 유지가 핵심이다.",
    },
    {
        "question": "포토 리소그래피에서 패턴 불량이 생기면?",
        "ground_truth": "포커스 오류, 노광량 이상, 레지스트 코팅 불균일, 현상 불량이 원인이다. "
                        "CD 편차가 ±5% 초과 시 재작업(rework)을 검토한다.",
    },
    {
        "question": "Setmonitor가 뭐야?",
        "ground_truth": "특정 공정 후 계측 지시를 자동 부여하는 기능이다. "
                        "계측 슬롯과 담당 엔지니어를 설정하며 이상 시 makeOnHold로 자동 연계된다.",
    },
    {
        "question": "Future Hold는 어떻게 설정해?",
        "ground_truth": "MES > LOT 관리 > makeOnHold에서 설정한다. "
                        "홀드 사유, 대상 공정, 담당 엔지니어를 필수로 입력해야 한다.",
    },
]


# ── 워크플로우 실행 & 결과 수집 ───────────────────────────────────────────
def collect_eval_data(golden_set: List[Dict], thread_prefix: str = "eval") -> List[Dict]:
    """각 질문을 워크플로우에 넣고 answer + contexts를 수집한다."""
    records = []
    for i, item in enumerate(golden_set):
        question = item["question"]
        print(f"\n[{i+1}/{len(golden_set)}] 평가 중: {question}")

        result = run_workflow(question, thread_id=f"{thread_prefix}-{i}")

        contexts = [
            doc["page_content"]
            for doc in result.get("retrieved_docs") or []
        ]
        if not contexts:
            print(f"  ⚠️  retrieved_docs 없음 — general 경로가 맞는지 확인하세요")

        records.append({
            "question":     question,
            "answer":       result.get("answer", ""),
            "contexts":     contexts,
            "ground_truth": item.get("ground_truth", ""),
        })
    return records


# ── RAGAS 평가 (0.4.x API) ────────────────────────────────────────────────
def run_ragas(records: List[Dict], use_ground_truth: bool = True):
    """
    ragas 0.4.x: Dataset → EvaluationDataset(SingleTurnSample 리스트)
    필드 매핑:
        question     → user_input
        answer       → response
        contexts     → retrieved_contexts
        ground_truth → reference
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["ground_truth"] if use_ground_truth else None,
        )
        for r in records
    ]
    dataset = EvaluationDataset(samples=samples)

    metrics = [faithfulness, answer_relevancy]
    if use_ground_truth and all(r["ground_truth"] for r in records):
        metrics += [context_precision, context_recall]

    print(f"\n[RAGAS] 메트릭: {[m.name for m in metrics]}")
    result = evaluate(dataset, metrics=metrics)
    return result


# ── 결과 출력 ─────────────────────────────────────────────────────────────
def print_report(result, records: List[Dict]) -> None:
    scores = result.to_pandas()
    numeric_cols = scores.select_dtypes("number").columns

    print("\n" + "=" * 70)
    print(f"  RAGAS 평가 리포트  ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print("=" * 70)

    print("\n[ 전체 평균 ]")
    for col in numeric_cols:
        avg = scores[col].mean()
        bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
        print(f"  {col:<30} {bar}  {avg:.3f}")

    print("\n[ 질문별 상세 ]")
    for i, row in scores.iterrows():
        print(f"\n  Q{i+1}. {records[i]['question']}")
        for col in numeric_cols:
            val = row[col]
            flag = "⚠️ " if (val is not None and val < 0.6) else "✅ "
            display = f"{val:.3f}" if val is not None else "N/A"
            print(f"       {flag}{col:<30} {display}")

    print("\n[ 주의 항목 (score < 0.6) ]")
    warnings = []
    for i, row in scores.iterrows():
        for col in numeric_cols:
            val = row[col]
            if val is not None and val < 0.6:
                warnings.append(
                    f"  Q{i+1} / {col}: {val:.3f}  ← '{records[i]['question']}'"
                )
    print("\n".join(warnings) if warnings else "  없음 — 전 항목 양호")
    print("\n" + "=" * 70)


# ── 결과 저장 ─────────────────────────────────────────────────────────────
def save_report(result, records: List[Dict], path: str = "ragas_report.json") -> None:
    df = result.to_pandas()
    numeric_cols = df.select_dtypes("number").columns
    output = {
        "evaluated_at": datetime.now().isoformat(),
        "summary":      {col: float(df[col].mean()) for col in numeric_cols},
        "details": [
            {
                "question": records[i]["question"],
                **{col: (float(row[col]) if row[col] is not None else None)
                   for col in numeric_cols},
            }
            for i, row in df.iterrows()
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {path}")


# ── 메인 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  Ceeria RAG 평가 시작")
    print(f"  골든셋 {len(GOLDEN_SET)}건")
    print("=" * 70)

    records = collect_eval_data(GOLDEN_SET)
    result  = run_ragas(records, use_ground_truth=True)
    print_report(result, records)
    save_report(result, records, path="ragas_report.json")
