# GAIA 개발 가이드

## 1. GAIA AGENT MANAGER 개요

LangGraph 기반으로 agentic AI 서비스 개발 및 성능 개선을 지원하는 플랫폼

워크플로우 함수, traceability & observation, multi turn 및 streaming output 등의 핵심 유틸리티 제공

---

## 2. 프로젝트 구조 및 chatversion 관리

- **핵심 디렉토리**: `cicd`, `src`
- **버전 관리 정책**: `src > workflows` 하위에 `v1_0` 형식으로 폴더 생성 및 관리

| 증가 조건 | 설명 |
|---|---|
| 1 | 워크플로우 구조(노드, 엣지) 변경 시 증가 |
| 2 | 내부 로직 및 참고 데이터 변경 시 증가 |
| 3 | LLM 모델 및 프롬프트 변경 시 증가 (폴더명에는 반영 X) |

---

## 3. Agentic AI 워크플로우 개발

### AgentManager 클래스

LangGraph를 매핑한 클래스로, 워크플로우 구성 필수

| 함수 | 설명 |
|---|---|
| `add_node` | 노드 생성 |
| `add_edge` | 노드 연결 |
| `add_conditional_edges` | 조건부 연결 |
| `set_entry_point` | 시작 노드 지정 |
| `compile` | 컴파일 |

### GaiaGraphState

노드 간 데이터 교환을 위한 pydantic 클래스 상속 필수

| 변수 | 설명 |
|---|---|
| `query` | 사용자 질의 |
| `chat_history` | 대화 이력 |
| `answer` | 최종 답변 |
| `docs` / `urls` / `db_data` | 참조 문서 / 링크 / DB 데이터 |

### GAIA 표준 Output 노드

Markdown 및 streaming 처리를 위한 최종 output 노드로 반드시 지정 필수

---

## 4. config 파일 설정

| 설정 항목 | 설명 |
|---|---|
| `SERVICE_ID` | 서비스 고유 코드 |
| `ACTIVE_MULTI_TURN`, `SEE_MAX_HISTORY` | 대화 이력 사용 여부 및 최대 저장 건수 설정 |
| `ACTIVE_TRACE` | `true` 시 플랫폼 DB에 이력 저장, 배포 시 `true` 필수 |
| `IS_SECURITY_SERVICE` | `true` 시 보안 규정 준수 로컬 LLM 사용, `false` 시 MS Azure ChatGPT 사용 |
| `GAIA/CUBE/API_OUTPUT_MARKDOWN` / `STREAM` | 채널별 Markdown 변환 및 Streaming 처리 여부 |
| 기타 | DB 연결 정보, Kafka 설정, knowhow 임베딩/vector DB 설정 등 |

---

## 5. AgentManager의 역할과 특징

### 정의

복잡한 LangChain / LangGraph의 Agent 구현 기능을 한 단계 래핑(Wrapping)하여 손쉽게 개발할 수 있도록 지원하는 라이브러리

### 장점

- A2A 연동, 시각화, 버전 관리 등의 편의 기능이 내장되어 있어, 개발자는 핵심 Agent 개발에만 집중 가능
- AgentManager로 Graph를 구성하면 GaIA 플랫폼의 다양한 편의 기능을 즉시 활용 가능

---

## 6. Workflow(Graph) 구성 요소

### Node 구현

- LangGraph 스타일과 동일하게 Graph의 State를 입력받아 반환하는 함수로 구현
- GaIA 플랫폼 동작을 위해 Graph당 필수적인 **메인 모델 선언** (`@manager.main_model_type()`) 이 필요함

### Node 선언

`add_node` 함수를 통해 구현된 노드를 Graph에 등록

### Edge 선언

`add_edge`, `set_entry_point` 등을 사용하여 노드 간 실행 순서와 관계를 정의

> GaIA UI에서 정상적인 답변을 받으려면 워크플로우가 반드시 `GAIA_STANDARD_OUTPUT_NODE` 노드로 끝나야 함

---

## 7. Workflow 컴파일 및 협업 주의사항

### 컴파일

`compile()` 완료 시 GaIA 플랫폼에 Graph가 업로드되어 서비스 사용 준비 완료

### Sub-Graph 제한

- GaIA AgentManager는 다중 Graph 방식을 지원하지 않고 **단일 Graph 형식**을 지향함
- 여러 sub-graph를 조합하여 multi-Graph를 구성하는 방식은 불가능

### 협업 개발 방식

- Workflow는 각 chat version의 단일 `workflow.py`에서 관리
- 개발자 간 역할 분담이 필요한 경우, Graph 단위가 아닌 **Node 단위**로 쪼개어 할당하는 방식으로 개발
