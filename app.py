import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os

# ──────────────────────────────────────────────
# 0. 페이지 설정
# ──────────────────────────────────────────────
st.set_page_config(page_title="LLM의 할루미네이션", layout="wide")

# ──────────────────────────────────────────────
# 1. 데이터 로드 및 전처리
# ──────────────────────────────────────────────

@st.cache_data
def load_data():
    csv_path = "rag-hallucination.csv"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # 예시 데이터 생성 (CSV가 없을 경우)
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            "index": range(n),
            "Interaction_ID": [f"INT-{i:04d}" for i in range(n)],
            "User_Prompt": [f"질문 {i}" for i in range(n)],
            "Provided_Context": [f"문맥 {i}" for i in range(n)],
            "LLM_Generated_Response": [f"응답 {i}" for i in range(n)],
            "LLM_Model_Name": np.random.choice(["GPT-4", "Claude-3", "Gemini", "Llama-3"], n),
            "Temperature_Setting": np.round(np.random.uniform(0.0, 1.0, n), 2),
            "Vector_DB_Similarity_Score": np.round(np.random.uniform(0.3, 1.0, n), 3),
            "Prompt_Tokens": np.random.randint(50, 2000, n),
            "Response_Tokens": np.random.randint(20, 1500, n),
            "Context_Response_Overlap_Ratio": np.round(np.random.uniform(0.0, 1.0, n), 3),
            "Entity_Match_Score": np.round(np.random.uniform(0.0, 1.0, n), 3),
            "Sentence_Complexity_Index": np.round(np.random.uniform(1.0, 10.0, n), 2),
            "Subjectivity_Score": np.round(np.random.uniform(0.0, 1.0, n), 3),
            "Contradiction_Words_Count": np.random.randint(0, 20, n),
            "Answer_Relevance_Score": np.round(np.random.uniform(0.0, 1.0, n), 3),
            "Context_Faithfulness_Score": np.round(np.random.uniform(0.0, 1.0, n), 3),
            "Hallucination_Type": np.random.choice(["사실 오류", "환상 생성", "문맥 이탈", "없음"], n),
            "Is_Hallucination": np.random.choice([0, 1], n, p=[0.65, 0.35]),
        })

    # ── 삭제할 컬럼 ──
    drop_cols = [
        "index",
        "Interaction_ID",
        "User_Prompt",
        "Provided_Context",
        "LLM_Generated_Response",
        "Subjectivity_Score",
        "Context_Response_Overlap_Ratio",
        "Contradiction_Words_Count",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ── 영문 → 한글 컬럼명 매핑 ──
    rename_map = {
        "LLM_Model_Name":       "모델 이름",
        "Temperature_Setting":  "온도 설정값",
        "Vector_DB_Similarity_Score": "데이터베이스 유사도 점수",
        "Prompt_Tokens":        "입력 토큰 수",
        "Response_Tokens":      "답변 토큰 수",
        "Entity_Match_Score":   "개체 일치 점수",
        "Sentence_Complexity_Index": "복잡도 지수",
        "Answer_Relevance_Score":   "답변 관련성 점수",
        "Context_Faithfulness_Score": "문맥 충실도 점수",
        "Hallucination_Type":   "환각 유형",
        "Is_Hallucination":     "환각 발생 여부",
    }
    df = df.rename(columns=rename_map)
    return df


df = load_data()

# ──────────────────────────────────────────────
# 2. 사이드바: 메뉴 선택 (가장 왼쪽)
# ──────────────────────────────────────────────
st.sidebar.title("메뉴")
menu_choice = st.sidebar.radio(
    "화면 선택",
    ["서론 화면", "막대그래프 화면"],
    index=1,  # 기본: 막대그래프 화면
)

# ──────────────────────────────────────────────
# 3. 사이드바: 토큰 버전 전환 버튼
# ──────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("그래프 버전 전환")

if "token_mode" not in st.session_state:
    st.session_state.token_mode = "입력"  # 기본값

token_btn = st.sidebar.button(
    label=f"현재: {st.session_state.token_mode} 토큰수",
    key="toggle_token",
)
if token_btn:
    st.session_state.token_mode = (
        "답변" if st.session_state.token_mode == "입력" else "입력"
    )

st.sidebar.caption("버튼을 눌러 입력 토큰수 ↔ 답변 토큰수를 전환하세요.")

# ──────────────────────────────────────────────
# 4. 모든 화면 공통: 상단 제목
# ──────────────────────────────────────────────
st.title("LLM의 할루미네이션")

# ──────────────────────────────────────────────
# 5. [서론 화면]
# ──────────────────────────────────────────────
if menu_choice == "서론 화면":
    st.markdown("""
    ## 개요
    이 앱은 **RAG(Retrieval-Augmented Generation) 파이프라인**에서 발생하는  
    LLM의 **환각(Hallucination)** 현상을 데이터로 탐색하는 도구입니다.

    ### 사용 데이터
    - 파일: `rag-hallucination.csv`
    - 총 19개 컬럼 중 분석에 필요한 11개 컬럼을 선별하여 사용합니다.

    ### 분석 지표
    | 지표 | 설명 |
    |------|------|
    | 입력 토큰 수 | 사용자 프롬프트의 토큰 길이 |
    | 답변 토큰 수 | LLM이 생성한 응답의 토큰 길이 |
    | 데이터베이스 유사도 점수 | 검색된 문맥과 쿼리 간 유사도 |
    | 개체 일치 점수 | 문맥과 응답 간 개체(Entity) 일치율 |
    | 복잡도 지수 | 응답 문장의 구조적 복잡도 |
    | 답변 관련성 점수 | 응답이 질문과 얼마나 관련되었는지 |
    | 문맥 충실도 점수 | 응답이 제공된 문맥에 얼마나 충실한가 |
    | 환각 유형 | 환각의 종류 분류 |
    | 환각 발생 여부 | 환각 발생 시 1, 미발생 시 0 |

    ### 분석 방법
    - **입력 토큰 수**와 **답변 토큰 수**를 각각 5개 구간(퀀타일)으로 나눕니다.
    - 각 구간별 **환각 발생률(%)** = (해당 구간 내 환각 발생 행 수 ÷ 해당 구간 전체 행 수) × 100
    - 이를 막대그래프로 시각화하여 토큰 길이와 환각 발생 간의 관계를 살펴봅니다.
    """)

    st.info("왼쪽 메뉴에서 **막대그래프 화면**을 선택하면 그래프를 볼 수 있습니다.")

# ──────────────────────────────────────────────
# 6. [막대그래프 화면]
# ──────────────────────────────────────────────

else:
    st.markdown("""
    아래 그래프는 **토큰 수 구간별 환각 발생률**을 나타냅니다.  
    사이드바 버튼을 눌러 **입력 토큰 수** 버전과 **답변 토큰 수** 버전을 전환할 수 있습니다.
    """)

    # ── 토큰 컬럼 선택 ──
    token_col = "입력 토큰 수" if st.session_state.token_mode == "입력" else "답변 토큰 수"
    label_txt = "입력 토큰 수" if st.session_state.token_mode == "입력" else "답변 토큰 수"

    # ── 5개 퀀타일 구간 생성 ──
    df["_bin"] = pd.qcut(
        df[token_col],
        q=5,
        labels=["구간 1", "구간 2", "구간 3", "구간 4", "구간 5"],
        duplicates="drop",
    )

    # ── 구간별 환각 발생률 계산 ──
    grouped = df.groupby("_bin")
    bin_stats = pd.DataFrame({
        "구간": grouped.size().index,
        "전체 행 수": grouped.size().values,
        "환각 발생 행 수": grouped["환각 발생 여부"].sum().values,
    })
    bin_stats["환각 발생률(%)"] = (
        bin_stats["환각 발생 행 수"] / bin_stats["전체 행 수"] * 100
    ).round(2)

    # ── Plotly 막대그래프 ──
    fig = px.bar(
        bin_stats,
        x="구간",
        y="환각 발생률(%)",
        text="환각 발생률(%)",
        color="환각 발생률(%)",
        color_continuous_scale="Reds",
        labels={"구간": f"{label_txt} 구간", "환각 발생률(%)": "환각 발생률 (%)"},
        title=f"{label_txt} 구간별 환각 발생률",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        xaxis_title=f"{label_txt} 구간 (퀀타일 5분할)",
        yaxis_title="환각 발생률 (%)",
        yaxis_range=[0, 100],
        template="plotly_white",
        height=480,
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── 구간별 상세 테이블 ──
    st.subheader("구간별 상세 통계")
    st.dataframe(
        bin_stats.rename(columns={
            "구간": "구간",
            "전체 행 수": "전체 행 수",
            "환각 발생 행 수": "환각 발생 행 수 (Is_Hallucination=1)",
            "환각 발생률(%)": "환각 발생률 (%)",
        }),
        hide_index=True,
        use_container_width=True,
    )

    # ── 구간 범위 정보 ──
    st.subheader("구간 범위")
    bin_ranges = pd.qcut(df[token_col], q=5, retbins=True, duplicates="drop")[1]
    range_rows = []
    labels = ["구간 1", "구간 2", "구간 3", "구간 4", "구간 5"]
    for i in range(len(labels)):
        lo = bin_ranges[i]
        hi = bin_ranges[i + 1]
        range_rows.append({
            "구간": labels[i],
            f"최소 {label_txt}": f"{lo:.1f}",
            f"최대 {label_txt}": f"{hi:.1f}",
        })
    st.dataframe(pd.DataFrame(range_rows), hide_index=True, use_container_width=True)

    st.caption(f"※ 구간 경계값은 `{token_col}` 컬럼 기준으로 산출되었습니다.")
