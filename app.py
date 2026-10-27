pip install matplotlib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. 데이터 로드 및 컬럼 정리
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("rag-hallucination.csv")

    # 삭제할 컬럼
    drop_cols = [
        "index",
        "Interaction_ID",
        "User_Prompt",
        "Provided_Context",
        "LLM_Generated_Response",
        "Subjectivity_Score",
        "Context_Response_Overlap_Ratio",
        "Contradiction_Words_Count",   # 원데이터 기준 (사용자 입력은 Contraction이지만 원열 이름 기준)
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 한글 rename
    rename_map = {
        "LLM_Model_Name":          "모델 이름",
        "Temperature_Setting":     "온도 설정값",
        "Vector_DB_Similarity_Score": "데이터베이스 유사도 점수",
        "Prompt_Tokens":           "입력 토큰 수",
        "Response_Tokens":         "답변 토큰 수",
        "Entity_Match_Score":      "개체 일치 점수",
        "Sentence_Complexity_Index":"복잡도 지수",
        "Answer_Relevance_Score":  "답변 관련성 점수",
        "Context_Faithfulness_Score":"문맥 충실도 점수",
        "Hallucination_Type":      "환각 유형",
        "Is_Hallucination":        "환각 발생 여부",
    }
    df = df.rename(columns=rename_map)
    return df

df = load_data()

# ─────────────────────────────────────────────
# 2. 막대그래프 계산 함수
#    → 해당 컬럼을 5개 범위로 나누고,
#      각 범위에서 환각발생여부==1인 행 수 / 전체 행 수 * 100
# ─────────────────────────────────────────────
def compute_hallucination_rate_by_bins(series, label):
    # 5개 구간 (등간격)
    bins = pd.cut(series, bins=5, precision=0)
    grouped = df.groupby(bins)["환각 발생 여부"]
    rates = (grouped.sum() / grouped.count() * 100).fillna(0)

    # 구간 레이블 정리
    labels = [f"{int(interval.left)}–{int(interval.right)}\n({interval.right - interval.left} 간격)"
              for interval in rates.index]
    return labels, rates.values

# ─────────────────────────────────────────────
# 3. Streamlit UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="LLM의 할루미네이션", layout="wide")

# ── 왼쪽 메뉴 (상단 고정) ──
st.sidebar.title("메뉴")
menu_choice = st.sidebar.radio("화면 선택", ["서론", "막대그래프"])

st.title("LLM의 할루미네이션")

# ─────────────────────────────────────────────
# 화면 1: 서론
# ─────────────────────────────────────────────
if menu_choice == "서론":
    st.markdown("""
    ### 소개
    이 앱은 **RAG 기반 LLM 응답에서 환각(Hallucination) 발생 패턴**을 시각화합니다.
    """)
    st.markdown("""
    **사용 데이터**
    - 파일: `rag-hallucination.csv`
    - 원본 19개 열 중 8개 열을 제거하고, 남은 11개 열을 한글 명칭으로 변경했습니다.
    """)
    st.markdown("""
    **제거한 열**
    - `index`, `Interaction_ID`, `User_Prompt`, `Provided_Context`,
      `LLM_Generated_Response`, `Subjectivity_Score`,
      `Context_Response_Overlap_Ratio`, `Contradiction_Words_Count`
    """)
    st.markdown("""
    **남은 열 (한글 rename)**
    | 원본 열 이름 | 변경 후 이름 |
    |-------------|-------------|
    | LLM_Model_Name | 모델 이름 |
    | Temperature_Setting | 온도 설정값 |
    | Vector_DB_Similarity_Score | 데이터베이스 유사도 점수 |
    | Prompt_Tokens | 입력 토큰 수 |
    | Response_Tokens | 답변 토큰 수 |
    | Entity_Match_Score | 개체 일치 점수 |
    | Sentence_Complexity_Index | 복잡도 지수 |
    | Answer_Relevance_Score | 답변 관련성 점수 |
    | Context_Faithfulness_Score | 문맥 충실도 점수 |
    | Hallucination_Type | 환각 유형 |
    | Is_Hallucination | 환각 발생 여부 |
    """)
    st.markdown("""
    **그래프 설명**
    - **입력 토큰 수** 또는 **답변 토큰 수**를 5개 범위로 나눕니다.
    - 각 범위에서 **환각 발생 여부(1)**인 행 수의 비율을 백분율로 나타냅니다.
    - 사이드바 버튼으로 두 버전 간 전환이 가능합니다.
    """)

# ─────────────────────────────────────────────
# 화면 2: 막대그래프
# ─────────────────────────────────────────────
else:
    st.markdown("""
    아래 막대그래프는 **입력 토큰 수** 또는 **답변 토큰 수**를 5개 범위로 나누어,
    각 범위 안에서 환각( Hallucination )이 발생한 행의 비율(%)을 보여줍니다.
    """)

    # ── 사이드바: 버전 전환 ──
    st.sidebar.title("그래프 버전")
    version = st.sidebar.radio(
        "막대그래프 버전 선택",
        ["입력 토큰 수 기준", "답변 토큰 수 기준"]
    )

    if version == "입력 토큰 수 기준":
        labels, rates = compute_hallucination_rate_by_bins(df["입력 토큰 수"], "입력 토큰 수")
        col_name = "입력 토큰 수"
    else:
        labels, rates = compute_hallucination_rate_by_bins(df["답변 토큰 수"], "답변 토큰 수")
        col_name = "답변 토큰 수"

    # ── 그래프 그리기 ──
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c" if r > 50 else "#f39c12" if r > 25 else "#27ae60"
              for r in rates]
    bars = ax.bar(labels, rates, color=colors, edgecolor="white", linewidth=0.8)

    # 막대 위에 수치 표시
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel(f"{col_name} 범위", fontsize=12)
    ax.set_ylabel("환각 발생 비율 (%)", fontsize=12)
    ax.set_title(f"{col_name} 구간별 환각 발생 비율", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(rates) * 1.25 + 5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=0, fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()

    st.pyplot(fig)

    # ── 데이터 요약 테이블 ──
    st.markdown("### 구간별 요약")
    summary_df = pd.DataFrame({
        "구간": labels,
        f"{col_name} 범위": [l.replace("\n", " ") for l in labels],
        "전체 행 수": [int((df[col_name].between(
            float(l.split("–")[0].strip()), float(l.split("–")[1].strip()
            )).sum())) for l in labels],
        "환각 발생 행 수": [int(r) for r in (df.groupby(pd.cut(df[col_name], bins=5))["환각 발생 여부"].sum())],
        "환각 발생 비율 (%)": [round(r, 2) for r in rates],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
