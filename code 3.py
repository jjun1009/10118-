# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(page_title="LLM Hallucination Analyzer", layout="wide")

# 유지할 컬럼
keep_columns = [
    "LLM_Model_Name",
    "Temperature_Setting",
    "Vector_DB_Similarity_Score",
    "Prompt_Tokens",
    "Response_Tokens",
    "Entity_Match_Score",
    "Sentence_Complexity_Index",
    "Answer_Relevance_Score",
    "Context_Faithfulness_Score",
    "Hallucination_Type",
    "Is_Hallucination",
]

rename_map = {
    "LLM_Model_Name": "모델 이름",
    "Temperature_Setting": "온도 설정값",
    "Vector_DB_Similarity_Score": "데이터베이스 유사도 점수",
    "Prompt_Tokens": "입력 토큰 수",
    "Response_Tokens": "답변 토큰 수",
    "Entity_Match_Score": "개체 일치 점수",
    "Sentence_Complexity_Index": "복잡도 지수",
    "Answer_Relevance_Score": "답변 관련성 점수",
    "Context_Faithfulness_Score": "문맥 충실도 점수",
    "Hallucination_Type": "환각 유형",
    "Is_Hallucination": "환각 발생 여부",
}

# 데이터 로드
@st.cache_data
def load_data():
    df = pd.read_csv("rag_hallucination.csv")
    df = df[keep_columns].rename(columns=rename_map)
    return df

df = load_data()

# ---------- 사이드바: 버전 전환 ----------
st.sidebar.title("📊 지표 선택")
version = st.sidebar.radio(
    "막대그래프 버전",
    ["입력 토큰 수 기준", "답변 토큰 수 기준"],
    index=0,
    help="입력 토큰수 또는 답변 토큰수 구간별 환각 발생률을 표시합니다.",
)

# 토큰 컬럼 선택
if version == "입력 토큰 수 기준":
    token_col = "입력 토큰 수"
else:
    token_col = "답변 토큰 수"

# ---------- 좌측 메뉴 ----------
if "menu" not in st.session_state:
    st.session_state.menu = "서론"

menu = st.sidebar.radio(
    "메뉴",
    ["서론", "막대그래프"],
    index=0 if st.session_state.menu == "서론" else 1,
    horizontal=False,
    label_visibility="collapsed",
)
st.session_state.menu = menu

# 모든 화면 공통 제목
st.title("LLM의 할루미네이션")

# ==================== 서론 화면 ====================
if menu == "서론":
    st.markdown("""
    ## 📝 개요
    이 앱은 **RAG(Retrieval-Augmented Generation)** 환경에서  
    LLM의 할루미네이션(환각) 발생 양상을 탐색하기 위한 시각화 도구입니다.

    ### 사용 방법
    1. 왼쪽 메뉴에서 **서론** 또는 **막대그래프**를 선택합니다.
    2. 사이드바에서 **입력 토큰 수 기준** / **답변 토큰 수 기준**을 전환합니다.
    3. 막대그래프 화면에서 구간별 환각 발생률을 확인합니다.

    ### 데이터
    - 원본 파일: `rag_hallucination.csv`
    - 주요 지표: 모델 이름, 온도 설정값, 데이터베이스 유사도 점수,  
      입력/답변 토큰 수, 개체 일치 점수, 복잡도 지수,  
      답변 관련성 점수, 문맥 충실도 점수, 환각 유형, 환각 발생 여부
    """)

# ==================== 막대그래프 화면 ====================
elif menu == "막대그래프":
    st.markdown(f"### {version} — 구간별 환각 발생률")

    # 5개 구간으로 나누기
    min_val = df[token_col].min()
    max_val = df[token_col].max()
    bins = pd.cut(
        df[token_col],
        bins=5,
        labels=[f"구간 {i+1}" for i in range(5)],
        include_lowest=True,
    )
    df["토큰 구간"] = bins

    # 구간별 집계 — ★★★ 컬럼명을 변경된 한국어 이름으로 사용 ★★★
    summary = (
        df.groupby("토큰 구간", observed=True)
        .agg(
            전체_행수=("환각 발생 여부", "count"),
            환각_행수=("환각 발생 여부", lambda x: (x == 1).sum()),
        )
        .reset_index()
    )
    summary["환각_발생률(%)"] = (
        summary["환각_행수"] / summary["전체_행수"] * 100
    ).round(2)

    # 막대그래프 (Plotly)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=summary["토큰 구간"],
            y=summary["환각_발생률(%)"],
            text=summary["환각_발생률(%)"].apply(lambda x: f"{x}%"),
            textposition="outside",
            marker_color="steelblue",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "전체 행수: %{customdata[0]}<br>"
                "환각 행수: %{customdata[1]}<br>"
                "발생률: %{y}%<extra></extra>"
            ),
            customdata=summary[["전체_행수", "환각_행수"]].values,
        )
    )
    fig.update_layout(
        xaxis_title=token_col,
        yaxis_title="환각 발생률 (%)",
        yaxis_range=[0, max(summary["환각_발생률(%)"].max() * 1.2, 10)],
        template="plotly_white",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    # 구간별 상세표
    st.subheader("📋 구간별 상세 통계")
    st.dataframe(
        summary.rename(
            columns={
                "토큰 구간": "구간",
                "전체_행수": "전체 행수",
                "환각_행수": "환각 행수 (환각 발생 여부=1)",
                "환각_발생률(%)": "환각 발생률 (%)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
