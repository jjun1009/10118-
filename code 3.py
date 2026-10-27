import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(page_title="LLM 할루미네이션 분석", layout="wide")

# 화면 구분용 세션 상태
if "page" not in st.session_state:
    st.session_state.page = "intro"

# 사이드바 - 버전 전환 버튼
st.sidebar.header("📊 그래프 버전")
graph_version = st.sidebar.radio(
    "표시할 막대그래프 버전 선택",
    ["입력 토큰수 기준", "답변 토큰수 기준"]
)

# 컬럼 매핑 (삭제할 컬럼 제외, 이름 변경)
COLUMN_RENAME = {
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

DELETE_COLUMNS = [
    "index", "Interaction_ID", "User_Prompt", "Provided_Context",
    "LLM_Generated_Response", "Subjectivity_Score",
    "Context_Response_Overlap_Ratio", "Contradiction_Words_Count",
]

def load_and_process_data():
    """CSV 로드 → 삭제 컬럼 제거 → 이름 변경 → 반환"""
    df = pd.read_csv("rag-hallucination.csv")
    df = df.drop(columns=DELETE_COLUMNS, errors="ignore")
    df = df.rename(columns=COLUMN_RENAME)
    return df

def compute_hallucination_rate_by_range(df, token_col, label_col):
    """
    토큰수 열을 5개 구간으로 나누고,
    각 구간 내 환각발생여부=1인 행 수 / 전체 행 수 * 100 계산
    """
    df = df.copy()
    df["_range"] = pd.cut(
        df[token_col],
        bins=5,
        labels=[f"구간 {i+1}" for i in range(5)],
        include_lowest=True,
    )
    grouped = df.groupby("_range")[label_col]
    total = grouped.count()
    hallucinated = grouped.sum()  # Is_Hallucination이 0/1이면 sum=환각건수
    rate = (hallucinated / total) * 100
    rate = rate.fillna(0).reset_index()
    rate.columns = ["구간", "환각 발생률 (%)"]
    return rate

def render_bar_chart(df, version):
    """버전 선택에 따라 해당 토큰수 기준 막대그래프 렌더링"""
    if version == "입력 토큰수 기준":
        token_col = "입력 토큰 수"
    else:
        token_col = "답변 토큰 수"

    rate_df = compute_hallucination_rate_by_range(df, token_col, "환각 발생 여부")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=rate_df["구간"],
        y=rate_df["환각 발생률 (%)"],
        text=rate_df["환각 발생률 (%)"].round(2),
        textposition="outside",
        marker_color="#6366f1",
        marker_line_color="#4f46e5",
        marker_line_width=1.5,
    ))
    fig.update_layout(
        title=f"{version} — 구간별 환각 발생률",
        xaxis_title="토큰수 구간",
        yaxis_title="환각 발생률 (%)",
        template="plotly_white",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # 구간별 세부 표
    st.subheader("구간별 상세 통계")
    detail = rate_df.copy()
    detail["환각 발생률 (%)"] = detail["환각 발생률 (%)"].round(2)
    st.dataframe(detail, use_container_width=True, hide_index=True)

# ===== 좌측 메뉴 (레이아웃) =====
col_menu, col_content = st.columns([1, 4], gap="large")

with col_menu:
    st.markdown("### 🧭 메뉴")
    if col_menu.button("📘 서론", use_container_width=True):
        st.session_state.page = "intro"
    if col_menu.button("📊 막대그래프", use_container_width=True):
        st.session_state.page = "chart"
    st.divider()
    st.caption("LLM 할루미네이션 분석 도구")

with col_content:
    # ===== 모든 화면 상단 제목 =====
    st.title("LLM의 할루미네이션")

    if st.session_state.page == "intro":
        st.markdown("""
        ## 📘 서론

        이 앱은 **RAG(Retrieval-Augmented Generation) 환경에서 LLM이 생성한 응답의 환각(hallucination) 발생 양상**을 분석합니다.

        ### 🔍 분석 흐름

        1. **데이터 로딩**  
           `rag-hallucination.csv`를 읽어와 분석에 불필요한 컬럼을 제거하고,  
           남은 컬럼을 한국어로 재명명합니다.

        2. **토큰수 구간 분할**  
           - 입력 토큰 수(Prompt Tokens)  
           - 답변 토큰 수(Response Tokens)  
           각각을 **5개 구간**으로 나눕니다.

        3. **구간별 환각 발생률 계산**  
           각 구간 내에서 `환각 발생 여부 = 1`인 행 수를 해당 구간 전체 행 수로 나누고  
           100을 곱해 **환각 발생률(%)** 을 산출합니다.

        4. **막대그래프 시각화**  
           Plotly를 사용해 구간별 환각 발생률을 막대그래프로 표시합니다.

        ### 🧭 사용 방법

        - **사이드바**에서 입력 토큰수 기준 / 답변 토큰수 기준 그래프를 전환할 수 있습니다.
        - **좌측 메뉴**에서 서론 화면과 막대그래프 화면을 오갈 수 있습니다.
        """)

    elif st.session_state.page == "chart":
        st.markdown("""
        ## 📊 막대그래프 — 구간별 환각 발생률

        현재 선택된 버전: **{version}**

        아래 막대그래프는 선택된 토큰수 열을 5개 구간으로 나누고,  
        각 구간에서 환각이 발생한 비율을 나타냅니다.
        """.format(version=graph_version))

        df = load_and_process_data()
        render_bar_chart(df, graph_version)
