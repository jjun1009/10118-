import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="LLM Hallucination Analyzer", layout="wide")

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


@st.cache_data
def load_data():
    df = pd.read_csv("rag_hallucination.csv")
    df = df[keep_columns].rename(columns=rename_map)
    return df


df = load_data()

st.sidebar.title("📊 지표 선택")
version = st.sidebar.radio(
    "막대그래프 버전",
    ["입력 토큰 수 기준", "답변 토큰 수 기준"],
    index=0,
    key="version_radio",
)

if version == "입력 토큰 수 기준":
    token_col = "입력 토큰 수"
else:
    token_col = "답변 토큰 수"

if "menu" not in st.session_state:
    st.session_state.menu = "서론"

menu = st.sidebar.radio(
    "메뉴",
    ["서론", "막대그래프", "산점도"],
    index=0 if st.session_state.menu == "서론" else (
        1 if st.session_state.menu == "막대그래프" else 2
    ),
    label_visibility="collapsed",
    key="menu_radio",
)
st.session_state.menu = menu

st.title("LLM의 할루미네이션")

if menu == "서론":
    st.markdown("""
    ## 📝 개요
    이 앱은 **RAG(Retrieval-Augmented Generation)** 환경에서  
    LLM의 할루미네이션(환각) 발생 양상을 탐색하기 위한 시각화 도구입니다.

    ### 사용 방법
    1. 왼쪽 메뉴에서 **서론**, **막대그래프**, **산점도** 중 선택합니다.
    2. 사이드바에서 **입력 토큰 수 기준** / **답변 토큰 수 기준**을 전환합니다.
    3. 막대그래프 화면: 구간 개수(5/6/7)와 표본수 표시 여부를 조정하고 구간별 환각 발생률을 확인합니다.
    4. 산점도 화면: y축 점수를 선택한 뒤, 온도 설정값 5개 구간별 산점도 5개를 나란히 확인합니다.

    ### 데이터
    - 원본 파일: `rag_hallucination.csv`
    - 주요 지표: 모델 이름, 온도 설정값, 데이터베이스 유사도 점수,  
      입력/답변 토큰 수, 개체 일치 점수, 복잡도 지수,  
      답변 관련성 점수, 문맥 충실도 점수, 환각 유형, 환각 발생 여부
    """)

elif menu == "막대그래프":
    col_left, col_right = st.columns([1, 1])

    with col_left:
        show_counts = st.checkbox(
            "구간별 표본수 표시",
            value=True,
            help="각 구간 하단의 전체 행수와 환각 행수 표시 여부를 전환합니다.",
            key="show_counts_checkbox",
        )

    with col_right:
        n_bins = st.selectbox(
            "구간 개수",
            options=[5, 6, 7],
            index=0,
            help="토큰 범위를 몇 개의 구간으로 나눌지 선택합니다.",
            key="n_bins_selectbox",
        )

    st.markdown(f"### {version} — 구간별 환각 발생률")

    bins_df = pd.cut(
        df[token_col],
        bins=n_bins,
        labels=[f"구간 {i+1}" for i in range(n_bins)],
        include_lowest=True,
    )
    df["토큰 구간"] = bins_df

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

    fig_bar = go.Figure()
    fig_bar.add_trace(
        go.Bar(
            x=summary["토큰 구간"],
            y=summary["환각_발생률(%)"],
            text=summary["환각_발생률(%)"].apply(lambda x: f"{x}%"),
            textposition="outside",
            marker_color="steelblue",
        )
    )
    fig_bar.update_layout(
        xaxis_title=token_col,
        yaxis_title="환각 발생률 (%)",
        yaxis_range=[0, max(summary["환각_발생률(%)"].max() * 1.2, 10)],
        template="plotly_white",
        height=500,
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    if show_counts:
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

elif menu == "산점도":
    y_options = [
        "개체 일치 점수",
        "답변 관련성 점수",
        "문맥 충실도 점수",
    ]

    st.sidebar.markdown("### 📍 산점도 설정")
    y_col = st.sidebar.selectbox(
        "y축 점수 선택",
        options=y_options,
        index=0,
        key="y_axis_selectbox_scatter",
    )

    st.markdown("### 🌡️ 온도 설정값 구간별 산점도 (5개)")

    temp_bins = pd.cut(
        df["온도 설정값"],
        bins=5,
        labels=[f"온도 구간 {i+1}" for i in range(5)],
        include_lowest=True,
    )
    df["온도 구간"] = temp_bins

    temp_summary = (
        df.groupby("온도 구간", observed=False)
        .agg(
            전체_행수=("환각 발생 여부", "count"),
            환각_행수=("환각 발생 여부", lambda x: (x == 1).sum()),
        )
        .reset_index()
    )

    cols = st.columns(5)

    for idx, (_, row) in enumerate(temp_summary.iterrows()):
        interval_label = row["온도 구간"]
        interval_df = df[df["온도 구간"] == interval_label].dropna(
            subset=["복잡도 지수", y_col, "환각 발생 여부"]
        )

        with cols[idx]:
            st.markdown(f"**{interval_label}**")
            st.caption(
                f"전체 {int(row['전체_행수'])}행 · 환각 {int(row['환각_행수'])}행"
            )

            if len(interval_df) == 0:
                st.warning("데이터 없음")
                continue

            fig_temp = go.Figure()

            df0_t = interval_df[interval_df["환각 발생 여부"] == 0]
            df1_t = interval_df[interval_df["환각 발생 여부"] == 1]

            if len(df0_t) > 0:
                fig_temp.add_trace(
                    go.Scatter(
                        x=df0_t["복잡도 지수"],
                        y=df0_t[y_col],
                        mode="markers",
                        name="0",
                        marker=dict(
                            color="blue",
                            size=5,
                            opacity=0.6,
                            line=dict(width=0.5, color="white"),
                        ),
                        hovertemplate=(
                            "복잡도 지수: %{x:.3f}<br>"
                            f"{y_col}: %{{y:.3f}}<br>"
                            "환각: 0<extra></extra>"
                        ),
                    )
                )

            if len(df1_t) > 0:
                fig_temp.add_trace(
                    go.Scatter(
                        x=df1_t["복잡도 지수"],
                        y=df1_t[y_col],
                        mode="markers",
                        name="1",
                        marker=dict(
                            color="red",
                            size=5,
                            opacity=0.6,
                            line=dict(width=0.5, color="white"),
                        ),
                        hovertemplate=(
                            "복잡도 지수: %{x:.3f}<br>"
                            f"{y_col}: %{{y:.3f}}<br>"
                            "환각: 1<extra></extra>"
                        ),
                    )
                )

            fig_temp.update_layout(
                xaxis_title="복잡도 지수",
                yaxis_title=y_col,
                template="plotly_white",
                height=320,
                margin=dict(l=40, r=20, t=30, b=40),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=1.05,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10),
                ),
            )

            st.plotly_chart(fig_temp, use_container_width=True, key=f"temp_fig_{idx}")
