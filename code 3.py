import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

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
    3. 막대그래프 화면에서 구간별 환각 발생률을 확인합니다.
    4. 산점도 화면에서 복잡도 지수와 각 점수 간 관계를 색상으로 확인합니다.
    5. 산점도 오른쪽 상단에서 추세선과 상관계수를 켜고 끌 수 있습니다.

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

    bins = pd.cut(
        df[token_col],
        bins=n_bins,
        labels=[f"구간 {i+1}" for i in range(n_bins)],
        include_lowest=True,
    )
    df["토큰 구간"] = bins

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

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=summary["토큰 구간"],
            y=summary["환각_발생률(%)"],
            text=summary["환각_발생률(%)"].apply(lambda x: f"{x}%"),
            textposition="outside",
            marker_color="steelblue",
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
    st.markdown("### 복잡도 지수 vs 점수 — 산점도")

    y_options = [
        "개체 일치 점수",
        "답변 관련성 점수",
        "문맥 충실도 점수",
    ]

    col_left, col_right = st.columns([1, 2])

    with col_left:
        y_col = st.selectbox(
            "y축 속성 선택",
            options=y_options,
            index=0,
            help="산점도의 y축에 표시할 점수를 선택합니다.",
            key="y_axis_selectbox",
        )

    with col_right:
        show_group_0 = st.checkbox(
            "환각 0군 표시 (파란색)",
            value=True,
            help="환각 발생 여부가 0인 그룹을 산점도에 표시합니다.",
            key="show_group0_checkbox",
        )
        show_group_1 = st.checkbox(
            "환각 1군 표시 (빨간색)",
            value=True,
            help="환각 발생 여부가 1인 그룹을 산점도에 표시합니다.",
            key="show_group1_checkbox",
        )
        show_trend_0 = st.checkbox(
            "0군 추세선 표시",
            value=False,
            help="환각 0군 데이터에 선형 추세선을 표시합니다.",
            key="trend0_checkbox",
        )
        show_trend_1 = st.checkbox(
            "1군 추세선 표시",
            value=False,
            help="환각 1군 데이터에 선형 추세선을 표시합니다.",
            key="trend1_checkbox",
        )
        show_corr_0 = st.checkbox(
            "0군 상관계수 표시",
            value=False,
            help="환각 0군의 피어슨 상관계수(r)를 그래프에 표시합니다.",
            key="corr0_checkbox",
        )
        show_corr_1 = st.checkbox(
            "1군 상관계수 표시",
            value=False,
            help="환각 1군의 피어슨 상관계수(r)를 그래프에 표시합니다.",
            key="corr1_checkbox",
        )

    plot_df = df[["복잡도 지수", y_col, "환각 발생 여부"]].dropna()

    fig = go.Figure()

    r0 = None
    r1 = None

    if show_group_0:
        df_0 = plot_df[plot_df["환각 발생 여부"] == 0]
        fig.add_trace(
            go.Scatter(
                x=df_0["복잡도 지수"],
                y=df_0[y_col],
                mode="markers",
                name="환각 없음 (0)",
                marker=dict(
                    color="blue",
                    size=6,
                    opacity=0.6,
                    line=dict(width=0.5, color="white"),
                ),
                hovertemplate=(
                    "복잡도 지수: %{x:.3f}<br>"
                    f"{y_col}: %{{y:.3f}}<br>"
                    "환각 발생 여부: 0<extra></extra>"
                ),
            )
        )
        if show_corr_0 and len(df_0) >= 2:
            r0 = np.corrcoef(
                df_0["복잡도 지수"], df_0[y_col]
            )[0, 1]

        if show_trend_0 and len(df_0) >= 2:
            slope, intercept = np.polyfit(
                df_0["복잡도 지수"], df_0[y_col], 1
            )
            x_sorted = np.sort(df_0["복잡도 지수"].values)
            y_trend = slope * x_sorted + intercept
            fig.add_trace(
                go.Scatter(
                    x=x_sorted,
                    y=y_trend,
                    mode="lines",
                    name="0군 추세선",
                    line=dict(
                        color="blue",
                        width=2,
                        dash="dash",
                    ),
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

    if show_group_1:
        df_1 = plot_df[plot_df["환각 발생 여부"] == 1]
        fig.add_trace(
            go.Scatter(
                x=df_1["복잡도 지수"],
                y=df_1[y_col],
                mode="markers",
                name="환각 발생 (1)",
                marker=dict(
                    color="red",
                    size=6,
                    opacity=0.6,
                    line=dict(width=0.5, color="white"),
                ),
                hovertemplate=(
                    "복잡도 지수: %{x:.3f}<br>"
                    f"{y_col}: %{{y:.3f}}<br>"
                    "환각 발생 여부: 1<extra></extra>"
                ),
            )
        )
        if show_corr_1 and len(df_1) >= 2:
            r1 = np.corrcoef(
                df_1["복잡도 지수"], df_1[y_col]
            )[0, 1]

        if show_trend_1 and len(df_1) >= 2:
            slope, intercept = np.polyfit(
                df_1["복잡도 지수"], df_1[y_col], 1
            )
            x_sorted = np.sort(df_1["복잡도 지수"].values)
            y_trend = slope * x_sorted + intercept
            fig.add_trace(
                go.Scatter(
                    x=x_sorted,
                    y=y_trend,
                    mode="lines",
                    name="1군 추세선",
                    line=dict(
                        color="red",
                        width=2,
                        dash="dash",
                    ),
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

    if r0 is not None and show_corr_0:
        fig.add_annotation(
            text=f"0군 r = {r0:.3f}",
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            bgcolor="white",
            bordercolor="blue",
            borderwidth=1,
            font=dict(color="blue"),
        )

    if r1 is not None and show_corr_1:
        fig.add_annotation(
            text=f"1군 r = {r1:.3f}",
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.90,
            xanchor="right",
            yanchor="top",
            bgcolor="white",
            bordercolor="red",
            borderwidth=1,
            font=dict(color="red"),
        )

    fig.update_layout(
        xaxis_title="복잡도 지수",
        yaxis_title=y_col,
        template="plotly_white",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 산점도 데이터 요약")
    st.dataframe(
        plot_df.rename(
            columns={
                "복잡도 지수": "복잡도 지수",
                y_col: y_col,
                "환각 발생 여부": "환각 발생 여부",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
