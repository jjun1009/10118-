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
    4. 산점도 화면에서 복잡도 지수와 각 점수 간 관계를 색상으로 확인하고,  
       그룹별 또는 전체 데이터에 대해 추세선과 상관계수를 표시할 수 있습니다.

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

    col_left, col_right = st.columns([1, 1])

    with col_left:
        y_col = st.selectbox(
            "y축 속성 선택",
            options=y_options,
            index=0,
            help="산점도의 y축에 표시할 점수를 선택합니다.",
            key="y_axis_selectbox",
        )
        use_overall = st.checkbox(
            "전체 데이터 모드",
            value=False,
            help="그룹 구분 없이 전체 데이터로 하나의 산점도·추세선·상관계수를 표시합니다.",
            key="overall_mode_checkbox",
        )

    with col_right:
        if not use_overall:
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
                help="환각 0군의 선형 추세선(회귀선)을 표시합니다.",
                key="trend0_checkbox",
            )
            show_trend_1 = st.checkbox(
                "1군 추세선 표시",
                value=False,
                help="환각 1군의 선형 추세선(회귀선)을 표시합니다.",
                key="trend1_checkbox",
            )
            show_corr_0 = st.checkbox(
                "0군 상관계수(r) 표시",
                value=False,
                help="환각 0군의 피어슨 상관계수를 그래프에 텍스트로 표시합니다.",
                key="corr0_checkbox",
            )
            show_corr_1 = st.checkbox(
                "1군 상관계수(r) 표시",
                value=False,
                help="환각 1군의 피어슨 상관계수를 그래프에 텍스트로 표시합니다.",
                key="corr1_checkbox",
            )
        else:
            show_trend_all = st.checkbox(
                "전체 추세선 표시",
                value=False,
                help="전체 데이터의 선형 추세선(회귀선)을 표시합니다.",
                key="trend_all_checkbox",
            )
            show_corr_all = st.checkbox(
                "전체 상관계수(r) 표시",
                value=False,
                help="전체 데이터의 피어슨 상관계수를 그래프에 텍스트로 표시합니다.",
                key="corr_all_checkbox",
            )
            show_group_0 = False
            show_group_1 = False
            show_trend_0 = False
            show_trend_1 = False
            show_corr_0 = False
            show_corr_1 = False

    plot_df = df[["복잡도 지수", y_col, "환각 발생 여부"]].dropna()

    fig = go.Figure()

    def add_trend_and_corr(fig, x_vals, y_vals, label, color, show_trend, show_corr):
        if show_trend and len(x_vals) >= 2:
            slope, intercept = np.polyfit(x_vals, y_vals, 1)
            x_line = np.array([x_vals.min(), x_vals.max()])
            y_line = slope * x_line + intercept
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name=f"{label} 추세선",
                    line=dict(color=color, width=2, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        if show_corr and len(x_vals) >= 2:
            r = np.corrcoef(x_vals, y_vals)[0, 1]
            fig.add_annotation(
                text=f"r = {r:.3f}",
                xref="paper",
                yref="paper",
                x=0.98,
                y=0.02,
                xanchor="right",
                yanchor="bottom",
                showarrow=False,
                font=dict(color=color, size=14),
            )

    if use_overall:
        x_all = plot_df["복잡도 지수"].values
        y_all = plot_df[y_col].values
        fig.add_trace(
            go.Scatter(
                x=x_all,
                y=y_all,
                mode="markers",
                name="전체",
                marker=dict(
                    color="purple",
                    size=6,
                    opacity=0.6,
                    line=dict(width=0.5, color="white"),
                ),
                hovertemplate=(
                    "복잡도 지수: %{x:.3f}<br>"
                    f"{y_col}: %{{y:.3f}}<br>"
                    "환각 발생 여부: %{customdata}<extra></extra>"
                ),
                customdata=plot_df["환각 발생 여부"].values,
            )
        )
        add_trend_and_corr(
            fig, x_all, y_all, "전체", "purple",
            show_trend_all, show_corr_all,
        )
    else:
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
            if len(df_0) > 0:
                add_trend_and_corr(
                    fig,
                    df_0["복잡도 지수"].values,
                    df_0[y_col].values,
                    "환각 없음 (0)",
                    "blue",
                    show_trend_0,
                    show_corr_0,
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
            if len(df_1) > 0:
                add_trend_and_corr(
                    fig,
                    df_1["복잡도 지수"].values,
                    df_1[y_col].values,
                    "환각 발생 (1)",
                    "red",
                    show_trend_1,
                    show_corr_1,
                )

    has_data = len(plot_df) > 0
    if not has_data:
        st.warning("표시할 데이터가 없습니다.")
    else:
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
