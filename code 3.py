import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


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
    ["서론", "막대그래프", "산점도", "상자그림", "히스토그램"],
    index=0 if st.session_state.menu == "서론" else (
        1 if st.session_state.menu == "막대그래프" else (
            2 if st.session_state.menu == "산점도"else(
                3 if st.session_state.menu == "상자그림" else 4
            )
        )
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
    1. 왼쪽 메뉴에서 **서론**, **막대그래프**, **산점도**, **히스토그램** 중 선택합니다.
    2. 사이드바에서 **입력 토큰 수 기준** / **답변 토큰 수 기준**을 전환합니다.
    3. 막대그래프 화면: 구간 개수(5/6/7)와 표본수 표시 여부를 조정하고 구간별 환각 발생률을 확인합니다.
    4. 산점도 화면: y축 점수를 선택하고, 온도 구간을 전환하며 구간별 산점도를 확인합니다.
       필요하면 1000개 무작위 샘플링 토글로 데이터 양을 제한할 수 있습니다.
    5. 히스토그램 화면: 문장복잡도 구간 개수(5/6/7)를 선택하고,  
       환각이 발생한 행(Is_Hallucination=1)만 필터링하여 세 점수 중 하나의 분포를 퍼센트 비율로 확인합니다.

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

    temp_bins = pd.cut(
        df["온도 설정값"],
        bins=5,
        labels=[f"온도 구간 {i+1}" for i in range(5)],
        include_lowest=True,
    )
    df["온도 구간"] = temp_bins

    temp_labels = temp_bins.cat.categories.tolist()

    selected_interval = st.sidebar.selectbox(
        "볼 온도 구간 선택",
        options=temp_labels,
        index=0,
        key="interval_selectbox",
    )

    use_sampling = st.sidebar.checkbox(
        "1000개 무작위 샘플링",
        value=False,
        help="선택한 구간의 데이터를 최대 1000개로 무작위 추출합니다. 대용량 데이터에서 렌더링 속도를 높일 때 사용합니다.",
        key="sampling_checkbox",
    )

    st.markdown("### 🌡️ 온도 설정값 구간별 산점도")

    interval_df = df[df["온도 구간"] == selected_interval].copy()

    full_count = len(interval_df)
    if use_sampling and full_count > 1000:
        interval_df = interval_df.sample(n=1000, random_state=42)
        sampled_count = 1000
    else:
        sampled_count = full_count

    st.markdown(f"**{selected_interval}**")
    st.caption(
        f"전체 행수: {full_count}행 · "
        f"표시 행수: {sampled_count}행"
        + (" (1000개 무작위 샘플링 적용)" if use_sampling and full_count > 1000 else "")
    )

    if len(interval_df) == 0:
        st.warning("선택한 구간에 데이터가 없습니다.")
    else:
        df0_t = interval_df[interval_df["환각 발생 여부"] == 0]
        df1_t = interval_df[interval_df["환각 발생 여부"] == 1]

        fig_temp = go.Figure()

        if len(df0_t) > 0:
            fig_temp.add_trace(
                go.Scatter(
                    x=df0_t["복잡도 지수"],
                    y=df0_t[y_col],
                    mode="markers",
                    name="환각 없음 (0)",
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
                    name="환각 발생 (1)",
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
            height=500,
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

        st.plotly_chart(fig_temp, use_container_width=True, key="scatter_main")

        st.subheader("📋 구간 데이터 요약")
        display_df = interval_df[["복잡도 지수", y_col, "환각 발생 여부"]].copy()
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

elif menu == "상자그림":
    st.markdown("### 문장복잡도 지수 구간별 상자그림 (환각 발생 여부 전체)")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        complexity_bins = st.selectbox(
            "문장복잡도 구간 개수",
            options=[5, 6, 7],
            index=1,
            help="문장복잡도 지수를 몇 개의 구간으로 나눌지 선택합니다.",
            key="complexity_bins_selectbox",
        )

        hist_y_options = [
            "개체 일치 점수",
            "답변 관련성 점수",
            "문맥 충실도 점수",
        ]
        hist_y_col = st.selectbox(
            "표시할 속성",
            options=hist_y_options,
            index=0,
            help="상자그림으로 표시할 점수를 선택합니다. 환각 발생 여부(0, 1) 전체 행을 표시합니다.",
            key="hist_y_selectbox",
        )

    # 복잡도 구간으로 나누기 (구간 라벨)
    def assign_complexity_bins(series, bins):
        return pd.cut(
            series,
            bins=bins,
            labels=[f"복잡도 구간 {i+1}" for i in range(bins)],
            include_lowest=True,
        )

    # 전체 데이터 복잡도 구간
    df["복잡도 구간"] = assign_complexity_bins(df["복잡도 지수"], complexity_bins)

    st.markdown(f"### {hist_y_col} — 복잡도 구간별 상자그림 (환각 여부 전체 행)")

    st.caption(
        f"전체 행수: {len(df)}행 · "
        f"환각 미발생(0): {(df['환각 발생 여부'] == 0).sum()}행 · "
        f"환각 발생(1): {(df['환각 발생 여부'] == 1).sum()}행"
    )

    fig_box = go.Figure()

    # 환각 발생 여부 == 1 (빨강)
    df_hall = df[df["환각 발생 여부"] == 1]
    for interval_label in df["복잡도 구간"].cat.categories:
        subset = df_hall[df_hall["복잡도 구간"] == interval_label]
        if len(subset) > 0:
            fig_box.add_trace(
                go.Box(
                    name=str(interval_label),
                    y=subset[hist_y_col],
                    marker_color="crimson",
                    boxpoints="outliers",
                    jitter=0.3,
                    pointpos=-1.8,
                    hovertemplate=(
                        f"{hist_y_col}: %{{y:.3f}}<br>"
                        "구간: " + interval_label + "<br>"
                        "환각 발생 여부: 1<extra></extra>"
                    ),
                )
            )

    # 환각 발생 여부 == 0 (파랑)
    df_nonhall = df[df["환각 발생 여부"] == 0]
    for interval_label in df["복잡도 구간"].cat.categories:
        subset = df_nonhall[df_nonhall["복잡도 구간"] == interval_label]
        if len(subset) > 0:
            fig_box.add_trace(
                go.Box(
                    name=str(interval_label),
                    y=subset[hist_y_col],
                    marker_color="steelblue",
                    boxpoints="outliers",
                    jitter=0.3,
                    pointpos=-1.8,
                    hovertemplate=(
                        f"{hist_y_col}: %{{y:.3f}}<br>"
                        "구간: " + interval_label + "<br>"
                        "환각 발생 여부: 0<extra></extra>"
                    ),
                )
            )

    # y축 범위 계산
    y_min = min(
        df[hist_y_col].min() * 0.95,
        0,
    )
    y_max = df[hist_y_col].max() * 1.05

    fig_box.update_layout(
        xaxis_title="복잡도 구간",
        yaxis_title=hist_y_col,
        template="plotly_white",
        height=450,
        boxmode="group",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.05,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
        yaxis_range=[y_min, y_max],
    )

    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("📊 복잡도 구간·환각 여부별 통계")
    grouped_stats = (
        df.groupby(["복잡도 구간", "환각 발생 여부"], observed=True)[hist_y_col]
        .agg(
            행수=("count"),
            평균=("mean"),
            표준편차=("std"),
            최소값=("min"),
            중앙값=("median"),
            최대값=("max"),
        )
        .reset_index()
    )
    grouped_stats["행수"] = grouped_stats["행수"].astype(int)
    for col_name in ["평균", "표준편차", "최소값", "중앙값", "최대값"]:
        grouped_stats[col_name] = grouped_stats[col_name].round(4)

    grouped_stats = grouped_stats.rename(
        columns={
            "복잡도 구간": "복잡도 구간",
            "환각 발생 여부": "환각 발생 여부",
            "행수": "행수",
            "평균": "평균",
            "표준편차": "표준편차",
            "최소값": "최소값",
            "중앙값": "중앙값",
            "최대값": "최대값",
        }
    )
    grouped_stats = grouped_stats.sort_values(
        ["복잡도 구간", "환각 발생 여부"]
    )

    st.dataframe(
        grouped_stats,
        use_container_width=True,
        hide_index=True,
    )

elif menu == "히스토그램":
    import plotly.graph_objects as go
    import numpy as np

    st.markdown("### 히스토그램 분석")

    # ── 사이드바: 환각유형 + 분석 점수 선택 ──
    with st.sidebar:
        st.markdown("#### 분석 설정")
        sel_type = st.selectbox(
            "환각 유형",
            ["Fabrication", "Contradiction", "Unrelated"],
            index=0,
            key="hist_type",
        )
        sel_score = st.selectbox(
            "분석 점수",
            score_cols,
            index=0,
            key="hist_score",
        )

    # ── 데이터 필터링 ──
    subset = df[df["환각 유형"] == sel_type]
    y_vals = subset[sel_score].dropna()
    total_n = len(subset)  # 선택된 환각유형의 전체 행수(NaN 포함)

    if len(y_vals) == 0:
        st.warning(
            f"선택된 환각유형 '{sel_type}'에 점수 '{sel_score}' 데이터가 없습니다."
        )
        st.stop()

    # ── 표시용 점수 이름(score_labels가 있으면 사용, 없으면 컬럼명) ──
    score_kr = score_labels.get(sel_score, sel_score) if "score_labels" in dir() else sel_score

    # ── 정보 표시 (상단) ──
    st.markdown(
        f"**환각유형:** {sel_type}  |  **분석 점수:** {score_kr}  |  "
        f"**총 표본수:** {total_n}행"
    )

    # ── 10개 빈 수동 계산 (y축: 전체표본수 대비 백분율 %) ──
    counts, bin_edges = np.histogram(y_vals, bins=10)
    pct = counts / total_n * 100          # 전체표본수(total_n) 기준 %
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=pct,
            name=sel_score,
            marker_color=type_color_map.get(sel_type, "#4c78a8"),
            opacity=0.85,
            hovertemplate=(
                "점수 구간: %{x:.3f}<br>"
                "비율: %{y:.2f}%<extra></extra>"
            ),
        )
    )

    # ── 축 범위/레이블 ──
    x_min = float(y_vals.min())
    x_max = float(y_vals.max())

    fig.update_xaxes(
        range=[x_min, x_max],
        title_text=score_kr,
        tickformat=".3f",
    )
    fig.update_yaxes(
        title_text="비율(%)",
        range=[0, max(100, pct.max() * 1.15)],
    )

    fig.update_layout(
        title=(
            f"히스토그램 — {sel_type} · {score_kr}<br>"
            f"(총 표본수: {total_n}행 · 10개 빈 · y축: 해당 환각유형 전체표본수 대비 %)"
        ),
        height=420,
        margin=dict(l=50, r=30, t=90, b=55),
        bargap=0.12,
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── 분포 요약 ──
    st.markdown("### 분포 요약")
    st.caption(
        f"선택 환각유형 **{sel_type}**의 전체 표본수: **{total_n}행**  "
        f"(점수 결측 제외: {len(y_vals)}행)"
    )
    st.caption(
        f"평균 {y_vals.mean():.3f} · 중앙값 {y_vals.median():.3f} · "
        f"표준편차 {y_vals.std():.3f} · 최소 {y_vals.min():.3f} · 최대 {y_vals.max():.3f}"
    )

    # ── 유형별 표본수 요약표 ──
    st.markdown("### 유형별 표본수")
    type_counts = (
        df["환각 유형"]
        .value_counts()
        .reindex(["Fabrication", "Contradiction", "Unrelated"], fill_value=0)
        .reset_index()
    )
    type_counts.columns = ["환각유형", "표본수"]
    type_counts["표본수"] = type_counts["표본수"].astype(int)

    st.dataframe(type_counts, use_container_width=True, hide_index=True)
