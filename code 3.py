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
    ["서론", "막대그래프", "산점도", "히스토그램", "상자그림"],
    index=0 if st.session_state.menu == "서론" else (
        1 if st.session_state.menu == "막대그래프" else (
            2 if st.session_state.menu == "산점도"else(
                3 if st.session_state.menu == "히스토그램" else 4
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

elif menu == "히스토그램":
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
elif menu == "상자그림":
    st.markdown("### 환각유형별·점수별 상자그림 (3 × 3)")

    # ── 세션스테이트 초기화 (처음 실행 시) ──
    if "box_hallucination_type" not in st.session_state:
        st.session_state["box_hallucination_type"] = "전체"
    if "box_show_stats" not in st.session_state:
        st.session_state["box_show_stats"] = True

    hallucination_type_options = [
        "전체",
        "Fabrication",
        "Contradiction",
        "Unrelated",
    ]

    # ── 사이드바: 환각유형 선택 + 통계 표시 토글 ──
    with st.sidebar:
        st.markdown("#### 상자그림 설정")
        box_hallucination_type = st.selectbox(
            "환각유형 값",
            options=hallucination_type_options,
            index=0,
            help="표시할 환각유형을 선택합니다. '전체' 선택 시 Fabrication·Contradiction·Unrelated 3행 모두 표시합니다.",
            key="box_hallucination_type_selectbox",
        )
        st.session_state["box_hallucination_type"] = box_hallucination_type

        box_show_stats = st.checkbox(
            "Q1·Q3·중앙값 표시 (데이터 점 포함)",
            value=True,
            help="토글 ON: Q1·Q3·중앙값 + 데이터 점(모든 표본) + 평균선 표시. "
                 "토글 OFF: Q1·Q3·중앙값 구조만 표시(데이터 점 숨김).",
            key="box_show_stats_checkbox",
        )
        st.session_state["box_show_stats"] = box_show_stats

    # ── 오른쪽 상단: 선택 정보 + 토글 상태 ──
    info_box = st.container()
    with info_box:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(
                f"**선택 환각유형:** `{box_hallucination_type}`"
            )
        with c2:
            st.markdown(
                f"**통계 요소 표시:** "
                f"{'ON (Q1·Q3·중앙값 + 데이터 점 + 평균선)' if box_show_stats else 'OFF (구조만)'}"
            )

    # ── 점수 3종 정의 ──
    score_cols = [
        "개체 일치 점수",
        "답변 관련성 점수",
        "문맥 충실도 점수",
    ]
    score_labels = [
        "개체 일치 점수",
        "답변 관련성 점수",
        "문맥 충실도 점수",
    ]

    # ── 환각유형 종류(전체일 때 사용) ──
    hallucination_types = [
        "Fabrication",
        "Contradiction",
        "Unrelated",
    ]

    # ── 표시 대상 데이터 ──
    if box_hallucination_type == "전체":
        df_plot = df.copy()
        types_to_plot = hallucination_types
        n_rows = 3
    else:
        df_plot = df[df["환각 유형"] == box_hallucination_type].copy()
        types_to_plot = [box_hallucination_type]
        n_rows = 1

    # y축 범위 통일을 위한 각 점수별 전체 데이터 기준 최소·최대
    y_range_per_score = {}
    for sc in score_cols:
        y_min_s = max(df[sc].min() * 0.95, 0)
        y_max_s = df[sc].max() * 1.05
        y_range_per_score[sc] = (y_min_s, y_max_s)

    # ── 3×3 서브플롯 생성 ──
    fig = make_subplots(
        rows=n_rows,
        cols=3,
        subplot_titles=[
            f"{ht} · ({sl})" for ht in types_to_plot for sl in score_labels
        ],
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )

    # 통계 표시 모드 결정
    if box_show_stats:
        boxmean_mode = "sd"          # 평균+표준편차 범위 표시
        boxpoints_mode = "all"       # 모든 데이터 점 표시
    else:
        boxmean_mode = False          # 평균선 미표시
        boxpoints_mode = "none"       # 데이터 점 숨김 (구조만)

    row_idx = 1
    for ht in types_to_plot:
        col_idx = 1
        for sc in score_cols:
            subset = df_plot[df_plot["환각 유형"] == ht]
            if len(subset) == 0:
                # 데이터가 없으면 빈 trace 대신 공백 처리용 더미 추가
                fig.add_trace(
                    go.Scatter(
                        x=[],
                        y=[],
                        mode="lines",
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row_idx,
                    col=col_idx,
                )
            else:
                fig.add_trace(
                    go.Box(
                        x=subset["복잡도 구간"].astype(str),
                        y=subset[sc],
                        name=f"{ht}",
                        boxmean=boxmean_mode,
                        boxpoints=boxpoints_mode,
                        jitter=0.3,
                        pointpos=-1.8,
                        marker_color=(
                            "#1f77b4" if ht == "Fabrication"
                            else "#d62728" if ht == "Contradiction"
                            else "#2ca02c"
                        ),
                        hovertemplate=(
                            f"점수: %{{y:.3f}}<br>"
                            f"구간: %{{x}}<br>"
                            f"환각유형: {ht}<br>"
                            f"{sc}: %{{y:.3f}}<extra></extra>"
                        ),
                    ),
                    row=row_idx,
                    col=col_idx,
                )

                # y축 범위 통일
                y_min_s, y_max_s = y_range_per_score[sc]
                fig.update_yaxes(
                    range=[y_min_s, y_max_s],
                    row=row_idx,
                    col=col_idx,
                    title_text="",
                )

                # x축 제목 제거(subplot_titles에 정보 포함)
                fig.update_xaxes(
                    title_text="",
                    row=row_idx,
                    col=col_idx,
                )

            col_idx += 1
        row_idx += 1

    # ── 레이아웃 마무리 ──
    row_height = 280 if n_rows == 3 else 360
    fig.update_layout(
        height=300 * n_rows + 80,
        width=1100,
        showlegend=(n_rows == 1 and len(types_to_plot) > 0),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.08,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=40, r=30, t=60, b=50),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── 총 표본수 + 환각유형별 표본수 ──
    st.markdown("### 표본수 요약")
    total_n = len(df_plot)
    st.caption(f"**전체 표본수: {total_n}행**")

    type_counts = (
        df_plot["환각 유형"]
        .value_counts()
        .reindex(hallucination_types, fill_value=0)
        .reset_index()
    )
    type_counts.columns = ["환각유형", "표본수"]
    type_counts["표본수"] = type_counts["표본수"].astype(int)

    st.dataframe(
        type_counts,
        use_container_width=True,
        hide_index=True,
    )    
