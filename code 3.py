import os                          # ← 추가 (CSV 존재 여부 확인용)
import numpy as np                 # ← 상단 이동 (항상 정의됨)
import pandas as pd                # ← 상단 이동 (항상 정의됨)
import streamlit as st
import plotly.graph_objects as go

# ──────────────────────────────────────────────
# 0. 데이터 로딩 · 열 삭제 · 한국어 개명
# ──────────────────────────────────────────────
CSV_PATH = "rag_hallucination.csv"

if not os.path.exists(CSV_PATH):
    np.random.seed(42)
    n = 500
    df_raw = pd.DataFrame({
        "index": range(n),
        "Interaction_ID":             [f"INT-{i:04d}" for i in range(n)],
        "User_Prompt":               [f"질문 {i}" for i in range(n)],
        "Provided_Context":          [f"문맥 {i}" for i in range(n)],
        "LLM_Generated_Response":    [f"응답 {i}" for i in range(n)],
        "LLM_Model_Name":            np.random.choice(
            ["gpt-4","claude-3","gemini-pro","llama-3"], n),
        "Temperature_Setting":       np.round(np.random.uniform(0,2,n), 2),
        "Vector_DB_Similarity_Score": np.round(np.random.uniform(0,1,n), 4),
        "Prompt_Tokens":             np.random.randint(50, 500, n),
        "Response_Tokens":           np.random.randint(20, 300, n),
        "Context_Response_Overlap_Ratio": np.round(np.random.uniform(0,1,n), 4),
        "Entity_Match_Score":        np.round(np.random.uniform(0,1,n), 4),
        "Sentence_Complexity_Index": np.round(np.random.uniform(0,1,n), 4),
        "Subjectivity_Score":        np.round(np.random.uniform(0,1,n), 4),
        "Contradiction_Words_Count": np.random.randint(0, 10, n),
        "Answer_Relevance_Score":    np.round(np.random.uniform(0,1,n), 4),
        "Context_Faithfulness_Score":np.round(np.random.uniform(0,1,n), 4),
        "Hallucination_Type":        np.random.choice(
            ["사실 왜곡","문맥 이탈","근거 부족","없음"], n),
        "Is_Hallucination":          np.random.choice([0,1], n, p=[0.6,0.4]),
    })
    df_raw.to_csv(CSV_PATH, index=False)

df = pd.read_csv(CSV_PATH)

DROP_COLS = [
    "index", "Interaction_ID", "User_Prompt", "Provided_Context",
    "LLM_Generated_Response", "Subjectivity_Score",
    "Context_Response_Overlap_Ratio", "Contradiction_Words_Count",
]
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

RENAME_MAP = {
    "LLM_Model_Name":             "모델 이름",
    "Temperature_Setting":        "온도 설정값",
    "Vector_DB_Similarity_Score": "데이터베이스 유사도 점수",
    "Prompt_Tokens":              "입력 토큰 수",
    "Response_Tokens":            "답변 토큰 수",
    "Entity_Match_Score":         "개체 일치 점수",
    "Sentence_Complexity_Index":  "복잡도 지수",
    "Answer_Relevance_Score":     "답변 관련성 점수",
    "Context_Faithfulness_Score": "문맥 충실도 점수",
    "Hallucination_Type":         "환각 유형",
    "Is_Hallucination":           "환각 발생 여부",
}
df = df.rename(columns=RENAME_MAP)

# ──────────────────────────────────────────────
# 라벨 변수 정의
# ──────────────────────────────────────────────
score_cols = [
    "개체 일치 점수",
    "답변 관련성 점수",
    "문맥 충실도 점수",
]

score_labels = {
    "개체 일치 점수":     "개체 일치 점수",
    "답변 관련성 점수":   "답변 관련성 점수",
    "문맥 충실도 점수":   "문맥 충실도 점수",
}

type_color_map = {
    "Fabrication":   "#1f77b4",
    "Contradiction": "#d62728",
    "Unrelated":     "#2ca02c",
}

# ──────────────────────────────────────────────
# 1. 사이드바 메뉴
# ──────────────────────────────────────────────
st.sidebar.title("메뉴")
menu = st.sidebar.radio("이동", ["서론", "막대그래프", "히스토그램"])

# ──────────────────────────────────────────────
# 2. 공통 제목
# ──────────────────────────────────────────────
st.title("LLM의 환각작용")

# ──────────────────────────────────────────────
# 3. 서론
# ──────────────────────────────────────────────
if menu == "서론":
    st.write("서론 내용은 아직 준비 중입니다.")

# ──────────────────────────────────────────────
# 4. 막대그래프
# ──────────────────────────────────────────────
elif menu == "막대그래프":
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

# ──────────────────────────────────────────────
# 5. 히스토그램
# ──────────────────────────────────────────────
elif menu == "히스토그램":
    st.markdown("### 히스토그램 분석")

    col_chart, col_ctrl = st.columns([2, 1])

    with col_ctrl:
        st.markdown("#### 분석 설정")
        sel_type = st.selectbox(
            "환각 유형",
            ["Fabrication", "Contradiction", "Unrelated"],   # ← 오타 수정
            index=0,
            key="hist_type",
        )
        sel_score = st.selectbox(
            "분석 점수",
            score_cols,
            index=0,
            key="hist_score",
        )

    subset = df[df["환각 유형"] == sel_type]
    y_vals = subset[sel_score].dropna()
    total_n = len(subset)

    if len(y_vals) == 0:
        st.warning(
            f"선택된 환각유형 '{sel_type}'에 점수 '{sel_score}' 데이터가 없습니다."
        )
        st.stop()

    score_kr = score_labels.get(sel_score, sel_score)

    with col_chart:
        st.markdown(
            f"**환각유형:** {sel_type}  |  **분석 점수:** {score_kr}  |  "
            f"**총 표본수:** {total_n}행"
        )

        counts, bin_edges = np.histogram(y_vals, bins=10)
        pct = counts / total_n * 100
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
                    "점수 구간 중심: %{x:.3f}<br>"
                    "비율: %{y:.2f}%<extra></extra>"
                ),
            )
        )

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
                f"(총 표본수: {total_n}행 · 10개 빈 · "
                f"y축: 해당 환각유형 전체표본수 대비 %)"
            ),
            height=420,
            margin=dict(l=50, r=30, t=90, b=55),
            bargap=0.12,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 분포 요약")
        st.caption(
            f"선택 환각유형 **{sel_type}**의 전체 표본수: **{total_n}행**  "
            f"(점수 결측 제외: {len(y_vals)}행)"
        )
        st.caption(
            f"평균 {y_vals.mean():.3f} · 중앙값 {y_vals.median():.3f} · "
            f"표준편차 {y_vals.std():.3f} · 최소 {y_vals.min():.3f} · 최대 {y_vals.max():.3f}"
        )

        st.markdown("### 유형별 표본수")
        type_counts = (
            df["환각 유형"]
            .value_counts()
            .reindex(["Fabrication", "Contradiction", "Unrelated"], fill_value=0)  # ← 오타 수정
            .reset_index()
        )
        type_counts.columns = ["환각유형", "표본수"]
        type_counts["표본수"] = type_counts["표본수"].astype(int)

        st.dataframe(type_counts, use_container_width=True, hide_index=True)
