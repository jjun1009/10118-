import os
import numpy as np
import pandas as pd
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
menu = st.sidebar.radio(
    "이동",
    ["서론", "막대그래프", "히스토그램", "히트맵", "히스토그램2"],
    key="menu_radio",
)

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
# 5. 히스토그램 (원래 첫 번째 코드 — 유지)
# ──────────────────────────────────────────────
elif menu == "히스토그램":
    col_sel, _ = st.columns([1, 5])
    with col_sel:
        group_choice = st.radio(
            "그룹 선택",
            ["0 (비환각)", "1 (환각)", "전체 비교 (6개 모두)"],
            horizontal=True,
            key="hist_group_radio",
        )
        attr_choice = st.radio(
            "표시 속성",
            score_cols,
            horizontal=True,
            key="hist_attr_radio",
        )

    n_total_0 = int((df["환각 발생 여부"] == 0).sum())
    n_total_1 = int((df["환각 발생 여부"] == 1).sum())

    def make_hist(data, attr, group_label, n_total):
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=data[attr],
            nbinsx=10,
            histnorm="percent",
            name=group_label,
            marker_color="#4C78A8" if "0" in group_label else "#E45759",
            opacity=0.85,
            text=[f"n = {n_total}"] * len(data),
            textposition="outside",
        ))
        fig.update_layout(
            title=f"{group_label}  ·  {attr}  ·  n = {n_total}",
            xaxis_title=attr,
            yaxis_title="비율 (%)",
            bargap=0.1,
            margin=dict(l=40, r=20, t=50, b=40),
            height=260,
            showlegend=False,
        )
        fig.update_xaxes(range=[0, 1], tickangle=0)
        fig.update_yaxes(ticksuffix="%")
        return fig

    if group_choice == "전체 비교 (6개 모두)":
        ph = [[st.empty() for _ in range(3)] for _ in range(2)]

        for ci, attr in enumerate(score_cols):
            g0 = df[df["환각 발생 여부"] == 0]
            ph[0][ci].write(make_hist(g0, attr, "그룹 0 (비환각)", n_total_0))

            g1 = df[df["환각 발생 여부"] == 1]
            ph[1][ci].write(make_hist(g1, attr, "그룹 1 (환각)", n_total_1))
    else:
        grp = 0 if "0" in group_choice else 1
        grp_label = "그룹 0 (비환각)" if grp == 0 else "그룹 1 (환각)"
        n_total   = n_total_0 if grp == 0 else n_total_1
        sub = df[df["환각 발생 여부"] == grp]
        st.write(make_hist(sub, attr_choice, grp_label, n_total))

# ──────────────────────────────────────────────
# 6. 히트맵  (신규 추가)
#    X: 온도 설정값 / Y: 데이터베이스 유사도 점수
#    셀 값: 해당 범주 내 환각률(%) = 환각행수/전체행수×100
# ──────────────────────────────────────────────
elif menu == "히트맵":
    # ── 사이드바 오른쪽 상단 컨트롤 ──
    col_c, col_g = st.columns([1, 1])
    with col_c:
        color_choice = st.radio(
            "히트맵 색상",
            ["붉은색", "푸른색", "초록색"],
            horizontal=True,
            key="heatmap_color_radio",
        )
    with col_g:
        grid_n = st.radio(
            "히트맵 형태",
            ["5×5", "6×6", "7×7"],
            horizontal=True,
            key="heatmap_grid_radio",
        )

    n_cells = int(grid_n.split("×")[0])   # 5, 6, 7

    # 색상 설정
    if color_choice == "붉은색":
        colorscale = [
            [0,   "#fff5f5"],
            [0.5, "#fca5a5"],
            [1,   "#b91c1c"],
        ]
        cmin, cmax = 0, 100
        colorbar_title = "환각률 (%)"
    elif color_choice == "푸른색":
        colorscale = [
            [0,   "#f0f9ff"],
            [0.5, "#38bdf8"],
            [1,   "#0c4a6e"],
        ]
        cmin, cmax = 0, 100
        colorbar_title = "환각률 (%)"
    else:  # 초록색
        colorscale = [
            [0,   "#f0fdf4"],
            [0.5, "#4ade80"],
            [1,   "#166534"],
        ]
        cmin, cmax = 0, 100
        colorbar_title = "환각률 (%)"

    temp_col = "온도 설정값"
    sim_col  = "데이터베이스 유사도 점수"

    n_total_all = len(df)

    # 구간 경계 계산 (분위수 기준, 중복 제거)
    p = np.linspace(0, 1, n_cells + 1)
    temp_edges = np.unique(np.quantile(df[temp_col], p))
    sim_edges  = np.unique(np.quantile(df[sim_col],  p))

    n_temp_bins = len(temp_edges) - 1
    n_sim_bins  = len(sim_edges)  - 1

    bin_labels_t = [f"Bin {i+1}" for i in range(n_temp_bins)]
    bin_labels_s = [f"Bin {i+1}" for i in range(n_sim_bins)]

    df["temp_bin"] = pd.cut(df[temp_col], bins=temp_edges, labels=bin_labels_t, include_lowest=True)
    df["sim_bin"]  = pd.cut(df[sim_col],  bins=sim_edges,  labels=bin_labels_s, include_lowest=True)

    # 환각률 집계
    agg = (
        df.groupby(["sim_bin", "temp_bin"], observed=True)
        .agg(
            전체_행수=("환각 발생 여부", "count"),
            환각_행수=("환각 발생 여부", lambda x: (x == 1).sum()),
        )
        .reset_index()
    )
    agg["환각률(%)"] = (agg["환각_행수"] / agg["전체_행수"] * 100).round(2)

    heatmap_data = agg.pivot_table(
        index="sim_bin",
        columns="temp_bin",
        values="환각률(%)",
        fill_value=0.0,
    )
    # 정렬: Bin1 → BinN 순서
    heatmap_data = heatmap_data.reindex(index=bin_labels_s, columns=bin_labels_t, fill_value=0.0)

    # count 매트릭스 (hover 용)
    count_data = agg.pivot_table(
        index="sim_bin",
        columns="temp_bin",
        values="전체_행수",
        fill_value=0,
    )
    count_data = count_data.reindex(index=bin_labels_s, columns=bin_labels_t, fill_value=0)

    fig_hm = go.Figure()

    fig_hm.add_trace(
        go.Heatmap(
            z=heatmap_data.values,
            x=bin_labels_t,
            y=bin_labels_s,
            colorscale=colorscale,
            zmin=cmin,
            zmax=cmax,
            colorbar=dict(title=colorbar_title, thickness=15, len=0.8),
            hoverongaps=False,
            hovertemplate=(
                "온도: %{x}<br>"
                "유사도: %{y}<br>"
                "환각률: %{z:.1f}%<br>"
                "<extra></extra>"
            ),
        )
    )

    fig_hm.update_layout(
        title=(
            f"히트맵 — 온도 설정값 × 데이터베이스 유사도 점수  ·  "
            f"환각률(%)  ·  {grid_n}  ·  전체 표본수: {n_total_all}행"
        ),
        xaxis_title=temp_col,
        yaxis_title=sim_col,
        xaxis=dict(
            tickmode="array",
            tickvals=bin_labels_t,
            ticktext=[f" Bin {i+1}" for i in range(n_temp_bins)],
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=bin_labels_s,
            ticktext=[f"Bin {i+1}" for i in range(n_sim_bins)],
            autorange="reversed",   # Bin1 이 위쪽
        ),
        template="plotly_white",
        height=560,
        margin=dict(l=60, r=30, t=80, b=60),
    )

    st.plotly_chart(fig_hm, use_container_width=True)

    # ── 범주 경계값 + 전체 표본수 (그래프 아래) ──
    st.markdown("#### 범주 경계값")

    temp_tbl_rows = []
    sim_tbl_rows  = []
    for i in range(n_temp_bins):
        lo = float(temp_edges[i])
        hi = float(temp_edges[i+1])
        temp_tbl_rows.append(f"온도 Bin {i+1}:  [{lo:.3f}, {hi:.3f})")
    for i in range(n_sim_bins):
        lo = float(sim_edges[i])
        hi = float(sim_edges[i+1])
        sim_tbl_rows.append(f"유사도 Bin {i+1}:  [{lo:.3f}, {hi:.3f})")

    st.caption("■ 온도 설정값 구간")
    for r in temp_tbl_rows:
        st.caption(r)

    st.caption("■ 데이터베이스 유사도 점수 구간")
    for r in sim_tbl_rows:
        st.caption(r)

    st.caption(f"■ 전체 표본수: {n_total_all}행  (환각 발생: {(df['환각 발생 여부']==1).sum()}행 / "
               f"비환각: {(df['환각 발생 여부']==0).sum()}행)")

# ──────────────────────────────────────────────
# 7. 히스토그램2  (문서 기반 — 분리 유지)
# ──────────────────────────────────────────────
elif menu == "히스토그램2":
    st.markdown("### 히스토그램 분석")

    col_chart, col_ctrl = st.columns([2, 1])

    with col_ctrl:
        st.markdown("#### 분석 설정")
        sel_type = st.selectbox(
            "환각 유형",
            ["Fabrication", "Contradiction", "Unrelated"],
            index=0,
            key="hist2_type",
        )
        sel_score = st.selectbox(
            "분석 점수",
            score_cols,
            index=0,
            key="hist2_score",
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
            .reindex(["Fabrication", "Contradiction", "Unrelated"], fill_value=0)
            .reset_index()
        )
        type_counts.columns = ["환각유형", "표본수"]
        type_counts["표본수"] = type_counts["표본수"].astype(int)

        st.dataframe(type_counts, use_container_width=True, hide_index=True)
