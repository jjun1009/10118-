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
    ["서론", "막대그래프", "히트맵", "히스토그램", "히스토그램2"],
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
# 4. 막대그래프  (완전 교체)
#    입력 토큰 수 / 복잡도 지수 — 각 8구간 → 구간별 환각률(%)
# ──────────────────────────────────────────────
elif menu == "막대그래프":
    # ── 사이드바 오른쪽 상단: X축 속성 선택 ──
    col_c1, col_c2 = st.columns([1, 1])
    with col_c1:
        x_attr = st.radio(
            "X축 속성",
            ["입력 토큰 수", "복잡도 지수", "둘 다 보기"],
            horizontal=True,
            key="bar_xattr_radio",
        )

    N_BINS = 8
    HALL_COL = "환각 발생 여부"

    def make_bar_graph(col_x, title_x):
        """col_x: 숫자형 열 이름, title_x: 그래프 제목용 문자열"""
        edges = np.quantile(df[col_x], np.linspace(0, 1, N_BINS + 1))
        edges = np.unique(edges)
        n_bins_actual = len(edges) - 1
        labels = [f"Bin {i+1}" for i in range(n_bins_actual)]

        df["_bin"] = pd.cut(df[col_x], bins=edges, labels=labels, include_lowest=True)

        agg = (
            df.groupby("_bin", observed=True)
            .agg(
                전체_행수=(HALL_COL, "count"),
                환각_행수=(HALL_COL, lambda x: (x == 1).sum()),
            )
            .reset_index()
        )
        agg["환각률(%)"] = (agg["환각_행수"] / agg["전체_행수"] * 100).round(2)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=agg["_bin"],
                y=agg["환각률(%)"],
                text=agg["환각률(%)"].apply(lambda v: f"{v}%"),
                textposition="outside",
                marker_color="steelblue",
                hovertemplate=(
                    "%{x}<br>"
                    "환각률: %{y:.2f}%<br>"
                    "전체 행수: %{customdata}행<extra></extra>"
                ),
                customdata=agg["전체_행수"].values,
            )
        )

        fig.update_layout(
            title=f"{title_x}  ·  구간별 환각 발생률 (%)  ·  8구간",
            xaxis_title=title_x,
            yaxis_title="환각 발생률 (%)",
            yaxis_range=[0, max(100, agg["환각률(%)"].max() * 1.15)],
            template="plotly_white",
            height=420,
            margin=dict(l=50, r=30, t=70, b=50),
        )
        return fig, edges, labels

    if x_attr == "입력 토큰 수":
        col_x = "입력 토큰 수"
        title_x = "입력 토큰 수"
        fig1, edges1, labels1 = make_bar_graph(col_x, title_x)
        st.plotly_chart(fig1, use_container_width=True)

        st.caption(f"■ {title_x} 구간 경계값 (8구간):")
        for i in range(len(edges1) - 1):
            st.caption(f"  Bin {i+1}:  [{float(edges1[i]):.2f}, {float(edges1[i+1]):.2f})")
        st.caption(f"■ 전체 표본수: {len(df)}행  "
                   f"(환각: {(df[HALL_COL]==1).sum()}행 / "
                   f"비환각: {(df[HALL_COL]==0).sum()}행)")

    elif x_attr == "복잡도 지수":
        col_x = "복잡도 지수"
        title_x = "복잡도 지수"
        fig2, edges2, labels2 = make_bar_graph(col_x, title_x)
        st.plotly_chart(fig2, use_container_width=True)

        st.caption(f"■ {title_x} 구간 경계값 (8구간):")
        for i in range(len(edges2) - 1):
            st.caption(f"  Bin {i+1}:  [{float(edges2[i]):.4f}, {float(edges2[i+1]):.4f})")
        st.caption(f"■ 전체 표본수: {len(df)}행  "
                   f"(환각: {(df[HALL_COL]==1).sum()}행 / "
                   f"비환각: {(df[HALL_COL]==0).sum()}행)")

    else:  # 둘 다 보기
        col1, col2 = st.columns(2)
        with col1:
            fig1, edges1, _ = make_bar_graph("입력 토큰 수", "입력 토큰 수")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2, edges2, _ = make_bar_graph("복잡도 지수", "복잡도 지수")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### 구간 경계값")
        st.caption("■ 입력 토큰 수 (8구간)")
        for i in range(len(edges1) - 1):
            st.caption(f"  Bin {i+1}:  [{float(edges1[i]):.2f}, {float(edges1[i+1]):.2f})")
        st.caption("■ 복잡도 지수 (8구간)")
        for i in range(len(edges2) - 1):
            st.caption(f"  Bin {i+1}:  [{float(edges2[i]):.4f}, {float(edges2[i+1]):.4f})")
        st.caption(f"■ 전체 표본수: {len(df)}행  "
                   f"(환각: {(df[HALL_COL]==1).sum()}행 / "
                   f"비환각: {(df[HALL_COL]==0).sum()}행)")

# ──────────────────────────────────────────────
# 5. 히트맵  (유지)
# ──────────────────────────────────────────────
elif menu == "히트맵":
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

    n_cells = int(grid_n.split("×")[0])

    if color_choice == "붉은색":
        colorscale = [[0, "#fff5f5"], [0.5, "#fca5a5"], [1, "#b91c1c"]]
    elif color_choice == "푸른색":
        colorscale = [[0, "#f0f9ff"], [0.5, "#38bdf8"], [1, "#0c4a6e"]]
    else:
        colorscale = [[0, "#f0fdf4"], [0.5, "#4ade80"], [1, "#166534"]]

    temp_col = "온도 설정값"
    sim_col  = "데이터베이스 유사도 점수"
    n_total_all = len(df)

    p = np.linspace(0, 1, n_cells + 1)
    temp_edges = np.unique(np.quantile(df[temp_col], p))
    sim_edges  = np.unique(np.quantile(df[sim_col],  p))
    n_temp_bins = len(temp_edges) - 1
    n_sim_bins  = len(sim_edges)  - 1
    bin_labels_t = [f"Bin {i+1}" for i in range(n_temp_bins)]
    bin_labels_s = [f"Bin {i+1}" for i in range(n_sim_bins)]

    df["temp_bin"] = pd.cut(df[temp_col], bins=temp_edges, labels=bin_labels_t, include_lowest=True)
    df["sim_bin"]  = pd.cut(df[sim_col],  bins=sim_edges,  labels=bin_labels_s, include_lowest=True)

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
        index="sim_bin", columns="temp_bin", values="환각률(%)", fill_value=0.0,
    ).reindex(index=bin_labels_s, columns=bin_labels_t, fill_value=0.0)

    fig_hm = go.Figure()
    fig_hm.add_trace(
        go.Heatmap(
            z=heatmap_data.values,
            x=bin_labels_t,
            y=bin_labels_s,
            colorscale=colorscale,
            zmin=0, zmax=100,
            colorbar=dict(title="환각률 (%)", thickness=15, len=0.8),
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
        xaxis=dict(tickmode="array", tickvals=bin_labels_t,
                   ticktext=[f" Bin {i+1}" for i in range(n_temp_bins)]),
        yaxis=dict(tickmode="array", tickvals=bin_labels_s,
                   ticktext=[f"Bin {i+1}" for i in range(n_sim_bins)],
                   autorange="reversed"),
        template="plotly_white",
        height=560,
        margin=dict(l=60, r=30, t=80, b=60),
    )

    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("#### 범주 경계값")
    st.caption("■ 온도 설정값 구간")
    for i in range(n_temp_bins):
        st.caption(f"  온도 Bin {i+1}:  [{float(temp_edges[i]):.3f}, {float(temp_edges[i+1]):.3f})")
    st.caption("■ 데이터베이스 유사도 점수 구간")
    for i in range(n_sim_bins):
        st.caption(f"  유사도 Bin {i+1}:  [{float(sim_edges[i]):.3f}, {float(sim_edges[i+1]):.3f})")
    st.caption(f"■ 전체 표본수: {n_total_all}행  "
               f"(환각 발생: {(df['환각 발생 여부']==1).sum()}행 / "
               f"비환각: {(df['환각 발생 여부']==0).sum()}행)")

# ──────────────────────────────────────────────
# 6. 히스토그램 (원래 첫 번째 코드 — 유지)
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
# 7. 히스토그램2 (문서 기반 — 유지)
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

        fig.update_xaxes(range=[x_min, x_max], title_text=score_kr, tickformat=".3f")
        fig.update_yaxes(title_text="비율(%)", range=[0, max(100, pct.max() * 1.15)])

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
