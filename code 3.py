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
# 공통 세션 상태 초기화 헬퍼
# ──────────────────────────────────────────────
def _init_session(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

def _cycle(arr, key, delta):
    """arr 내에서 key에 해당하는 인덱스를 delta 만큼 이동 (경계 순환)"""
    n = len(arr)
    cur = st.session_state[key]
    nxt = (cur + delta) % n
    st.session_state[key] = nxt
    return arr[nxt]

def _cycle_int(min_val, max_val, key, delta):
    """min~max 정수 순환"""
    cur = st.session_state[key]
    nxt = cur + delta
    if nxt < min_val:
        nxt = max_val
    elif nxt > max_val:
        nxt = min_val
    st.session_state[key] = nxt
    return nxt

# ──────────────────────────────────────────────
# 1. 사이드바 메뉴  (서론 → 히트맵 → 막대그래프 → 히스토그램1)
# ──────────────────────────────────────────────
st.sidebar.title("메뉴")
menu = st.sidebar.radio(
    "이동",
    ["서론", "히트맵", "막대그래프", "히스토그램1"],
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
# 4. 히트맵
#    Y축(데이터베이스 유사도): 작은 값이 아래
#    행렬수 5×5~15×15, 색상 선택, 축 양끝값 표시,
#    표본수 리스트, 결론칸
# ──────────────────────────────────────────────
elif menu == "히트맵":
    # ── 사이드바 오른쪽 상단 컨트롤 ──
    col1, col2 = st.columns([1, 1])

    # (1) 행렬수 5×5 ~ 15×15  (화살표)
    with col1:
        _init_session("hm_n", 6)
        n_cells = _cycle_int(5, 15, "hm_n", 0)   # 현재 값 읽기 용
        hm_prev = st.button("◀", key="hm_prev")
        hm_next = st.button("▶", key="hm_next")
        if hm_prev:
            n_cells = _cycle_int(5, 15, "hm_n", -1)
        if hm_next:
            n_cells = _cycle_int(5, 15, "hm_n", 1)
        st.caption(f"행렬수: {n_cells}×{n_cells}")

    # (2) 색상 선택 (화살표)
    hm_colors = ["붉은색", "푸른색", "초록색"]
    _init_session("hm_ci", 0)
    with col2:
        if st.button("◀", key="hm_color_left"):
            _cycle(hm_colors, "hm_ci", -1)
        if st.button("▶", key="hm_color_right"):
            _cycle(hm_colors, "hm_ci", 1)
        color_choice = hm_colors[st.session_state["hm_ci"]]
        st.caption(f"색상: {color_choice}")

    # 색상 설정
    if color_choice == "붉은색":
        colorscale = [[0, "#fff5f5"], [0.5, "#fca5a5"], [1, "#b91c1c"]]
    elif color_choice == "푸른색":
        colorscale = [[0, "#f0f9ff"], [0.5, "#38bdf8"], [1, "#0c4a6e"]]
    else:
        colorscale = [[0, "#f0fdf4"], [0.5, "#4ade80"], [1, "#166534"]]

    temp_col     = "온도 설정값"
    sim_col      = "데이터베이스 유사도 점수"
    n_total_all  = len(df)
    n_hall_total = int((df["환각 발생 여부"] == 1).sum())

    p = np.linspace(0, 1, n_cells + 1)
    temp_edges = np.unique(np.quantile(df[temp_col], p))
    sim_edges  = np.unique(np.quantile(df[sim_col],  p))
    n_temp_bins = len(temp_edges) - 1
    n_sim_bins  = len(sim_edges)  - 1

    # 행렬 크기가 실제 구간 수보다 크면 맞춤
    n_tb = max(n_temp_bins, n_cells)
    n_sb = max(n_sim_bins,  n_cells)

    bin_labels_t = [f"Bin {i+1}" for i in range(n_tb)]
    bin_labels_s = [f"Bin {i+1}" for i in range(n_sb)]

    df["temp_bin"] = pd.cut(
        df[temp_col],
        bins=np.sort(np.unique(np.concatenate([
            [df[temp_col].min() - 1e-9],
            temp_edges,
            [df[temp_col].max() + 1e-9],
        ]))),
        labels=bin_labels_t,
        include_lowest=True,
    )
    df["sim_bin"] = pd.cut(
        df[sim_col],
        bins=np.sort(np.unique(np.concatenate([
            [df[sim_col].min() - 1e-9],
            sim_edges,
            [df[sim_col].max() + 1e-9],
        ]))),
        labels=bin_labels_s,
        include_lowest=True,
    )

    agg = (
        df.groupby(["sim_bin", "temp_bin"], observed=True)
        .agg(
            전체_행수=("환각 발생 여부", "count"),
            환각_행수=("환각 발생 여부", lambda x: (x == 1).sum()),
        )
        .reset_index()
    )
    agg["환각률(%)"] = (agg["환각_행수"] / agg["전체_행수"] * 100).round(2)

    hm_data  = agg.pivot_table(
        index="sim_bin", columns="temp_bin",
        values="환각률(%)", fill_value=0.0,
    ).reindex(index=bin_labels_s, columns=bin_labels_t, fill_value=0.0)
    cnt_data = agg.pivot_table(
        index="sim_bin", columns="temp_bin",
        values="전체_행수", fill_value=0,
    ).reindex(index=bin_labels_s, columns=bin_labels_t, fill_value=0)

    fig_hm = go.Figure()
    fig_hm.add_trace(
        go.Heatmap(
            z=hm_data.values,
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

    temp_lo = float(np.min(temp_edges))
    temp_hi = float(np.max(temp_edges))
    sim_lo  = float(np.min(sim_edges))
    sim_hi  = float(np.max(sim_edges))

    fig_hm.update_layout(
        title=(
            f"히트맵 — 온도 설정값 × 데이터베이스 유사도 점수  ·  "
            f"환각률(%)  ·  {n_cells}×{n_cells}  ·  "
            f"전체 표본수: {n_total_all}행"
        ),
        xaxis_title=temp_col,
        yaxis_title=sim_col,
        xaxis=dict(
            tickmode="array",
            tickvals=bin_labels_t,
            ticktext=[f" Bin {i+1}" for i in range(n_tb)],
            range=[0, n_tb - 1],
            annotation=dict(
                x=0, y=-0.18,
                text=f"최소: {temp_lo:.3f}",
                showarrow=False, xanchor="left",
                font=dict(size=11, color="gray"),
            ),
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=bin_labels_s,
            ticktext=[f"Bin {i+1}" for i in range(n_sb)],
            autorange="reversed",   # 작은 값이 아래
            range=[0, n_sb - 1],
            annotation=dict(
                x=-0.12, y=0,
                text=f"최소: {sim_lo:.4f}",
                showarrow=False, yanchor="top",
                font=dict(size=11, color="gray"),
            ),
        ),
        template="plotly_white",
        height=560,
        margin=dict(l=70, r=30, t=80, b=70),
    )

    # 축 양끝값 메모 (레이아웃 annotation 으로도 표시)
    fig_hm.add_annotation(
        x=n_tb - 1, y=-0.18,
        text=f"최대: {temp_hi:.3f}",
        showarrow=False, xanchor="right",
        font=dict(size=11, color="gray"),
    )
    fig_hm.add_annotation(
        x=-0.12, y=n_sb - 1,
        text=f"최대: {sim_hi:.4f}",
        showarrow=False, yanchor="bottom",
        font=dict(size=11, color="gray"),
    )

    st.plotly_chart(fig_hm, use_container_width=True)

    # ── 결론칸 ──
    st.markdown(
        "#### 결론"
    )
    st.markdown(
        "> **데이터베이스와 유사할 수록 환각률이 낮고, 답변의 무작위성을 높게 설정할 수록 "
        "환각률이 높게 나타난다.**"
    )

    # ── 표본수 리스트 (히트맵과 동일한 정사각형 격자, 색상 없음) ──
    st.markdown("#### 칸별 표본수")
    st.caption("※ 히트맵과 동일한 격자 구조이며, 각 셀에는 해당 범위 내 전체 행수를 표시합니다.")

    # DataFrame 으로 표 형식 표시
    cnt_df = pd.DataFrame(cnt_data.values,
                          index=bin_labels_s,
                          columns=bin_labels_t)
    cnt_df.index.name = "유사도 \\ 온도"
    st.dataframe(cnt_df, use_container_width=True, hide_index=True)
    st.caption(f"※ 전체 표본수: {n_total_all}행  "
               f"(환각: {n_hall_total}행 / 비환각: {n_total_all - n_hall_total}행)")

    # 축 양끝값 텍스트
    st.markdown("#### 축 양끝값")
    st.caption(f"■ 온도 설정값 (X축):  최소 {temp_lo:.3f}  ~  최대 {temp_hi:.3f}")
    st.caption(f"■ 데이터베이스 유사도 점수 (Y축, 아래=작은값):  최소 {sim_lo:.4f}  ~  최대 {sim_hi:.4f}")

# ──────────────────────────────────────────────
# 5. 막대그래프
#    X축: 입력 토큰 수 / 복잡도 지수 / 데이터베이스 유사도 점수
#    구간 개수 1~10 (화살표), X축 속성 (화살표 버튼 압축)
#    결론칸
# ──────────────────────────────────────────────
elif menu == "막대그래프":
    # ── 사이드바 오른쪽 상단 컨트롤 ──
    col1, col2 = st.columns([1, 1])

    # (1) X축 속성 선택 — 버튼 하나로 압축 + 화살표
    x_options_bar = [
        "입력 토큰 수",
        "복잡도 지수",
        "데이터베이스 유사도 점수",
    ]
    _init_session("bar_xidx", 0)
    with col1:
        if st.button("◀", key="bar_x_left"):
            _cycle(x_options_bar, "bar_xidx", -1)
        if st.button("▶", key="bar_x_right"):
            _cycle(x_options_bar, "bar_xidx", 1)
        x_attr = x_options_bar[st.session_state["bar_xidx"]]
        st.caption(f"X축: {x_attr}")

    # (2) 구간 개수 1~10 — 버튼+화살표
    _init_session("bar_n", 8)
    with col2:
        if st.button("−", key="bar_n_minus"):
            _cycle_int(1, 10, "bar_n", -1)
        if st.button("+", key="bar_n_plus"):
            _cycle_int(1, 10, "bar_n", 1)
        n_bins = st.session_state["bar_n"]
        st.caption(f"구간: {n_bins}개")

    N_BINS  = n_bins
    HALL_COL = "환각 발생 여부"

    def make_bar_graph(col_x, title_x):
        edges = np.quantile(df[col_x], np.linspace(0, 1, N_BINS + 1))
        edges = np.unique(edges)
        n_actual = len(edges) - 1
        labels = [f"Bin {i+1}" for i in range(n_actual)]

        df["_bin"] = pd.cut(
            df[col_x],
            bins=np.sort(np.unique(np.concatenate([
                [df[col_x].min() - 1e-9],
                edges,
                [df[col_x].max() + 1e-9],
            ]))),
            labels=labels,
            include_lowest=True,
        )

        agg = (
            df.groupby("_bin", observed=True)
            .agg(
                전체_행수=(HALL_COL, "count"),
                환각_행수=(HALL_COL, lambda x: (x == 1).sum()),
            )
            .reset_index()
        )
        agg["환각률(%)"] = (agg["환각_행수"] / agg["전체_행수"] * 100).round(2)

        n_actual = len(agg)

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
            title=f"{title_x}  ·  구간별 환각 발생률 (%)  ·  {N_BINS}구간",
            xaxis_title=title_x,
            yaxis_title="환각 발생률 (%)",
            yaxis_range=[0, max(100, agg["환각률(%)"].max() * 1.15)],
            template="plotly_white",
            height=420,
            margin=dict(l=50, r=30, t=70, b=50),
        )
        return fig, edges, agg["전체_행수"].values

    col_x   = {
        "입력 토큰 수":             "입력 토큰 수",
        "복잡도 지수":              "복잡도 지수",
        "데이터베이스 유사도 점수": "데이터베이스 유사도 점수",
    }[x_attr]
    title_x = x_attr

    fig, edges, counts = make_bar_graph(col_x, title_x)
    st.plotly_chart(fig, use_container_width=True)

    # ── 결론칸 ──
    st.markdown("#### 결론")
    st.markdown(
        "> **입력토큰수에 따라 환각률은 고르고, 문장복잡도에 따라서는 뚜렷한 결과가 보이지 않으며, "
        "데이터베이스유사도에 따라선 x축값이 증가함에 따라 환각률이 낮아지는 경향이 두드러진다.**"
    )

    # ── 구간 경계값 + 전체 표본수 ──
    st.markdown("#### 구간 경계값")
    st.caption(f"■ {title_x} 구간 경계값 ({N_BINS}구간):")
    for i in range(len(edges) - 1):
        fmt = ".4f" if "유사도" in title_x or "복잡도" in title_x else ".2f"
        st.caption(f"  Bin {i+1}:  [{float(edges[i]):{fmt}}, {float(edges[i+1]):{fmt}})")
    st.caption(f"■ 전체 표본수: {len(df)}행  "
               f"(환각: {(df[HALL_COL]==1).sum()}행 / "
               f"비환각: {(df[HALL_COL]==0).sum()}행)")

# ──────────────────────────────────────────────
# 6. 히스토그램1
#    그룹 0/1 × 3속성 = 6개 히스토그램
#    "전체 비교" 옵션 제거
#    관찰할 속성 · 그룹 선택 각 버튼+화살표 압축
#    결론칸
# ──────────────────────────────────────────────
elif menu == "히스토그램1":
    # ── 사이드바 오른쪽 상단 컨트롤 ──
    col1, col2 = st.columns([1, 1])

    # (1) 관찰할 속성 — 버튼 하나로 압축 + 화살표
    _init_session("hist1_attr_idx", 0)
    with col1:
        if st.button("◀", key="hist1_attr_left"):
            _cycle(score_cols, "hist1_attr_idx", -1)
        if st.button("▶", key="hist1_attr_right"):
            _cycle(score_cols, "hist1_attr_idx", 1)
        sel_attr = score_cols[st.session_state["hist1_attr_idx"]]
        st.caption(f"관찰 속성: {sel_attr}")

    # (2) 그룹 선택 — 버튼 하나로 압축 + 화살표  (0/1 만, 전체삭제)
    _init_session("hist1_grp_idx", 0)
    grp_options = ["0 (비환각)", "1 (환각)"]
    with col2:
        if st.button("◀", key="hist1_grp_left"):
            _cycle(grp_options, "hist1_grp_idx", -1)
        if st.button("▶", key="hist1_grp_right"):
            _cycle(grp_options, "hist1_grp_idx", 1)
        sel_group = grp_options[st.session_state["hist1_grp_idx"]]
        st.caption(f"그룹: {sel_group}")

    n_total_0 = int((df["환각 발생 여부"] == 0).sum())
    n_total_1 = int((df["환각 발생 여부"] == 1).sum())

    grp = 0 if "0" in sel_group else 1
    grp_label = "그룹 0 (비환각)" if grp == 0 else "그룹 1 (환각)"
    n_total   = n_total_0 if grp == 0 else n_total_1

    sub = df[df["환각 발생 여부"] == grp]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=sub[sel_attr],
        nbinsx=10,
        histnorm="percent",
        name=grp_label,
        marker_color="#4C78A8" if grp == 0 else "#E45759",
        opacity=0.85,
        text=[f"n = {n_total}"] * len(sub),
        textposition="outside",
    ))
    fig.update_layout(
        title=f"{grp_label}  ·  {sel_attr}  ·  n = {n_total}",
        xaxis_title=sel_attr,
        yaxis_title="비율 (%)",
        bargap=0.1,
        margin=dict(l=40, r=20, t=50, b=40),
        height=360,
        showlegend=False,
    )
    fig.update_xaxes(range=[0, 1], tickangle=0)
    fig.update_yaxes(ticksuffix="%")

    st.plotly_chart(fig, use_container_width=True)

    # ── 결론칸 ──
    st.markdown("#### 결론")
    st.markdown(
        "> **환각여부가 0인 그룹과 비교했을때 환각여부가 1인 그룹에서 답변에서 제공된 "
        "문맥에서의 각 개체가 일치하는 정도, 답변이 질문과 관련되어 있는 정도, "
        "답변이 제공된 문맥에 근거하는 정도가 전부 상대적으로 낮은 값에 분포하고 있다.**"
    )

    st.caption(f"■ 선택된 그룹: {grp_label}  |  "
               f"총 표본수: {n_total}행  |  "
               f"분석 속성: {sel_attr}")
