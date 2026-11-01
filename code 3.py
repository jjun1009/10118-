import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ──────────────────────────────────────────────
# 0. 데이터 로딩 · 열 선택 · 한국어 개명
# ──────────────────────────────────────────────
CSV_PATH = "rag_hallucination.csv"

KEEP_COLS = [
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
    "Subjectivity_Score",
]

keep_map = {
    "LLM_Model_Name":             "모델 이름",
    "Temperature_Setting":        "온도 설정값",
    "Vector_DB_Similarity_Score": "데이터베이스 유사도 점수",
    "Prompt_Tokens":              "입력 토큰 수",
    "Response_Tokens":            "답변 토큰 수",
    "Entity_Match_Score":         "개체 일치 점수",
    "Sentence_Complexity_Index":  "문장 복잡도 지수",
    "Answer_Relevance_Score":     "답변 관련성 점수",
    "Context_Faithfulness_Score": "문맥 충실도 점수",
    "Hallucination_Type":         "환각 유형",
    "Is_Hallucination":           "환각 발생 여부",
    "Subjectivity_Score":         "주관성 점수",
}

def _generate_example(n=500):
    np.random.seed(42)
    raw = pd.DataFrame({
        "LLM_Model_Name":            np.random.choice(
            ["gpt-4","claude-3","gemini-pro","llama-3"], n),
        "Temperature_Setting":       np.round(np.random.uniform(0, 2, n), 2),
        "Vector_DB_Similarity_Score": np.round(np.random.uniform(0, 1, n), 4),
        "Prompt_Tokens":             np.random.randint(50, 500, n),
        "Response_Tokens":           np.random.randint(20, 300, n),
        "Entity_Match_Score":        np.round(np.random.uniform(0, 1, n), 4),
        "Sentence_Complexity_Index": np.round(np.random.uniform(0, 1, n), 4),
        "Answer_Relevance_Score":    np.round(np.random.uniform(0, 1, n), 4),
        "Context_Faithfulness_Score":np.round(np.random.uniform(0, 1, n), 4),
        "Hallucination_Type":        np.random.choice(
            ["사실 왜곡","문맥 이탈","근거 부족","없음"], n),
        "Is_Hallucination":          np.random.choice([0, 1], n, p=[0.6, 0.4]),
        "Subjectivity_Score":        np.round(np.random.uniform(0, 1, n), 4),
    })
    raw.to_csv(CSV_PATH, index=False)
    return raw.rename(columns=keep_map)

try:
    df_raw = pd.read_csv(CSV_PATH)
    missing = [c for c in KEEP_COLS if c not in df_raw.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    df = df_raw[KEEP_COLS].rename(columns=keep_map)
except FileNotFoundError:
    st.warning("CSV 파일을 찾을 수 없습니다. 예시 데이터로 대체합니다.")
    df = _generate_example()
except KeyError as e:
    st.warning(f"CSV에 필요한 열이 부족합니다: {e}. 예시 데이터로 대체합니다.")
    df = _generate_example()

# ──────────────────────────────────────────────
# 파생 열: 차이 절댓값 3종
# ──────────────────────────────────────────────
df["개체-문맥"]             = (df["개체 일치 점수"] - df["문맥 충실도 점수"]).abs()
df["개체-답변관련성"]       = (df["개체 일치 점수"] - df["답변 관련성 점수"]).abs()
df["답변관련성-문맥충실도"]  = (df["답변 관련성 점수"] - df["문맥 충실도 점수"]).abs()

# ──────────────────────────────────────────────
# 라벨 변수
# ──────────────────────────────────────────────
score_cols = [
    "개체 일치 점수",
    "답변 관련성 점수",
    "문맥 충실도 점수",
]

score_labels = {
    "개체 일치 점수":   "개체 일치 점수",
    "답변 관련성 점수": "답변 관련성 점수",
    "문맥 충실도 점수": "문맥 충실도 점수",
}

# ──────────────────────────────────────────────
# 세션 상태 헬퍼
# ──────────────────────────────────────────────
def _init_session(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

def _cycle(arr, key, delta):
    n = len(arr)
    cur = st.session_state[key]
    nxt = (cur + delta) % n
    st.session_state[key] = nxt
    return arr[nxt]

def _cycle_int(min_val, max_val, key, delta):
    cur = st.session_state[key]
    nxt = cur + delta
    if nxt < min_val:
        nxt = max_val
    elif nxt > max_val:
        nxt = min_val
    st.session_state[key] = nxt
    return nxt

# ──────────────────────────────────────────────
# 1. 페이지 설정
# ──────────────────────────────────────────────
st.set_page_config(page_title="LLM 환각 분석", layout="wide")

# ──────────────────────────────────────────────
# 2. 사이드바 메뉴  (서론 → 막대그래프 → 히트맵 → 히스토그램1)
# ──────────────────────────────────────────────
st.sidebar.title("메뉴")
menu = st.sidebar.radio(
    "이동",
    ["서론", "막대그래프", "히트맵", "히스토그램1"],
    key="menu_radio",
)

# ──────────────────────────────────────────────
# 3. 공통 제목
# ──────────────────────────────────────────────
st.title("LLM의 환각작용")

# ============================================================
# 4. 서론
# ============================================================
if menu == "서론":
    st.title("LLM의 환각 작용은 어떤 조건과 관련도가 높은가?")

    st.markdown("---")

    st.markdown("### 데이터 설명")
    st.markdown(
        "- **데이터 구조**: RAG 환경 바탕 합성 데이터 30,000건 "
        "(GPT-4o, Claude-3.5-Sonnet, Mistral-7B, Llama-3-70B 모델 포함)"
    )
    st.markdown("#### 데이터 속성")
    st.markdown("- **입력 토큰 수**: 프롬프트에 입력된 토큰 수")
    st.markdown("- **문장 복잡도 지수**: 답변을 구성하는 구조, 언어적 복잡성")
    st.markdown(
        "- **데이터베이스 유사도 점수**: 벡터 데이터베이스에서 검색된 자료와 "
        "검색 기준 사이의 유사도"
    )
    st.markdown(
        "- **개체 일치 점수**: 답변의 개체가 제공된 문맥과 일치하는 정도"
    )
    st.markdown(
        "- **답변 관련성 점수**: 답변이 질문과 관련 있고 의도에 부합하는 정도"
    )
    st.markdown(
        "- **문맥 충실도 점수**: 답변이 제공된 문맥에 근거하는 정도"
    )
    st.markdown(
        "- **주관성 점수**: 답변에 나타나는 주관성의 정도"
    )
    st.markdown("- **환각 발생 여부**: 없으면 0, 있으면 1로 표시")
    st.warning(
        "실제 실험이 아닌 합성 데이터이고 4개의 모델에 한정되어 있으므로 "
        "분석 결과를 보편적인 LLM으로 해석하지 마십시오."
    )
    st.markdown(
        "[원본 데이터셋 (HuggingFace)]"
        "(https://huggingface.co/datasets/vkshdev/rag-hallucination-benchmark)"
    )

    st.markdown("---")

    st.markdown("### 작은 질문")
    st.markdown(
        "- **입력 토큰 수, 문장 복잡도, 데이터베이스 유사도의 수치에 따라 "
        "환각률은 어떻게 나타나는가?** "
        "→ [막대그래프]에서 확인할 수 있습니다."
    )
    st.markdown(
        "- **데이터베이스 유사도와 온도 설정값에 따라 환각률은 어떻게 변화하는가?** "
        "→ [히트맵]에서 확인할 수 있습니다."
    )
    st.markdown(
        "- **환각 발생 여부가 0, 1일 때 각각 개체 일치도, 답변 관련성, "
        "문맥 충실도 점수는 어떻게 분포하는가?** "
        "→ [히스토그램1]에서 확인할 수 있습니다."
    )

    st.markdown("---")

    st.markdown(
        f"### 분석 대상 표본 수: **{len(df)}행**"
    )


# ============================================================
# 5. 막대그래프
# ============================================================
elif menu == "막대그래프":
    x_options_bar = [
        "입력 토큰 수",
        "문장 복잡도 지수",
        "주관성 점수",
    ]
    _init_session("bar_xidx", 0)
    _init_session("bar_n", 8)

    c0, c1, c2, c3, c4, c5 = st.columns([1, 2, 1, 1, 1.5, 1])
    with c0:
        if st.button("◀", key="bar_x_left"):
            _cycle(x_options_bar, "bar_xidx", -1)
    with c1:
        st.caption(x_options_bar[st.session_state["bar_xidx"]])
    with c2:
        if st.button("▶", key="bar_x_right"):
            _cycle(x_options_bar, "bar_xidx", 1)
    with c3:
        if st.button("−", key="bar_n_minus"):
            _cycle_int(1, 10, "bar_n", -1)
    with c4:
        st.caption(f"{st.session_state['bar_n']}개")
    with c5:
        if st.button("+", key="bar_n_plus"):
            _cycle_int(1, 10, "bar_n", 1)

    N_BINS   = st.session_state["bar_n"]
    HALL_COL = "환각 발생 여부"
    x_attr   = x_options_bar[st.session_state["bar_xidx"]]

    col_x_map = {
        "입력 토큰 수":     "입력 토큰 수",
        "문장 복잡도 지수": "문장 복잡도 지수",
        "주관성 점수":     "주관성 점수",
    }
    col_x   = col_x_map[x_attr]
    title_x = x_attr

    def make_bar_graph(col_x, title_x):
        edges_full = np.sort(np.unique(np.concatenate([
            [df[col_x].min() - 1e-9],
            np.quantile(df[col_x], np.linspace(0, 1, N_BINS + 1)),
            [df[col_x].max() + 1e-9],
        ])))
        n_actual = len(edges_full) - 1
        labels    = [f"Bin {i+1}" for i in range(n_actual)]

        df["_bin"] = pd.cut(
            df[col_x], bins=edges_full,
            labels=labels, include_lowest=True,
        )

        agg = (
            df.groupby("_bin", observed=True)
            .agg(
                전체_행수=(HALL_COL, "count"),
                환각_행수=(HALL_COL, lambda x: (x == 1).sum()),
            )
            .reset_index()
        )
        agg["환각률(%)"] = agg.apply(
            lambda r: round(r["환각_행수"] / r["전체_행수"] * 100, 2)
            if r["전체_행수"] > 0 else np.nan,
            axis=1,
        )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=agg["_bin"].astype(str).tolist(),
                y=agg["환각률(%)"].astype(float).tolist(),
                text=agg["환각률(%)"].apply(
                    lambda v: f"{v}%" if not np.isnan(v) else "—"
                ).astype(str).tolist(),
                textposition="outside",
                marker_color="steelblue",
                hovertemplate=(
                    "%{x}<br>"
                    "환각률: %{y:.2f}%<br>"
                    "전체 행수: %{customdata}행<extra></extra>"
                ),
                customdata=agg["전체_행수"].astype(int).tolist(),
            )
        )

        fig.update_layout(
            title=f"{title_x}  ·  구간별 환각 발생률 (%)  ·  {N_BINS}구간",
            xaxis_title=title_x,
            yaxis_title="환각 발생률 (%)",
            yaxis_range=[0, max(100, float(agg["환각률(%)"].max()) * 1.15)],
            template="plotly_white",
            height=420,
            margin=dict(l=50, r=30, t=70, b=50),
        )
        return fig, edges_full

    fig, edges_full = make_bar_graph(col_x, title_x)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 결론")
    st.markdown(
        "> **입력토큰수에 따라 환각률은 고르고, 문장복잡도에 따라서는 뚜렷한 "
        "결과가 보이지 않으며, 데이터베이스유사도에 따라선 x축값이 증가함에 따라 "
        "환각률이 낮아지는 경향이 두드러진다.**"
    )

    st.markdown("#### 구간 경계값")
    st.caption(f"■ {title_x} 구간 경계값 ({N_BINS}구간):")
    n_show = len(edges_full) - 1
    for i in range(n_show):
        lo = float(edges_full[i])
        hi = float(edges_full[i+1])
        if "유사도" in title_x or "복잡도" in title_x or "주관성" in title_x:
            fmt = ".4f"
        else:
            fmt = ".2f"
        st.caption(f"  Bin {i+1}:  [{lo:{fmt}}, {hi:{fmt}})")
    st.caption(
        f"■ 전체 표본수: {len(df)}행  "
        f"(환각: {(df[HALL_COL]==1).sum()}행 / "
        f"비환각: {(df[HALL_COL]==0).sum()}행)"
    )


# ============================================================
# 6. 히트맵
# ============================================================
elif menu == "히트맵":
    hm_colors = ["붉은색", "푸른색", "초록색"]
    _init_session("hm_n", 6)
    _init_session("hm_ci", 0)

    c0, c1, c2, c3, c4, c5 = st.columns([1, 1.2, 1, 1, 1.5, 1])
    with c0:
        if st.button("◀", key="hm_n_left"):
            _cycle_int(5, 15, "hm_n", -1)
    with c1:
        st.caption(f"{st.session_state['hm_n']}×{st.session_state['hm_n']}")
    with c2:
        if st.button("▶", key="hm_n_right"):
            _cycle_int(5, 15, "hm_n", 1)
    with c3:
        if st.button("◀", key="hm_c_left"):
            _cycle(hm_colors, "hm_ci", -1)
    with c4:
        st.caption(hm_colors[st.session_state["hm_ci"]])
    with c5:
        if st.button("▶", key="hm_c_right"):
            _cycle(hm_colors, "hm_ci", 1)

    n_cells = st.session_state["hm_n"]
    color_choice = hm_colors[st.session_state["hm_ci"]]

    if color_choice == "붉은색":
        colorscale = [[0, "#fff5f5"], [0.5, "#fca5a5"], [1, "#b91c1c"]]
    elif color_choice == "푸른색":
        colorscale = [[0, "#f0f9ff"], [0.5, "#38bdf8"], [1, "#0c4a6e"]]
    else:
        colorscale = [[0, "#f0fdf4"], [0.5, "#4ade80"], [1, "#166534"]]

    temp_col = "온도 설정값"
    sim_col  = "데이터베이스 유사도 점수"
    n_total_all  = len(df)
    n_hall_total = int((df["환각 발생 여부"] == 1).sum())

    bin_labels_t = [f"Bin {i+1}" for i in range(n_cells)]
    bin_labels_s = [f"Bin {i+1}" for i in range(n_cells)]

    df["temp_bin"] = pd.cut(
        df[temp_col], bins=n_cells,
        labels=bin_labels_t, include_lowest=True,
    )
    df["sim_bin"] = pd.cut(
        df[sim_col], bins=n_cells,
        labels=bin_labels_s, include_lowest=True,
    )

    agg = (
        df.groupby(["sim_bin", "temp_bin"], observed=True)
        .agg(
            전체_행수=("환각 발생 여부", "count"),
            환각_행수=("환각 발생 여부", lambda x: (x == 1).sum()),
        )
        .reset_index()
    )
    agg["환각률(%)"] = agg.apply(
        lambda r: round(r["환각_행수"] / r["전체_행수"] * 100, 2)
        if r["전체_행수"] > 0 else np.nan,
        axis=1,
    )

    hm_data = agg.pivot_table(
        index="sim_bin", columns="temp_bin",
        values="환각률(%)", fill_value=np.nan,
    ).reindex(index=bin_labels_s, columns=bin_labels_t, fill_value=np.nan)
    cnt_data = agg.pivot_table(
        index="sim_bin", columns="temp_bin",
        values="전체_행수", fill_value=0,
    ).reindex(index=bin_labels_s, columns=bin_labels_t, fill_value=0)

    x_max = float(df[temp_col].max())
    x_min = float(df[temp_col].min())
    y_max = float(df[sim_col].max())
    y_min = float(df[sim_col].min())

    fig_hm = go.Figure()
    fig_hm.add_trace(
        go.Heatmap(
            z=hm_data.values,
            x=list(range(n_cells)),
            y=list(range(n_cells)),
            colorscale=colorscale,
            zmin=0, zmax=100,
            colorbar=dict(title="환각률 (%)", thickness=15, len=0.8),
            hoverongaps=False,
            hovertemplate=(
                "온도 구간: %{x}<br>"
                "유사도 구간: %{y}<br>"
                "환각률: %{z:.1f}%<extra></extra>"
            ),
        )
    )

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
            tickvals=list(range(n_cells)),
            ticktext=bin_labels_t,
            showspikes=False,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(n_cells)),
            ticktext=bin_labels_s,
            autorange="reversed",
            showspikes=False,
        ),
        template="plotly_white",
        height=560,
        margin=dict(l=70, r=30, t=80, b=70),
    )

    fig_hm.add_annotation(
        x=-0.04, y=-0.10,
        text="0",
        showarrow=False,
        xref="paper", yref="paper",
        font=dict(size=11),
    )
    fig_hm.add_annotation(
        x=1.04, y=-0.10,
        text=str(round(x_max, 3)),
        showarrow=False,
        xref="paper", yref="paper",
        font=dict(size=11),
    )
    fig_hm.add_annotation(
        x=-0.08, y=-0.10,
        text=str(round(y_max, 3)),
        showarrow=False,
        xref="paper", yref="paper",
        font=dict(size=11),
    )

    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("#### 결론")
    st.markdown(
        "> **데이터베이스와 유사할 수록 환각률이 낮고, 답변의 무작위성을 높게 "
        "설정할 수록 환각률이 높게 나타난다.**"
    )

    st.markdown("#### 칸별 표본수")
    st.caption(
        "※ 히트맵과 동일한 격자 구조이며, 각 셀에는 해당 범위 내 전체 행수를 표시합니다."
    )

    cnt_df = pd.DataFrame(cnt_data.values,
                          index=bin_labels_s,
                          columns=bin_labels_t)
    cnt_df.index.name = "유사도 \\ 온도"
    st.dataframe(cnt_df, use_container_width=True, hide_index=True)
    st.caption(
        f"※ 전체 표본수: {n_total_all}행  "
        f"(환각: {n_hall_total}행 / 비환각: {n_total_all - n_hall_total}행)"
    )

    st.markdown("#### 축 양끝값")
    st.caption(f"■ 온도 설정값 (X축):  {x_min:.3f}  ~  {x_max:.3f}")
    st.caption(
        f"■ 데이터베이스 유사도 점수 (Y축, 아래=작은값):  {y_min:.4f}  ~  {y_max:.4f}"
    )


# ============================================================
# 7. 히스토그램1  (개별 보기 + 겹침 보기)
#    겹침 보기: 세 속성 각각 [min, max] → [0, 1] 정규화 후 동일 축 겹침
# ============================================================
elif menu == "히스토그램1":
    _init_session("hist1_attr_idx", 0)
    _init_session("hist1_grp_idx", 0)
    _init_session("hist1_overlay", False)
    grp_options = ["0 (비환각)", "1 (환각)"]

    c0, c1, c2, c3, c4, c5, c6 = st.columns([1, 2.5, 1, 1, 2, 1, 1.6])
    with c0:
        if st.button("◀", key="h1_attr_left"):
            _cycle(score_cols, "hist1_attr_idx", -1)
    with c1:
        st.caption(score_cols[st.session_state["hist1_attr_idx"]])
    with c2:
        if st.button("▶", key="h1_attr_right"):
            _cycle(score_cols, "hist1_attr_idx", 1)
    with c3:
        if st.button("◀", key="h1_grp_left"):
            _cycle(grp_options, "hist1_grp_idx", -1)
    with c4:
        st.caption(grp_options[st.session_state["hist1_grp_idx"]])
    with c5:
        if st.button("▶", key="h1_grp_right"):
            _cycle(grp_options, "hist1_grp_idx", 1)
    with c6:
        if st.button(
            "겹침 보기" if not st.session_state["hist1_overlay"] else "개별 보기",
            key="h1_overlay_toggle",
        ):
            st.session_state["hist1_overlay"] = not st.session_state["hist1_overlay"]

    overlay_mode = st.session_state["hist1_overlay"]

    n_total_0 = int((df["환각 발생 여부"] == 0).sum())
    n_total_1 = int((df["환각 발생 여부"] == 1).sum())

    if not overlay_mode:
        # ── 개별 보기 (기존 코드) ──
        sel_attr  = score_cols[st.session_state["hist1_attr_idx"]]
        sel_group = grp_options[st.session_state["hist1_grp_idx"]]

        grp       = 0 if "0" in sel_group else 1
        grp_label = "그룹 0 (비환각)" if grp == 0 else "그룹 1 (환각)"
        n_total   = n_total_0 if grp == 0 else n_total_1

        sub = df[df["환각 발생 여부"] == grp]

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=sub[sel_attr].dropna().astype(float),
            nbinsx=10,
            histnorm="percent",
            name=grp_label,
            marker_color="#4C78A8" if grp == 0 else "#E45759",
            opacity=0.85,
            hovertemplate=(
                "%{x:.3f}<br>"
                "비율: %{y:.2f}%<br>"
                "n = " + str(n_total) + "<extra></extra>"
            ),
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

        st.markdown("#### 결론")
        st.markdown(
            "> **환각여부가 0인 그룹과 비교했을때 환각여부가 1인 그룹에서 답변에서 "
            "제공된 문맥에서의 각 개체가 일치하는 정도, 답변이 질문과 관련되어 있는 정도, "
            "답변이 제공된 문맥에 근거하는 정도가 전부 상대적으로 낮은 값에 분포하고 있다.**"
        )

        st.caption(
            f"■ 선택된 그룹: {grp_label}  |  "
            f"총 표본수: {n_total}행  |  "
            f"분석 속성: {sel_attr}"
        )

    else:
        # ── 겹침 보기: 세 속성을 각각 [min, max] → [0, 1] 정규화 후 동일 축 겹침 ──
        sel_group = grp_options[st.session_state["hist1_grp_idx"]]
        grp       = 0 if "0" in sel_group else 1
        grp_label = "그룹 0 (비환각)" if grp == 0 else "그룹 1 (환각)"
        n_total   = n_total_0 if grp == 0 else n_total_1

        sub = df[df["환각 발생 여부"] == grp].copy()

        # 빈 그룹 가드
        if len(sub) == 0:
            st.warning(
                f"선택한 그룹 '{grp_label}'의 데이터가 없습니다. "
                "다른 그룹을 선택하거나 개별 보기로 전환하세요."
            )
        else:
            # 정규화: 각 속성별 [min, max] → [0, 1]
            norm_cols = {}
            for col in score_cols:
                col_s = sub[col].dropna().astype(float)
                vmin = float(col_s.min())
                vmax = float(col_s.max())
                if vmax - vmin > 1e-9:
                    sub[col + "_정규화"] = (sub[col] - vmin) / (vmax - vmin)
                else:
                    sub[col + "_정규화"] = 0.0
                norm_cols[col] = col + "_정규화"

            COLORS = {
                "개체 일치 점수":   "#4C78A8",
                "답변 관련성 점수": "#E45759",
                "문맥 충실도 점수": "#52B95E",
            }
            MARKERS = {
                "개체 일치 점수":   "circle",
                "답변 관련성 점수": "square",
                "문맥 충실도 점수": "diamond",
            }

            fig = go.Figure()
            for col in score_cols:
                x_vals = sub[norm_cols[col]].dropna().astype(float)
                fig.add_trace(go.Histogram(
                    x=x_vals,
                    nbinsx=10,
                    histnorm="percent",
                    name=col,
                    marker_color=COLORS[col],
                    marker_symbol=MARKERS[col],
                    opacity=0.75,
                    hovertemplate=(
                        "%{x:.3f}<br>"
                        "비율: %{y:.2f}%<br>"
                        "n = " + str(n_total) + "<extra></extra>"
                    ),
                ))

            fig.update_layout(
                title=(
                    f"{grp_label}  ·  세 속성 정규화 겹침  ·  "
                    f"n = {n_total}행  ·  (각 속성별 [min, max] → [0, 1])"
                ),
                xaxis_title="정규화 점수 (0 ~ 1)",
                yaxis_title="비율 (%)",
                bargap=0.05,
                barmode="overlay",
                margin=dict(l=50, r=20, t=60, b=50),
                height=400,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                ),
            )
            fig.update_xaxes(range=[0, 1], tickangle=0)
            fig.update_yaxes(ticksuffix="%")

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 결론")
            st.markdown(
                "> **세 속성을 각각 최솟값 0, 최댓값 1로 정규화하여 동일 축에 "
                "겹쳐 본 결과, 환각 그룹(1)과 비환각 그룹(0) 간에 "
                "세 속성의 분포 경향이 서로 다른 위치에서 집중되는 양상을 "
                "확인할 수 있다.**"
            )

            st.caption(
                f"■ 선택된 그룹: {grp_label}  |  "
                f"총 표본수: {n_total}행  |  "
                f"표시: **세 속성 정규화 겹침** (각 속성별 [min, max] → [0, 1])"
            )
