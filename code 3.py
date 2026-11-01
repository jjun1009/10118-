import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────
# 데이터 로드
#    - CSV가 있고 열 이름도 일치하면 실제 데이터 사용
#    - CSV가 없거나 열 이름이 맞지 않으면 예시 데이터 500행 생성
# ─────────────────────────────────────────────────────────────
CSV_PATH = "rag_hallucination.csv"

keep_map = {
    "Temperature": "온도 설정값",
    "Database_Similarity_Score": "데이터베이스 유사도 점수",
    "Subjectivity_Score": "주관성 점수",
    "Context_Response_Overlap_Ratio": "문맥-답변 중첩 비율",
    "Contradiction_Words_Count": "모순 단어 수",
    "Input_Token_Count": "입력 토큰 수",
    "Sentence_Complexity_Index": "문장 복잡도 지수",
    "Entity_Match_Score": "개체 일치 점수",
    "Answer_Relevance_Score": "답변 관련성 점수",
    "Context_Faithfulness_Score": "문맥 충실도 점수",
    "Hallucination_Status": "환각 발생 여부",
}
expected_cols = list(keep_map.keys())

try:
    df_raw = pd.read_csv(CSV_PATH)
    missing = [c for c in expected_cols if c not in df_raw.columns]
    if missing:
        st.warning(
            f"CSV에 필요한 열이 부족합니다: {', '.join(missing)}. "
            "예시 데이터로 대체합니다."
        )
        raise KeyError(f"Missing columns: {missing}")
    df = df_raw[expected_cols].rename(columns=keep_map)
except (FileNotFoundError, KeyError):
    np.random.seed(42)
    n = 500
    temp = np.round(np.random.uniform(0.0, 1.0, n), 2)
    db_sim = np.round(np.random.uniform(0.0, 1.0, n), 2)
    subj = np.round(np.random.uniform(0.0, 1.0, n), 2)
    overlap = np.round(np.random.uniform(0.0, 1.0, n), 2)
    contra = np.random.randint(0, 15, n)
    token = np.random.randint(20, 500, n)
    complexity = np.round(np.random.uniform(0.3, 1.0, n), 2)
    entity = np.round(np.random.uniform(0.2, 1.0, n), 2)
    relevance = np.round(np.random.uniform(0.2, 1.0, n), 2)
    faithful = np.round(np.random.uniform(0.2, 1.0, n), 2)
    hall = np.random.choice([0, 1], n, p=[0.55, 0.45])
    df = pd.DataFrame({
        "온도 설정값": temp,
        "데이터베이스 유사도 점수": db_sim,
        "주관성 점수": subj,
        "문맥-답변 중첩 비율": overlap,
        "모순 단어 수": contra,
        "입력 토큰 수": token,
        "문장 복잡도 지수": complexity,
        "개체 일치 점수": entity,
        "답변 관련성 점수": relevance,
        "문맥 충실도 점수": faithful,
        "환각 발생 여부": hall,
    })
    df.to_csv(CSV_PATH, index=False)

# 파생 열 3종
df["개체-문맥"] = (df["개체 일치 점수"] - df["문맥 충실도 점수"]).abs()
df["개체-답변관련성"] = (df["개체 일치 점수"] - df["답변 관련성 점수"]).abs()
df["답변관련성-문맥충실도"] = (df["답변 관련성 점수"] - df["문맥 충실도 점수"]).abs()

# ─────────────────────────────────────────────────────────────
# 전역 변수
# ─────────────────────────────────────────────────────────────
score_cols = ["개체 일치 점수", "답변 관련성 점수", "문맥 충실도 점수"]
score_labels = {
    "개체 일치 점수": "개체 일치 점수",
    "답변 관련성 점수": "답변 관련성 점수",
    "문맥 충실도 점수": "문맥 충실도 점수",
}

version = "버전 1.0"

# ─────────────────────────────────────────────────────────────
# 페이지 설정 + 사이드바
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="LLM 환각 분석", layout="wide")
st.sidebar.title("LLM 환각 분석")
st.sidebar.caption(version)

menu = st.sidebar.selectbox(
    "차례",
    ["서론", "히트맵", "막대그래프", "히스토그램1"],
    index=0,
    key="menu_selectbox",
)

# ============================================================
# 1. 서론  (상자그림 삭제, 데이터 설명, 분석대상표본수 추가)
# ============================================================
if menu == "서론":
    st.title("LLM의 환각 작용은 어떤 조건과 관련도가 높은가?")

    # ── 분석대상표본수 ────────────────────────────────────────
    total_n = len(df)
    hall_n = int((df["환각 발생 여부"] == 1).sum())
    non_hall_n = total_n - hall_n
    st.markdown(
        f"### 분석 대상 표본 수: **{total_n}행**  "
        f"(환각: {hall_n}행 · 비환각: {non_hall_n}행)"
    )

    st.markdown(
        """
        대규모 언어 모델(LLM)이 생성하는 응답에는 사실과 다른 내용, 무관한 내용,
        내부적으로 모순된 내용이 포함될 수 있습니다. 이를 **환각(hallucination)** 이라 하며,
        어떤 조건에서 환각이 더 자주 발생하는지 살펴보는 것이 이 분석의 목적입니다.
        """
    )

    # ── 데이터 설명 ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 데이터 설명")

    st.markdown(
        """
        - **데이터 구조**: RAG 환경 바탕 합성 데이터 30,000건
          (GPT-4o, Claude-3.5-Sonnet, Mistral-7B, Llama-3-70B 모델 포함)
        """
    )
    st.markdown("#### 데이터 속성")
    st.markdown(
        """
        | 속성 | 설명 |
        |------|------|
        | 입력 토큰 수 | 프롬프트에 입력된 토큰 수 |
        | 문장 복잡도 | 답변을 구성하는 구조, 언어적 복잡성 |
        | 데이터베이스 유사도 | 벡터 데이터베이스에서 검색된 자료와 검색 기준 사이의 유사도 |
        | 개체 일치도 | 답변의 개체가 제공된 문맥과 일치하는 정도 |
        | 답변 관련성 | 답변이 질문과 관련되고 의도에 부합하는 정도 |
        | 문맥 충실도 | 답변이 제공된 문맥에 근거하는 정도 |
        | 환각 발생 여부 | 없으면 0, 있으면 1로 표시 |
        """
    )

    st.warning(
        "실제 실험이 아닌 합성 데이터이고 4개의 모델에 한정되어 있으므로 "
        "분석 결과를 보편적인 LLM으로 해석하지 마십시오."
    )

    st.markdown(
        "[원본 데이터셋 (HuggingFace)](https://huggingface.co/datasets/vkshdev/rag-hallucination-benchmark?utm_source=chatgpt.com)"
    )

    st.markdown("---")
    st.markdown("### 작은 질문")

    st.markdown(
        """
        1. **입력 토큰 수, 문장 복잡도, 데이터베이스 유사도의 수치에 따라 환각률은
        어떻게 나타나는가?**
        → [막대그래프] 에서 구간별 환각 발생률을 확인할 수 있습니다.
        """
    )
    st.markdown(
        """
        2. **데이터베이스 유사도와 온도 설정값에 따라 환각률은 어떻게 변화하는가?**
        → [히트맵] 에서 두 변수의 조합별 환각률을 색깔로 비교할 수 있습니다.
        """
    )
    st.markdown(
        """
        3. **환각 발생 여부가 0, 1일 때 각각 개체 일치도, 답변 관련성,
        문맥 충실도 점수는 어떻게 분포하는가?**
        → [히스토그램1] 에서 그룹별 점수 분포를 비교할 수 있습니다.
        """
    )

    st.markdown("---")

# ============================================================
# 2. 히트맵
# ============================================================
elif menu == "히트맵":
    st.title("LLM의 환각작용")

    st.sidebar.markdown("#### 히트맵 설정")

    hm_n = st.sidebar.selectbox(
        "행렬수",
        options=list(range(5, 16)),
        index=4,
        key="hm_n_selectbox",
    )
    hm_n = int(hm_n)

    color_cycle = ["reds", "blues", "greens"]
    hm_color_idx = st.sidebar.selectbox(
        "색상",
        options=range(len(color_cycle)),
        index=0,
        format_func=lambda i: color_cycle[i],
        key="hm_color_selectbox",
    )
    hm_colorscale = color_cycle[hm_color_idx]

    x_col = "온도 설정값"
    y_col = "데이터베이스 유사도 점수"
    z_col = "환각 발생 여부"

    x_min, x_max = float(df[x_col].min()), float(df[x_col].max())
    y_min, y_max = float(df[y_col].min()), float(df[y_col].max())

    x_bins = np.linspace(x_min, x_max, hm_n + 1)
    y_bins = np.linspace(y_min, y_max, hm_n + 1)

    df["_z_bin"] = pd.cut(
        df[z_col],
        bins=hm_n,
        labels=[f"구간 {i+1}" for i in range(hm_n)],
        include_lowest=True,
    )

    pivot_z = df.pivot_table(
        index=pd.cut(df[y_col], bins=y_bins, include_lowest=True),
        columns=pd.cut(df[x_col], bins=x_bins, include_lowest=True),
        values="_z_bin",
        aggfunc=lambda s: (s == 1).sum() / s.count() * 100
        if s.count() > 0
        else np.nan,
        observed=True,
    )
    pivot_z = pivot_z.sort_index(ascending=False)

    pivot_counts = df.pivot_table(
        index=pd.cut(df[y_col], bins=y_bins, include_lowest=True),
        columns=pd.cut(df[x_col], bins=x_bins, include_lowest=True),
        values="_z_bin",
        aggfunc="count",
        observed=True,
    )
    pivot_counts = pivot_counts.sort_index(ascending=False)

    fig_hm = go.Figure()
    fig_hm.add_trace(
        go.Heatmap(
            z=pivot_z.values,
            x=[f"구간 {i+1}" for i in range(hm_n)],
            y=[f"구간 {i+1}" for i in range(hm_n)],
            colorscale=hm_colorscale,
            showscale=True,
            zmin=0,
            zmax=100,
            hoverongaps=False,
            colorbar=dict(title="환각률(%)"),
        )
    )
    fig_hm.update_yaxes(autorange="reversed")
    fig_hm.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        template="plotly_white",
        height=520,
        margin=dict(l=60, r=30, t=30, b=50),
    )

    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown(
        "**데이터베이스와 유사할 수록 환각률이 낮고, 답변의 무작위성을 높게 설정할 수록 "
        "환각률이 높게 나타난다.**"
    )

    st.markdown("### 구간별 표본수")
    st.dataframe(
        pivot_counts.rename_axis(None, axis=0).rename_axis(None, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 축 양끝값")
    c1, c2, c3 = st.columns(3)
    c1.caption("X축 왼쪽 끝: 0")
    c2.caption(f"X축 오른쪽 끝: {x_max}")
    c3.caption(f"Y축 아래쪽 끝: {y_max}")

# ============================================================
# 3. 막대그래프  (JSON 직렬화 오류 수정)
# ============================================================
elif menu == "막대그래프":
    st.title("LLM의 환각작용")

    st.sidebar.markdown("#### 막대그래프 설정")

    bar_x_options = [
        ("입력 토큰 수", "입력 토큰 수"),
        ("문장 복잡도 지수", "문장 복잡도 지수"),
        ("데이터베이스 유사도 점수", "데이터베이스 유사도 점수"),
    ]
    bar_x_idx = st.sidebar.selectbox(
        "X축 속성",
        options=range(len(bar_x_options)),
        index=0,
        format_func=lambda i: bar_x_options[i][0],
        key="bar_x_selectbox",
    )
    bar_x_col = bar_x_options[bar_x_idx][1]

    bar_n = st.sidebar.selectbox(
        "구간 개수",
        options=list(range(1, 11)),
        index=7,
        key="bar_n_selectbox",
    )
    bar_n = int(bar_n)

    try:
        bins = pd.qcut(df[bar_x_col], q=bar_n, duplicates="drop")
        bins = bins.cat.remove_unused_categories()
        n_actual = bins.nunique()
        labels = [f"구간 {i+1}" for i in range(n_actual)]
        df["_bar_bin"] = pd.cut(
            df[bar_x_col],
            bins=bins.cat.categories.right,
            include_lowest=True,
            labels=labels[:n_actual],
        )
    except ValueError:
        n_actual = min(bar_n, len(df) // 5)
        if n_actual < 1:
            n_actual = 1
        bins = np.linspace(df[bar_x_col].min(), df[bar_x_col].max(), n_actual + 1)
        df["_bar_bin"] = pd.cut(df[bar_x_col], bins=bins, include_lowest=True)

    agg = (
        df.groupby("_bar_bin", observed=True)
        .agg(
            전체_행수=("환각 발생 여부", "count"),
            환각_행수=("환각 발생 여부", lambda x: (x == 1).sum()),
        )
        .reset_index()
    )
    agg["환각률(%)"] = (agg["환각_행수"] / agg["전체_행수"] * 100).round(2)

    boundaries = sorted(set(df["_bar_bin"].dropna().cat.categories))

    # ── JSON 직렬화 호환: 범주형 → 문자열, 실수 → float ──────
    x_labels = [str(x) for x in agg["_bar_bin"]]
    y_vals_bar = [float(y) for y in agg["환각률(%)"]]
    text_vals = [f"{y:.2f}%" for y in agg["환각률(%)"]]

    fig_bar = go.Figure()
    fig_bar.add_trace(
        go.Bar(
            x=x_labels,
            y=y_vals_bar,
            text=text_vals,
            textposition="outside",
            marker_color="steelblue",
            name=f"{bar_x_col} 구간별 환각률(%)",
        )
    )
    fig_bar.update_layout(
        xaxis_title=bar_x_col,
        yaxis_title="환각 발생률 (%)",
        yaxis_range=[0, max(max(y_vals_bar) * 1.2, 10)],
        template="plotly_white",
        height=500,
        bargap=0.25,
        margin=dict(l=50, r=30, t=30, b=55),
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown(
        "**입력 토큰 수에 따라 환각률은 고르고, 문장 복잡도에 따라서는 뚜렷한 결과가 보이지 않으며, "
        "데이터베이스 유사도에 따라선 x축 값이 증가함에 따라 환각률이 낮아지는 경향이 두드러진다.**"
    )

    st.markdown("### 구간 경계값")
    st.caption(" | ".join([str(b) for b in boundaries]))

    st.markdown("### 전체 표본수")
    st.caption(f"전체: {total_n}행 | 환각: {hall_n}행 | 비환각: {non_hall_n}행")

# ============================================================
# 4. 히스토그램1
# ============================================================
elif menu == "히스토그램1":
    st.title("LLM의 환각작용")

    st.sidebar.markdown("#### 히스토그램1 설정")

    hist_score_idx = st.sidebar.selectbox(
        "관찰 속성",
        options=range(len(score_cols)),
        index=0,
        format_func=lambda i: score_cols[i],
        key="hist_score_selectbox",
    )
    sel_score = score_cols[hist_score_idx]

    group_options = [("0 (비환각)", 0), ("1 (환각)", 1)]
    group_idx = st.sidebar.selectbox(
        "그룹",
        options=range(len(group_options)),
        index=0,
        format_func=lambda i: group_options[i][0],
        key="hist_group_selectbox",
    )
    sel_group = group_options[group_idx][1]

    score_kr = score_labels.get(sel_score, sel_score)
    group_name = f"{sel_group} ( {'비환각' if sel_group == 0 else '환각'} )"
    subset = df[df["환각 발생 여부"] == sel_group][sel_score].dropna()
    group_n = len(df[df["환각 발생 여부"] == sel_group])

    if len(subset) == 0:
        st.warning(f"선택된 그룹 '{group_name}'에 점수 '{score_kr}' 데이터가 없습니다.")
        st.stop()

    counts, bin_edges = np.histogram(subset, bins=10)
    pct = counts / len(subset) * 100
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Bar(
            x=[float(c) for c in bin_centers],
            y=[float(p) for p in pct],
            name=f"{score_kr} ({group_name})",
            marker_color="#4C78A8",
            opacity=0.85,
            hovertemplate=(
                "점수 구간 중심: %{x:.3f}<br>"
                "비율: %{y:.2f}%<extra></extra>"
            ),
        )
    )

    x_min = float(subset.min())
    x_max = float(subset.max())
    fig_hist.update_xaxes(range=[x_min, x_max], title_text=score_kr, tickformat=".3f")
    fig_hist.update_yaxes(title_text="비율(%)", range=[0, max(100, float(pct.max()) * 1.15)])

    fig_hist.update_layout(
        title=f"히스토그램 — {score_kr} · {group_name} (n={group_n})<br>"
        f"10개 빈 · y축: 해당 그룹 내 백분율(%)",
        height=420,
        margin=dict(l=50, r=30, t=80, b=55),
        bargap=0.12,
    )

    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown(
        f"**환각 여부가 {sel_group}인 그룹에서 {score_kr} 점수가 "
        f"전체적으로 {'높은' if sel_group == 0 else '낮은'} 값에 분포하고 있다.**"
    )

    st.markdown("### 분포 요약")
    st.caption(
        f"그룹 {group_name} 전체 표본수: {group_n}행 "
        f"(점수 결측 제외: {len(subset)}행)"
    )
    st.caption(
        f"평균 {subset.mean():.3f} · 중앙값 {subset.median():.3f} · "
        f"표준편차 {subset.std():.3f} · 최소 {subset.min():.3f} · 최대 {subset.max():.3f}"
    )

    st.markdown("### 전체 표본수")
    st.caption(
        f"전체: {total_n}행 | 환각(1): {hall_n}행 | 비환각(0): {non_hall_n}행"
    )

else:
    st.stop()
