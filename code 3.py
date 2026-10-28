# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# ──────────────────────────────────────────────
# 1. CSV가 없으면 예시 데이터 생성
# ──────────────────────────────────────────────
CSV_PATH = "rag_hallucination.csv"

if not os.path.exists(CSV_PATH):
    import numpy as np
    np.random.seed(42)
    n = 500
    df_raw = pd.DataFrame({
        "index": range(n),
        "Interaction_ID": [f"INT-{i:04d}" for i in range(n)],
        "User_Prompt": [f"질문 {i}" for i in range(n)],
        "Provided_Context": [f"문맥 {i}" for i in range(n)],
        "LLM_Generated_Response": [f"응답 {i}" for i in range(n)],
        "LLM_Model_Name": np.random.choice(["gpt-4", "claude-3", "gemini-pro", "llama-3"], n),
        "Temperature_Setting": np.round(np.random.uniform(0, 2, n), 2),
        "Vector_DB_Similarity_Score": np.round(np.random.uniform(0, 1, n), 4),
        "Prompt_Tokens": np.random.randint(50, 500, n),
        "Response_Tokens": np.random.randint(20, 300, n),
        "Context_Response_Overlap_Ratio": np.round(np.random.uniform(0, 1, n), 4),
        "Entity_Match_Score": np.round(np.random.uniform(0, 1, n), 4),
        "Sentence_Complexity_Index": np.round(np.random.uniform(0, 1, n), 4),
        "Subjectivity_Score": np.round(np.random.uniform(0, 1, n), 4),
        "Contradiction_Words_Count": np.random.randint(0, 10, n),
        "Answer_Relevance_Score": np.round(np.random.uniform(0, 1, n), 4),
        "Context_Faithfulness_Score": np.round(np.random.uniform(0, 1, n), 4),
        "Hallucination_Type": np.random.choice(["사실 왜곡", "문맥 이탈", "근거 부족", "없음"], n),
        "Is_Hallucination": np.random.choice([0, 1], n, p=[0.6, 0.4]),
    })
    df_raw.to_csv(CSV_PATH, index=False)

# ──────────────────────────────────────────────
# 2. CSV 로드 → 삭제·개명 처리
# ──────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

DROP_COLS = [
    "index", "Interaction_ID", "User_Prompt", "Provided_Context",
    "LLM_Generated_Response", "Subjectivity_Score",
    "Context_Response_Overlap_Ratio", "Contradiction_Words_Count",
]
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

RENAME_MAP = {
    "LLM_Model_Name":       "모델 이름",
    "Temperature_Setting":  "온도 설정값",
    "Vector_DB_Similarity_Score": "데이터베이스 유사도 점수",
    "Prompt_Tokens":        "입력 토큰 수",
    "Response_Tokens":      "답변 토큰 수",
    "Entity_Match_Score":   "개체 일치 점수",
    "Sentence_Complexity_Index": "복잡도 지수",
    "Answer_Relevance_Score":    "답변 관련성 점수",
    "Context_Faithfulness_Score":"문맥 충실도 점수",
    "Hallucination_Type":   "환각 유형",
    "Is_Hallucination":     "환각 발생 여부",
}
df = df.rename(columns=RENAME_MAP)

# ──────────────────────────────────────────────
# 3. 속성 목록 (히스토그램 대상 3개)
# ──────────────────────────────────────────────
ATTR_OPTIONS = ["개체 일치 점수", "답변 관련성 점수", "문맥 충실도 점수"]

# ──────────────────────────────────────────────
# 4. 사이드바 메뉴
# ──────────────────────────────────────────────
st.sidebar.title("메뉴")
page = st.sidebar.radio("이동", ["서론", "히스토그램"])

# ──────────────────────────────────────────────
# 5. 공통 제목
# ──────────────────────────────────────────────
st.title("LLM의 환각작용")

# ──────────────────────────────────────────────
# 6. 서론 페이지
# ──────────────────────────────────────────────
if page == "서론":
    st.write("서론 내용은 아직 준비 중입니다.")

# ──────────────────────────────────────────────
# 7. 히스토그램 페이지
# ──────────────────────────────────────────────
if page == "히스토그램":
    col_sel, _ = st.columns([1, 5])
    with col_sel:
        group_choice = st.radio(
            "그룹 선택",
            ["0 (비환각)", "1 (환각)", "전체 비교 (6개 모두)"],
            horizontal=True,
        )
        attr_choice = st.radio(
            "표시 속성",
            ATTR_OPTIONS,
            horizontal=True,
        )

    if group_choice == "전체 비교 (6개 모두)":
        groups_to_show = [0, 1]
        show_all = True
    else:
        grp = 0 if "0" in group_choice else 1
        groups_to_show = [grp]
        show_all = False

    # 각 그룹 표본 수 미리 계산
    n_total_0 = int((df["환각 발생 여부"] == 0).sum())
    n_total_1 = int((df["환각 발생 여부"] == 1).sum())

    # 히스토그램 생성 함수
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

    if show_all:
        # 2행 × 3열 그리드 (그룹 0: 상단, 그룹 1: 하단)
        placeholders = []
        for r in range(2):
            row_placeholders = []
            for c in range(3):
                row_placeholders.append(st.empty())
            placeholders.append(row_placeholders)

        for ci, attr in enumerate(ATTR_OPTIONS):
            g0 = df[df["환각 발생 여부"] == 0]
            fig0 = make_hist(g0, attr, "그룹 0 (비환각)", n_total_0)
            placeholders[0][ci].write(fig0)

            g1 = df[df["환각 발생 여부"] == 1]
            fig1 = make_hist(g1, attr, "그룹 1 (환각)", n_total_1)
            placeholders[1][ci].write(fig1)
    else:
        # 선택한 그룹·속성 1개만 표시
        grp_label = "그룹 0 (비환각)" if groups_to_show[0] == 0 else "그룹 1 (환각)"
        n_total = n_total_0 if groups_to_show[0] == 0 else n_total_1
        sub = df[df["환각 발생 여부"] == groups_to_show[0]]
        fig = make_hist(sub, attr_choice, grp_label, n_total)
        st.write(fig)
