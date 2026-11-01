# 히스토그램1 겹침 보기 (오류 수정 부분)
else:
    sel_group = grp_options[st.session_state["hist1_grp_idx"]]
    grp       = 0 if "0" in sel_group else 1
    grp_label = "그룹 0 (비환각)" if grp == 0 else "그룹 1 (환각)"
    n_total   = n_total_0 if grp == 0 else n_total_1

    sub = df[df["환각 발생 여부"] == grp].copy()

    if len(sub) == 0:
        st.warning(f"선택한 그룹 '{grp_label}'의 데이터가 없습니다.")
    else:
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

        COLORS = {"개체 일치 점수":"#4C78A8","답변 관련성 점수":"#E45759","문맥 충실도 점수":"#52B95E"}
        MARKERS = {"개체 일치 점수":"circle","답변 관련성 점수":"square","문맥 충실도 점수":"diamond"}

        fig = go.Figure()
        any_trace = False
        for col in score_cols:
            x_vals = sub[norm_cols[col]].dropna().astype(float)
            if len(x_vals) == 0:
                continue
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
            any_trace = True

        if not any_trace:
            st.warning("정규화된 데이터가 없습니다. 다른 그룹을 선택하세요.")
        else:
            fig.update_layout(
                title=(f"{grp_label}  ·  세 속성 정규화 겹침  ·  n = {n_total}행  ·  (각 속성별 [min, max] → [0, 1])"),
                xaxis_title="정규화 점수 (0 ~ 1)",
                yaxis_title="비율 (%)",
                bargap=0.05,
                barmode="overlay",
                margin=dict(l=50, r=20, t=60, b=50),
                height=400,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            )
            fig.update_xaxes(range=[0, 1], tickangle=0)
            fig.update_yaxes(ticksuffix="%")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 결론")
            st.markdown("> **세 속성을 각각 최솟값 0, 최댓값 1로 정규화하여 동일 축에 겹쳐 본 결과, 환각 그룹(1)과 비환각 그룹(0) 간에 세 속성의 분포 경향이 서로 다른 위치에서 집중되는 양상을 확인할 수 있다.**")
            st.caption(f"■ 선택된 그룹: {grp_label}  |  총 표본수: {n_total}행  |  표시: **세 속성 정규화 겹침** (각 속성별 [min, max] → [0, 1])")
