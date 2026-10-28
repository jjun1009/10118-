elif menu == "히스토그램":
    st.markdown("### 문장복잡도 지수 구간별 히스토그램 (환각 발생 행만)")

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
            help="히스토그램으로 표시할 점수를 선택합니다. 환각 발생 행(Is_Hallucination=1)만 대상으로 합니다.",
            key="hist_y_selectbox",
        )

    with col_right:
        complexity_bins_cut = pd.cut(
            df["복잡도 지수"],
            bins=complexity_bins,
            labels=[f"복잡도 구간 {i+1}" for i in range(complexity_bins)],
            include_lowest=True,
        )
        df["복잡도 구간"] = complexity_bins_cut

        complexity_labels = complexity_bins_cut.cat.categories.tolist()

        selected_complexity = st.selectbox(
            "볼 복잡도 구간",
            options=complexity_labels,
            index=0,
            help="히스토그램을 표시할 문장복잡도 구간을 선택합니다.",
            key="complexity_interval_selectbox",
        )

    # 해당 구간 전체에서 환각 발생 여부가 1인 행만 필터
    interval_all = df[df["복잡도 구간"] == selected_complexity]
    interval_hallucinated = interval_all[interval_all["환각 발생 여부"] == 1].dropna(
        subset=[hist_y_col]
    )

    total_in_interval = len(interval_all)
    hallucinated_count = len(interval_hallucinated)

    st.markdown(f"### {selected_complexity} — {hist_y_col} 분포 (환각 발생 행만, %)")

    st.caption(
        f"해당 구간 전체 행수: {total_in_interval}행 · "
        f"환각 발생 행수: {hallucinated_count}행"
    )

    if hallucinated_count == 0:
        st.warning(
            f"선택한 구간({selected_complexity})에 환각 발생 행(Is_Hallucination=1)이 없습니다."
        )
    else:
        import numpy as np

        x_vals = interval_hallucinated[hist_y_col].values
        total = total_in_interval

        counts, bin_edges = np.histogram(x_vals, bins=30)
        percentages = counts / total * 100

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Bar(
                x=bin_centers,
                y=percentages,
                name=f"{hist_y_col} 분포",
                marker_color="crimson",
                opacity=0.75,
                hovertemplate=(
                    f"{hist_y_col}: %{{x:.3f}}<br>"
                    "비율: %{{y:.2f}}%<br>"
                    "도수: %{customdata}<extra></extra>"
                ),
                customdata=counts,
                width=(bin_edges[1] - bin_edges[0]) * 0.9,
            )
        )

        fig_hist.update_layout(
            xaxis_title=hist_y_col,
            yaxis_title=f"비율 (% of 전체 {total_in_interval}행)",
            template="plotly_white",
            height=450,
            bargap=0.05,
            yaxis_range=[0, max(percentages.max() * 1.2, 1)],
        )

        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("📊 환각 발생 행 기본 통계")
        stats_data = {
            "통계량": ["환각 행수", "평균", "표준편차", "최소값", "25%", "중앙값", "75%", "최대값"],
            "값": [
                hallucinated_count,
                round(interval_hallucinated[hist_y_col].mean(), 4),
                round(interval_hallucinated[hist_y_col].std(), 4),
                round(interval_hallucinated[hist_y_col].min(), 4),
                round(interval_hallucinated[hist_y_col].quantile(0.25), 4),
                round(interval_hallucinated[hist_y_col].median(), 4),
                round(interval_hallucinated[hist_y_col].quantile(0.75), 4),
                round(interval_hallucinated[hist_y_col].max(), 4),
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
