import pandas as pd
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go
import streamlit as st


def public_trust_text():
    st.markdown("""


    ### Purpose 
    Analyzing confidence levels in public institutions throughout the war period.

    ### How To Use
    - **Main Plot**: View the average trust levels during the war for each institution or public figure.
    - **Drill Down**: Click on a specific bar to see the trust levels throughout the war.
    - **Demographic Breakdown**: Choose a demographic dimension to see changes in trust over time within each subgroup.
    - **Scoring Range**: Trust scores range from 1 (lowest) to 4 (highest).
    - **Over Time view**: Click on a line in the plot or a category in the legend to hide it. Click again on the legend to bring it back. 
    """)


def create_trust_dashboard(bibi_data_path, tzal_data_path, mishtara_data_path, memshala_data_path, selected_demo=None):

    def prepare_monthly_data_cases(data, keyword, columns):
        filtered_data = data[data['q_full'].str.contains(keyword, case=False, na=False)].copy()
        filtered_data['date'] = pd.to_datetime(filtered_data['date'], errors='coerce')
        filtered_data.dropna(subset=['date'], inplace=True)
        filtered_data['month_year'] = filtered_data['date'].dt.to_period('M')

        available_cols = [c for c in columns if c in filtered_data.columns]
        if not available_cols:
            return pd.DataFrame(columns=['month_year', 'trust_score'])

        filtered_data[available_cols] = filtered_data[available_cols].apply(pd.to_numeric, errors='coerce')
        weights = list(range(len(available_cols), 0, -1))

        filtered_data['trust_score'] = (
            sum(filtered_data[c] * w for c, w in zip(available_cols, weights))
            / filtered_data[available_cols].sum(axis=1)
        )
        return filtered_data

    def aggregate_monthly(df):
        if df.empty:
            return pd.DataFrame(columns=['month_year', 'trust_score'])
        return df.groupby('month_year', as_index=False)['trust_score'].mean().sort_values('month_year')

    @st.cache_data
    def load_data():
        try:
            bibi_data = pd.read_excel(bibi_data_path)
            tzal_data = pd.read_excel(tzal_data_path)
            mishtara_data = pd.read_excel(mishtara_data_path)
            memshala_data = pd.read_excel(memshala_data_path)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

        bibi_rows = prepare_monthly_data_cases(bibi_data, "ראש ממשלה", ["a1","a2","a3","a4"])
        tzal_rows = prepare_monthly_data_cases(tzal_data, "צהל", ["a1","a2","a3","a4"])
        mish_rows = prepare_monthly_data_cases(mishtara_data, "משטרה", ["a1","a2","a3","a4"])
        mems_rows = prepare_monthly_data_cases(memshala_data, "ממשלה", ["a1","a2","a3","a4"])

        bibi_monthly = aggregate_monthly(bibi_rows)
        tzal_monthly = aggregate_monthly(tzal_rows)
        mish_monthly = aggregate_monthly(mish_rows)
        memshala_monthly = aggregate_monthly(mems_rows)

        for df in [bibi_monthly, tzal_monthly, mish_monthly, memshala_monthly]:
            df['month_year_str'] = df['month_year'].astype(str)

        all_months = sorted(
            set(bibi_monthly['month_year_str'])
            .union(tzal_monthly['month_year_str'])
            .union(mish_monthly['month_year_str'])
            .union(memshala_monthly['month_year_str'])
        )

        def df_to_dict(df):
            return dict(zip(df['month_year_str'], df['trust_score']))

        return {
            "bibi_scores": df_to_dict(bibi_monthly),
            "tzal_scores": df_to_dict(tzal_monthly),
            "mishtara_scores": df_to_dict(mish_monthly) if not mish_monthly.empty else {},
            "memshala_scores": df_to_dict(memshala_monthly),
            "months": all_months,
            "bibi_rows": bibi_rows,
            "tzal_rows": tzal_rows,
            "mishtara_rows": mish_rows,
            "memshala_rows": mems_rows
        }

    data = load_data()

    institutions = {
        "bibi": "Prime Minister",
        "tzal": "IDF",
        "mishtara": "Police",
        "memshala": "Government"
    }

    def plot_scatter_chart():
        scatter_data = []
        for inst, name in institutions.items():
            key = f"{inst}_scores"
            if key not in data:
                continue

            scores_dict = data[key]
            if scores_dict:
                avg_trust = sum(scores_dict.values()) / len(scores_dict)
            else:
                avg_trust = 0

            scatter_data.append({
                "Institution": name,
                "Trust Score": avg_trust,
                "Key": inst
            })

        scatter_df = pd.DataFrame(scatter_data).sort_values("Trust Score")
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=scatter_df["Institution"],
            y=scatter_df["Trust Score"],
            marker=dict(color="gray", line=dict(width=2, color="white")),
            text=[f"{val:.2f}" for val in scatter_df["Trust Score"]],
            textposition="outside",
            hovertemplate="<b>Institution</b>: %{x}<br><b>Trust Score</b>: %{y:.2f}<extra></extra>"
        ))

        fig.update_layout(
            title="Trust Scores by Institution (overall average : 2.36)",
            xaxis=dict(title="Institution"),
            yaxis=dict(title="Trust Score", range=[0,4]),
            showlegend=False,
            bargap=0.5
        )
        return fig

    bar_fig = plot_scatter_chart()
    selected_point = plotly_events(bar_fig, click_event=True, key="bar_chart")
    return selected_point, data, institutions

def create_demographic_time_series(selected_inst_key, selected_demo, data, institutions):
    trust_scores = data.get(f"{selected_inst_key}_rows", pd.DataFrame())
    if trust_scores.empty:
        st.warning(f"No data available for {selected_inst_key}")
        return
    institution_name = institutions.get(selected_inst_key, "Unknown Institution")
    demo_mapping = {
        "District": {
            "North": {"hebrew": "צפון", "color": "rgb(1, 56, 255)"},
            "Haifa": {"hebrew": "חיפה", "color": "rgb(255, 165, 0)"},
            "Center": {"hebrew": "מרכז", "color": "rgb(1, 255, 244)"},
            "Tel Aviv": {"hebrew": "תל אביב", "color": "rgb(255, 0, 246)"},
            "Jerusalem": {"hebrew": "ירושלים", "color": "rgb(237, 255, 0)"},
            "Judea & Samaria": {"hebrew": "יהודה ושומרון", "color": "rgb(128, 0, 128)"},
            "South": {"hebrew": "דרום", "color": "rgb(220, 20, 60)"}
        },
        "Religiousness": {
            "Ultra-Orthodox": {"hebrew": "חרדי", "color": "rgb(0, 0, 139)"},
            "Religious": {"hebrew": "דתי", "color": "rgb(0, 0, 205)"},
            "Traditional": {"hebrew": "מסורתי", "color": "rgb(30, 144, 255)"},
            "Secular": {"hebrew": "חילוני", "color": "rgb(135, 206, 250)"}
        },
        "Political stance": {
            "Right": {"hebrew": "ימין", "color": "rgb(255, 0, 0)"},
            "Center": {"hebrew": "מרכז", "color": "rgb(0, 255, 0)"},
            "Left": {"hebrew": "שמאל", "color": "rgb(0, 0, 255)"},
            "Refuses to Answer": {"hebrew": "מסרב", "color": "rgb(128, 128, 128)"}
        },
        "Age": {
            "75+": {"hebrew": "75+", "color": "rgb(70, 0, 0)"},
            "65-74": {"hebrew": "65-74", "color": "rgb(100, 34, 34)"},
            "55-64": {"hebrew": "55-64", "color": "rgb(160, 80, 80)"},
            "45-54": {"hebrew": "45-54", "color": "rgb(213, 109, 109)"},
            "35-44": {"hebrew": "35-44", "color": "rgb(230, 160, 143)"},
            "25-34": {"hebrew": "25-34", "color": "rgb(240, 200, 160)"},
            "18-24": {"hebrew": "18-24", "color": "rgb(250, 229, 190)"}
        }
    }

    fig = go.Figure()

    if selected_demo == "All" or not selected_demo:
        # overall average
        avg_trust = trust_scores.groupby("month_year")["trust_score"].mean().reset_index()
        avg_trust["month_year_str"] = avg_trust["month_year"].astype(str)

        fig.add_trace(go.Scatter(
            x=avg_trust["month_year_str"],
            y=avg_trust["trust_score"],
            name="Overall Average Trust",
            mode="lines+markers",
            line=dict(width=2, color="black"),
            marker=dict(size=8, color="black")
        ))
    else:
        selected_map = demo_mapping.get(selected_demo, {})
        for eng_label, value in selected_map.items():
            sub_data = trust_scores[trust_scores["sub_subject"] == value["hebrew"]]
            if not sub_data.empty:
                monthly_avg = sub_data.groupby("month_year")["trust_score"].mean().reset_index()
                monthly_avg["month_year_str"] = monthly_avg["month_year"].astype(str)
                fig.add_trace(go.Scatter(
                    x=monthly_avg["month_year_str"],
                    y=monthly_avg["trust_score"],
                    name=eng_label,
                    mode="lines+markers",
                    line=dict(width=2, color=value["color"]),
                    marker=dict(size=8, color=value["color"])
                ))

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Trust Score",
        yaxis_range=[1, 4],
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        margin=dict(b=100),
        annotations=[
            dict(
                xref="paper", yref="paper",
                x=0.5, y=1.15,
                text=f"Trust Scores for {institution_name} by {selected_demo or 'All'} Over Time",
                showarrow=False,
                font=dict(size=16, family="Arial", color="black"),
                xanchor="center"
            )
        ]
    )

    st.plotly_chart(fig, use_container_width=True)
