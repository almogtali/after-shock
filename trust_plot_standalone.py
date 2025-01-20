import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

# ---- FILE PATHS ----
bibi_data_path = "data_storage/bibi.xlsx"
tzal_data_path = "data_storage/tzal.xlsx"
mishtara_data_path = "data_storage/mishtra.xlsx"
memshala_data_path = "data_storage/memshla.xlsx"


# ---- DATA PROCESSING FUNCTIONS ----
def prepare_monthly_data_cases(data, keyword, columns):
    """Filter rows containing keyword and compute weighted trust score."""
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
    """Group by month and return average trust score per month."""
    if df.empty:
        return pd.DataFrame(columns=['month_year', 'trust_score'])
    return df.groupby('month_year', as_index=False)['trust_score'].mean().sort_values('month_year')


@st.cache_data
def load_data():
    """Load and process data from Excel files."""
    try:
        bibi_data = pd.read_excel(bibi_data_path)
        tzal_data = pd.read_excel(tzal_data_path)
        mishtara_data = pd.read_excel(mishtara_data_path)
        memshala_data = pd.read_excel(memshala_data_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Process row-level data
    bibi_rows = prepare_monthly_data_cases(bibi_data, keyword="ראש ממשלה", columns=["a1", "a2", "a3", "a4"])
    tzal_rows = prepare_monthly_data_cases(tzal_data, keyword="צהל", columns=["a1", "a2", "a3", "a4"])
    mish_rows = prepare_monthly_data_cases(mishtara_data, keyword="משטרה", columns=["a1", "a2", "a3", "a4"])
    mems_rows = prepare_monthly_data_cases(memshala_data, keyword="ממשלה", columns=["a1", "a2", "a3", "a4"])

    # Aggregate Monthly Data
    bibi_monthly = aggregate_monthly(bibi_rows)
    tzal_monthly = aggregate_monthly(tzal_rows)
    mish_monthly = aggregate_monthly(mish_rows)
    memshala_monthly = aggregate_monthly(mems_rows)

    for df in [bibi_monthly, tzal_monthly, mish_monthly, memshala_monthly]:
        df['month_year_str'] = df['month_year'].astype(str)

    all_months = sorted(set(bibi_monthly['month_year_str'])
                        .union(tzal_monthly['month_year_str'])
                        .union(mish_monthly['month_year_str'])
                        .union(memshala_monthly['month_year_str']))

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


# ---- STREAMLIT DASHBOARD ----
st.title("Israeli Sentiments Dashboard")
st.subheader("Trust in Institutions Over Time")

# Load Data
data = load_data()
months = data["months"]

# Institution Mapping
institutions = {
    "bibi": "Prime Minister",
    "tzal": "IDF",
    "mishtara": "Police",
    "memshala": "Government"
}

# Color Mapping
color_map = {
    "bibi": "#FF5733",
    "tzal": "#2ECC71",
    "mishtara": "#3498DB",
    "memshala": "#F39C12"
}



# ---- Updated Scatter Plot: Trust vs. Institution ----
def plot_scatter_chart():
    scatter_data = []
    min_size, max_size = 20, 100  # Define limits for bubble size

    for inst, name in institutions.items():
        if f"{inst}_scores" not in data:
            continue

        avg_trust = sum(data[f"{inst}_scores"].values()) / len(data[f"{inst}_scores"]) if data[f"{inst}_scores"] else 0
        size = min_size + (avg_trust * 10)  # Scale size slightly, preventing overlap
        size = max(min_size, min(size, max_size))  # Ensure size stays within range

        scatter_data.append({
            "Institution": name,
            "Trust Score": avg_trust,
            "Key": inst,
            "Size": size
        })

    scatter_df = pd.DataFrame(scatter_data)

    fig = go.Figure()
    for _, row in scatter_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["Institution"]],
            y=[row["Trust Score"]],
            mode="markers+text",
            marker=dict(
                size=row["Size"],
                color=color_map[row["Key"]],
                line=dict(width=2, color="white")
            ),
            text=row["Institution"],
            textposition="top center",
            hoverinfo="text+name"
        ))

    fig.update_layout(
        title="Trust Scores by Institution",
        yaxis=dict(title="Trust Score (1-5)", range=[1, 5]),
        xaxis=dict(title="Institution"),
        showlegend=False
    )
    return fig


# Interactive event capture
selected_point = plotly_events(plot_scatter_chart(), click_event=True)



def create_demographic_time_series(selected_inst_key):
    trust_scores = data.get(f"{selected_inst_key}_rows", pd.DataFrame())

    if trust_scores.empty:
        st.warning(f"No data available for {selected_inst_key}")
        return None

    # Get the full institution name for the title
    institution_name = institutions.get(selected_inst_key, "Unknown Institution")

    demo_mapping = {
        "District": {
            "North": "צפון",
            "Haifa": "חיפה",
            "Center": "מרכז",
            "Tel Aviv": "תל אביב",
            "Jerusalem": "ירושלים",
            "Judea & Samaria": "יהודה ושומרון",
            "South": "דרום"
        },
        "Religiousness": {
            "Secular": "חילוני",
            "Traditional": "מסורתי",
            "Religious": "דתי",
            "Ultra-Orthodox": "חרדי"
        },
        "Political stance": {
            "Right": "ימין",
            "Center": "מרכז",
            "Left": "שמאל",
            "Refuses to Answer": "מסרב"
        },
        "Age": {
            "18-24": "18-24",
            "25-34": "25-34",
            "35-44": "35-44",
            "45-54": "45-54",
            "55-64": "55-64",
            "65-74": "65-74",
            "75+": "75+"
        }
    }

    demo_choice = st.selectbox("Choose a demographic dimension:", list(demo_mapping.keys()))
    selected_map = demo_mapping[demo_choice]

    fig = go.Figure()

    for eng_label, hebrew_label in selected_map.items():
        # Filter data for the current demographic
        sub_data = trust_scores[trust_scores["sub_subject"] == hebrew_label].copy()

        if not sub_data.empty:
            # Calculate monthly averages
            monthly_avg = sub_data.groupby("month_year")["trust_score"].mean().reset_index()
            monthly_avg["month_year_str"] = monthly_avg["month_year"].astype(str)

            # Add trace for this demographic
            fig.add_trace(go.Scatter(
                x=monthly_avg["month_year_str"],
                y=monthly_avg["trust_score"],
                name=eng_label,
                mode="lines+markers",
                connectgaps=True,
                line=dict(width=2),
                marker=dict(size=8)
            ))

    # Update layout with institution name in title
    fig.update_layout(
        title=f"Trust Scores for {institution_name} Over Time by {demo_choice}",
        xaxis_title="Month",
        yaxis_title="Trust Score",
        yaxis_range=[1, 5],
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    st.plotly_chart(fig, use_container_width=True)




if selected_point:
    selected_inst = selected_point[0]["x"]
    selected_inst_key = next((key for key, val in institutions.items() if val == selected_inst), None)
    create_demographic_time_series(selected_inst_key)