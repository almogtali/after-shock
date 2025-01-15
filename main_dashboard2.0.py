import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import plotly.subplots as sp
import plotly.express as px
import geopandas as gpd
from datetime import date
from pyngrok import ngrok
import subprocess
import json

############################
# --- FILE PATHS (EXAMPLE) -
############################
PM_DATA_PATH = "data_storage/bibi.xlsx"
ARMY_DATA_PATH = "data_storage/tzal.xlsx"
POLICE_DATA_PATH = "data_storage/mishtra.xlsx"
GOVT_DATA_PATH = "data_storage/memshla.xlsx"


def keep_only_political_consolidated(df):
    """
    1) Keep only rows where subject == "Political Consolidated"
    2) Convert certain sub_subject values (e.g., "Moderate Right" -> "Right", etc.)
    3) Keep only sub_subject in {"Right", "Center", "Left", "Refused"}
    """
    if 'subject' not in df.columns or 'sub_subject' not in df.columns:
        return df

    # 1. Keep only subject == "Political Consolidated"
    df = df[df['subject'] == "Political Consolidated"].copy()

    # 2. Unify sub_subject values
    replacements = {
        "Moderate Right": "Right",
        "Moderate Left": "Left",
    }
    df['sub_subject'] = df['sub_subject'].replace(replacements)

    # 3. Keep only the final four categories
    valid_stances = {"Right", "Center", "Left", "Refused"}
    df = df[df['sub_subject'].isin(valid_stances)]

    return df


def prepare_monthly_data_cases(data, keyword, columns):
    """Prepare monthly trust data for institutions."""
    # Implementation remains the same, just updated variable names
    filtered_data = data[data['q_full'].str.contains(keyword, case=False, na=False)].copy()

    filtered_data['date'] = pd.to_datetime(filtered_data['date'], errors='coerce')
    filtered_data.dropna(subset=['date'], inplace=True)
    filtered_data['month_year'] = filtered_data['date'].dt.to_period('M')

    available_cols = [c for c in columns if c in filtered_data.columns]
    if not available_cols:
        return pd.DataFrame(columns=['month_year', 'trust_score'])

    for col in available_cols:
        filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')

    weights = list(range(len(available_cols), 0, -1))
    filtered_data['trust_score'] = (
            sum(filtered_data[c] * w for c, w in zip(available_cols, weights))
            / filtered_data[available_cols].sum(axis=1)
    )

    return filtered_data


def aggregate_monthly(df):
    """Groups by month_year and returns average trust_score per month."""
    if df.empty:
        return pd.DataFrame(columns=['month_year', 'trust_score'])
    return (
        df.groupby('month_year', as_index=False)['trust_score']
        .mean()
        .sort_values('month_year')
    )


def load_data():
    """Load and prepare all institution data."""
    try:
        pm_data = pd.read_excel(PM_DATA_PATH)
        army_data = pd.read_excel(ARMY_DATA_PATH)
        police_data = pd.read_excel(POLICE_DATA_PATH)
        govt_data = pd.read_excel(GOVT_DATA_PATH)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Prepare row-level data
    pm_rows = prepare_monthly_data_cases(pm_data, keyword="Prime Minister", columns=["a1", "a2", "a3", "a4"])
    army_rows = prepare_monthly_data_cases(army_data, keyword="IDF", columns=["a1", "a2", "a3", "a4"])
    police_rows = prepare_monthly_data_cases(police_data, keyword="Police", columns=["a1", "a2", "a3", "a4"])
    govt_rows = prepare_monthly_data_cases(govt_data, keyword="Government", columns=["a1", "a2", "a3", "a4"])

    # Monthly aggregates
    pm_monthly = aggregate_monthly(pm_rows)
    army_monthly = aggregate_monthly(army_rows)
    police_monthly = aggregate_monthly(police_rows)
    govt_monthly = aggregate_monthly(govt_rows)

    # Convert to string format for unified months
    for df in [pm_monthly, army_monthly, police_monthly, govt_monthly]:
        df['month_year_str'] = df['month_year'].astype(str)

    # Get union of all months
    all_months = sorted(set().union(
        pm_monthly['month_year_str'],
        army_monthly['month_year_str'],
        police_monthly['month_year_str'],
        govt_monthly['month_year_str']
    ))

    # Convert to dictionaries for lookup
    def df_to_dict(df):
        return dict(zip(df['month_year_str'], df['trust_score']))

    pm_dict = df_to_dict(pm_monthly)
    army_dict = df_to_dict(army_monthly)
    police_dict = df_to_dict(police_monthly)
    govt_dict = df_to_dict(govt_monthly)

    # Build final score lists
    pm_scores = [pm_dict.get(m, None) for m in all_months]
    army_scores = [army_dict.get(m, None) for m in all_months]
    police_scores = [police_dict.get(m, None) for m in all_months]
    govt_scores = [govt_dict.get(m, None) for m in all_months]

    return {
        "pm_rows": pm_rows,
        "army_rows": army_rows,
        "police_rows": police_rows,
        "govt_rows": govt_rows,
        "months": all_months,
        "pm_scores": pm_scores,
        "army_scores": army_scores,
        "police_scores": police_scores,
        "govt_scores": govt_scores,
    }


def rocket_strikes_page():
    """Rocket Strikes and Sentiments Page"""
    st.header("Rocket Strikes and Regional Safety Sentiments")

    @st.cache_data
    def load_strike_data():
        gdf = gpd.read_file("hellel_data/Mechozot_all/Mechozot_all.shp")
        counties_data = pd.read_csv("hellel_data/CountiesData.csv")
        alarms_data = pd.read_csv("hellel_data/AlarmsData.csv")

        # Convert dates
        counties_data["Date"] = pd.to_datetime(counties_data["Date"], dayfirst=True).dt.date
        alarms_data["date"] = pd.to_datetime(alarms_data["date"], errors="coerce", dayfirst=True).dt.date
        alarms_data["time"] = pd.to_datetime(alarms_data["time"], errors="coerce").dt.time

        return gdf, counties_data, alarms_data

    gdf, counties_data, alarms_data = load_strike_data()

    # Time Period Selection
    st.sidebar.write("Select Time Period")
    start_date, end_date = st.sidebar.slider(
        "Date Range",
        value=(date(2023, 10, 1), date(2024, 12, 31)),
        format="YYYY-MM",
        key="time_range"
    )

    # Filter data
    filtered_counties = counties_data[
        (counties_data["Date"] >= start_date) &
        (counties_data["Date"] <= end_date)
        ]

    filtered_alarms = alarms_data[
        (alarms_data["date"] >= filtered_counties["Date"].min() - pd.Timedelta(weeks=1)) &
        (alarms_data["date"] <= end_date)
        ]

    # Calculate safety percentages
    feel_safe = (
        filtered_counties.groupby("machoz")
        .apply(lambda x: (x.iloc[:, 2:4].sum(axis=1).mean() * 100).round(3))
        .reset_index(name="feel_safe_percentage")
    )

    # Calculate detailed responses per region
    column_avgs = (
        filtered_counties.groupby("machoz")
        .apply(lambda x: (x.iloc[:, 2:8].mean() * 100).round(decimals=3))
        .reset_index()
    )

    # Merge data
    curr_data = pd.merge(feel_safe, column_avgs, on="machoz")
    merged_gdf = gdf.merge(curr_data, on="machoz", how="left").to_crs(epsg=4326)

    # Create map visualization
    fig = create_rocket_strikes_map(merged_gdf, filtered_alarms)
    st.plotly_chart(fig, use_container_width=True)


def create_rocket_strikes_map(merged_gdf, filtered_alarms):
    """Create the rocket strikes map visualization"""
    # Create choropleth layer
    choropleth = go.Choroplethmapbox(
        geojson=json.loads(merged_gdf.to_json()),
        locations=merged_gdf["machoz"],
        featureidkey="properties.machoz",
        z=merged_gdf["feel_safe_percentage"],
        colorscale="RdBu",
        colorbar=dict(title="Feel Safe (%)"),
        marker_opacity=0.9,
        hovertemplate=(
                "%{properties.machoz}<br>" +
                "Don't know: %{properties.Unknown}<br>" +
                "Low: %{properties.Low}<br>" +
                "Very Low: %{properties.Very_Low}<br>" +
                "Medium: %{properties.Medium}<br>" +
                "High: %{properties.High}<br>" +
                "Very High: %{properties.Very_High}"
        )
    )

    # Create scatter layer for strikes
    scatter = go.Scattermapbox(
        lat=filtered_alarms["outLat"],
        lon=filtered_alarms["outLong"],
        mode="markers",
        marker=dict(size=5, color="red"),
        text=filtered_alarms["data"],
        hoverinfo="text"
    )

    # Combine layers
    fig = go.Figure(data=[choropleth, scatter])

    # Update layout
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center={"lat": 31.5, "lon": 35},
            zoom=7
        ),
        width=900,
        height=750,
        title="Rocket Strikes and Regional Safety Sentiments"
    )

    return fig


def trust_in_institutions_page():
    """Trust in Institutions Page"""
    st.header("Trust in Institutions Over Time")

    # Load data
    data = load_data()

    # Institution selection
    inst_options = {
        "Prime Minister": "pm",
        "IDF": "army",
        "Police": "police",
        "Government": "govt"
    }

    chosen_insts = st.multiselect(
        "Select Institutions:",
        options=list(inst_options.keys()),
        default=list(inst_options.keys())
    )

    if not chosen_insts:
        st.warning("Please select at least one institution.")
        return

    selected_keys = [inst_options[i] for i in chosen_insts]

    # Create main time series chart
    main_chart = create_trust_time_chart(data, selected_keys)
    st.plotly_chart(main_chart, use_container_width=True)

    # Demographic breakdown
    st.subheader("Trust by Demographics")
    demo_choice = st.selectbox(
        "Select Demographic Category:",
        ["Gender", "Region", "Religious Observance", "Political Stance", "Age Group"]
    )

    # Create demographic breakdown chart
    demo_chart = create_demographic_chart(data, demo_choice, selected_keys)
    st.plotly_chart(demo_chart, use_container_width=True)


def create_trust_time_chart(data, selected_institutions):
    """Create main time series chart for trust in institutions"""
    colors = {
        'pm': '#1f77b4',
        'army': '#2ca02c',
        'police': '#d62728',
        'govt': '#ff7f0e'
    }

    inst_to_scores = {
        'pm': data['pm_scores'],
        'army': data['army_scores'],
        'police': data['police_scores'],
        'govt': data['govt_scores']
    }

    inst_names = {
        'pm': 'Prime Minister',
        'army': 'IDF',
        'police': 'Police',
        'govt': 'Government'
    }

    fig = go.Figure()

    for inst in selected_institutions:
        y_vals = inst_to_scores[inst]
        fig.add_trace(go.Scatter(
            x=data['months'],
            y=y_vals,
            mode="lines+markers",
            name=inst_names[inst],
            line=dict(color=colors.get(inst, "#000"), width=2),
            marker=dict(size=6),
            hovertemplate=(
                    f"<b>{inst_names[inst]}</b><br>" +
                    "Month: %{x}<br>" +
                    "Trust Score: %{y:.2f}<extra></extra>"
            )
        ))

    fig.update_layout(
        title="Trust in Institutions Over Time",
        xaxis_title="Month",
        yaxis_title="Trust Score",
        yaxis=dict(range=[0, 5]),
        legend=dict(x=1.0, y=1.0, xanchor='left', yanchor='top'),
        margin=dict(l=60, r=60, t=80, b=80),
        height=450,
    )

    return fig


def create_demographic_chart(data, demo_choice, selected_institutions):
    """Create chart showing trust breakdown by demographic category"""
    demo_mappings = {
        "Gender": {
            "Male": "Male",
            "Female": "Female"
        },
        "Region": {
            "North": "North",
            "Haifa": "Haifa",
            "Central": "Central",
            "Tel Aviv": "Tel Aviv",
            "Jerusalem": "Jerusalem",
            "West Bank": "West Bank",
            "South": "South"
        },
        "Religious Observance": {
            "Secular": "Secular",
            "Traditional": "Traditional",
            "Religious": "Religious",
            "Ultra-Orthodox": "Ultra-Orthodox"
        },
        "Political Stance": {
            "Right": "Right",
            "Center": "Center",
            "Left": "Left",
            "Refused": "Refused"
        },
        "Age Group": {
            "18-24": "18-24",
            "25-34": "25-34",
            "35-44": "35-44",
            "45-54": "45-54",
            "55-64": "55-64",
            "65-74": "65-74",
            "75+": "75+"
        }
    }

    colors = {
        'Male': '#1f77b4',
        'Female': '#ff7f0e',
        'Right': '#2ca02c',
        'Center': '#d62728',
        'Left': '#9467bd',
        'Refused': '#8c564b',
    }

    fig = go.Figure()

    selected_mapping = demo_mappings[demo_choice]

    for inst in selected_institutions:
        # Get appropriate dataset
        if inst == 'pm':
            df = data['pm_rows']
        elif inst == 'army':
            df = data['army_rows']
        elif inst == 'police':
            df = data['police_rows']
        elif inst == 'govt':
            df = data['govt_rows']
        else:
            continue

        demo_column = 'sub_subject'  # Adjust based on your actual column name

        for display_label, data_value in selected_mapping.items():
            df_filtered = df[df[demo_column] == data_value].copy()
            if df_filtered.empty:
                continue

            monthly_avg = (
                df_filtered.groupby('month_year', as_index=False)['trust_score']
                .mean()
                .sort_values('month_year')
            )
            monthly_avg['month_year_str'] = monthly_avg['month_year'].astype(str)

            score_dict = dict(zip(monthly_avg['month_year_str'], monthly_avg['trust_score']))
            y_vals = [score_dict.get(m, None) for m in data['months']]

            line_name = f"{inst.upper()} ({display_label})"
            color = colors.get(display_label, "#000000")

            fig.add_trace(go.Scatter(
                x=data['months'],
                y=y_vals,
                mode="lines+markers",
                name=line_name,
                line=dict(color=color),
                marker=dict(size=6),
                hovertemplate=(
                        f"<b>{line_name}</b><br>" +
                        "Month: %{x}<br>" +
                        "Trust Score: %{y:.2f}<extra></extra>"
                )
            ))

    fig.update_layout(
        title=f"Trust in Institutions by {demo_choice}",
        xaxis_title="Month",
        yaxis_title="Trust Score",
        yaxis=dict(range=[0, 5]),
        legend=dict(x=1.0, y=1.0, xanchor='left', yanchor='top'),
        margin=dict(l=60, r=60, t=80, b=80),
        height=450,
    )

    return fig
def solidarity_page():
    """Solidarity in Israeli Society Page"""
    st.header("Social Solidarity Indicators")

    # Survey selection
    survey_options = {
        'Solidarity': 'solidarity_data/solidarity.xlsx',
        'Social State': 'solidarity_data/matzv_chvrati.xlsx',
        'Crisis': 'solidarity_data/mashber.xlsx'
    }

    selected_survey = st.selectbox(
        'Select Survey:',
        list(survey_options.keys())
    )

    # Visualization type
    viz_type = st.radio(
        "Select Visualization Type:",
        ["Bar Chart", "Time Series"]
    )

    # Load and process survey data
    try:
        df = pd.read_excel(survey_options[selected_survey])
        df['date'] = pd.to_datetime(df['date'])
    except FileNotFoundError:
        st.error(f"Survey data not found: {survey_options[selected_survey]}")
        return

    # Question selection
    questions_map = {
        'Gender': 'Gender',
        'Combat Participation': 'Are you or immediate family members participating in combat?',
        'Border Residence': 'Do you or immediate family live in border regions?'
    }

    selected_question = st.selectbox(
        'Select Question:',
        list(questions_map.keys())
    )

    # Create visualization
    if viz_type == "Bar Chart":
        create_solidarity_bar_chart(df, questions_map[selected_question])
    else:
        create_solidarity_time_series(df, questions_map[selected_question])


def main():
    """Main application entry point"""
    st.title("Israeli Public Opinion Dashboard")
    st.sidebar.title("Navigation")

    # Page selection
    page = st.sidebar.selectbox(
        "Select View",
        [
            "Rocket Strikes and Safety",
            "Trust in Institutions",
            "Social Solidarity"
        ]
    )

    # Route to appropriate page
    if page == "Rocket Strikes and Safety":
        rocket_strikes_page()
    elif page == "Trust in Institutions":
        trust_in_institutions_page()
    elif page == "Social Solidarity":
        solidarity_page()


if __name__ == "__main__":
    main()