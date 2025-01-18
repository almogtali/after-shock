import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import plotly.subplots as sp
import plotly.express as px
import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
from datetime import date
from pyngrok import ngrok
import subprocess
import json

############################
# --- FILE PATHS (EXAMPLE) -
############################
bibi_data_path = "data_storage/bibi.xlsx"
tzal_data_path = "data_storage/tzal.xlsx"
mishtara_data_path = "data_storage/mishtra.xlsx"
memshala_data_path = "data_storage/memshla.xlsx"


###################################
# --- HELPER: KEEP ONLY "פוליטית מקובץ"
###################################
def keep_only_politit_mekuvatz(df):
    """
    1) Keep only rows where subject == "פוליטית מקובץ".
    2) Convert certain sub_subject values (e.g., "ימין מתון" -> "ימין", etc.) if needed.
    3) Keep only sub_subject in {"ימין", "מרכז", "שמאל", "מסרב"}.
    """
    if 'subject' not in df.columns or 'sub_subject' not in df.columns:
        # If these columns don't exist in this dataset, just return df as-is
        return df

    # 1. Keep only subject == "פוליטית מקובץ"
    df = df[df['subject'] == "פוליטית מקובץ"].copy()

    # 2. If needed, unify sub_subject values:
    replacements = {
        "ימין מתון": "ימין",
        "שמאל מתון": "שמאל",
        # Add other replacements as needed
    }
    df['sub_subject'] = df['sub_subject'].replace(replacements)

    # 3. Keep only the final four categories
    valid_stances = {"ימין", "מרכז", "שמאל", "מסרב"}
    df = df[df['sub_subject'].isin(valid_stances)]

    return df


#################################################
# --- FUNCTION: PREPARE MONTHLY DATA (HEBREW) ---
#################################################
def prepare_monthly_data_cases(data, keyword, columns):
    """
    Filters rows where 'q_full' contains the keyword (institution name),
    then computes a weighted trust score per row. Finally returns a
    row-level DataFrame with:
      - all original columns,
      - an added 'trust_score' column,
      - a 'month_year' (Period) column.
    """
    # 1) Filter by institution (keyword)
    filtered_data = data[data['q_full'].str.contains(keyword, case=False, na=False)].copy()

    # 2) Example usage: if you specifically want to keep only "פוליטית מקובץ" rows
    #    for stance-based filtering, uncomment the next line:
    #    filtered_data = keep_only_politit_mekuvatz(filtered_data)

    # 3) Convert 'date' to datetime and keep valid rows
    filtered_data['date'] = pd.to_datetime(filtered_data['date'], errors='coerce')
    filtered_data.dropna(subset=['date'], inplace=True)

    # 4) Create a month_year period
    filtered_data['month_year'] = filtered_data['date'].dt.to_period('M')

    # 5) Make sure trust columns are numeric
    available_cols = [c for c in columns if c in filtered_data.columns]
    if not available_cols:
        return pd.DataFrame(columns=['month_year', 'trust_score'])

    for col in available_cols:
        filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')

    # 6) Weighted trust score
    weights = list(range(len(available_cols), 0, -1))
    # sum(...) is a generator expression for each row
    filtered_data['trust_score'] = (
            sum(filtered_data[c] * w for c, w in zip(available_cols, weights))
            / filtered_data[available_cols].sum(axis=1)
    )

    return filtered_data


def aggregate_monthly(df):
    """ Groups by month_year and returns average trust_score per month. """
    if df.empty:
        return pd.DataFrame(columns=['month_year', 'trust_score'])
    return (
        df.groupby('month_year', as_index=False)['trust_score']
        .mean()
        .sort_values('month_year')
    )


###########################
# --- LOAD ALL THE DATA ---
###########################
def load_data():
    """Load data from Excel and keep row-level trust data for each institution."""
    # 1) Read Excel
    try:
        bibi_data = pd.read_excel(bibi_data_path)
        tzal_data = pd.read_excel(tzal_data_path)
        mishtara_data = pd.read_excel(mishtara_data_path)
        memshala_data = pd.read_excel(memshala_data_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # 2) Prepare row-level data
    bibi_rows = prepare_monthly_data_cases(bibi_data, keyword="ראש ממשלה", columns=["a1", "a2", "a3", "a4"])
    tzal_rows = prepare_monthly_data_cases(tzal_data, keyword="צהל", columns=["a1", "a2", "a3", "a4"])
    mish_rows = prepare_monthly_data_cases(mishtara_data, keyword="משטרה", columns=["a1", "a2", "a3", "a4"])
    mems_rows = prepare_monthly_data_cases(memshala_data, keyword="ממשלה", columns=["a1", "a2", "a3", "a4"])

    # 3) For overall monthly aggregates (top chart)
    bibi_monthly = aggregate_monthly(bibi_rows)
    tzal_monthly = aggregate_monthly(tzal_rows)
    mish_monthly = aggregate_monthly(mish_rows)
    memshala_monthly = aggregate_monthly(mems_rows)

    # 4) Convert month_year -> string for union of months
    bibi_monthly['month_year_str'] = bibi_monthly['month_year'].astype(str)
    tzal_monthly['month_year_str'] = tzal_monthly['month_year'].astype(str)
    mish_monthly['month_year_str'] = mish_monthly['month_year'].astype(str)
    memshala_monthly['month_year_str'] = memshala_monthly['month_year'].astype(str)

    all_months = sorted(
        set(bibi_monthly['month_year_str'])
        .union(tzal_monthly['month_year_str'])
        .union(mish_monthly['month_year_str'])
        .union(memshala_monthly['month_year_str'])
    )

    def df_to_dict(df):
        return dict(zip(df['month_year_str'], df['trust_score']))

    # Build dictionary for each institution
    bibi_dict = df_to_dict(bibi_monthly)
    tzal_dict = df_to_dict(tzal_monthly)
    mish_dict = df_to_dict(mish_monthly)
    memshala_dict = df_to_dict(memshala_monthly)

    # Build final numeric lists for top chart
    bibi_scores = [bibi_dict.get(m, None) for m in all_months]
    tzal_scores = [tzal_dict.get(m, None) for m in all_months]
    mish_scores = [mish_dict.get(m, None) for m in all_months]
    memshala_scores = [memshala_dict.get(m, None) for m in all_months]

    return {
        "bibi_rows": bibi_rows,
        "tzal_rows": tzal_rows,
        "mish_rows": mish_rows,
        "mems_rows": mems_rows,
        "months": all_months,
        "bibi_scores": bibi_scores,
        "tzal_scores": tzal_scores,
        "mish_scores": mish_scores,
        "memshala_scores": memshala_scores,
    }


###############################################################################
# --- DEMO CHART: GROUP-BY A DEMO CATEGORY (E.G. פוליטית מקובץ) -------------
###############################################################################
def demo_chart(row_data, months, institutions, demo_category, demo_subgroups_map):
    """
    row_data: dict from load_data
    demo_category: which column in the DataFrame to group by, e.g. 'subject' or 'sub_subject'
    demo_subgroups_map: dictionary of displayed_label -> actual sub_subject string in the data
    """
    color_map = {
        'זכר': '#1f77b4',
        'נקבה': '#ff7f0e',
        # ... add more as needed ...
        'ימין': '#2ca02c',
        'מרכז': '#d62728',
        'שמאל': '#9467bd',
        'מסרב': '#8c564b',
        # etc...
    }

    fig = go.Figure()

    for inst in institutions:
        # pick row-level DataFrame
        if inst == 'bibi':
            df = row_data['bibi_rows']
        elif inst == 'tzahal':
            df = row_data['tzal_rows']
        elif inst == 'mishtara':
            df = row_data['mish_rows']
        elif inst == 'memshala':
            df = row_data['mems_rows']
        else:
            continue

        # Suppose your code for the demographic column is actually 'sub_subject'
        # or possibly 'subject' if you're storing stances there. Adjust as needed:
        demo_column = 'sub_subject'

        for label, real_val in demo_subgroups_map.items():
            # Filter
            df_sub = df[df[demo_column] == real_val].copy()
            if df_sub.empty:
                continue

            # Group by month
            monthly_avg = (
                df_sub.groupby('month_year', as_index=False)['trust_score']
                .mean()
                .sort_values('month_year')
            )
            monthly_avg['month_year_str'] = monthly_avg['month_year'].astype(str)

            score_dict = dict(zip(monthly_avg['month_year_str'], monthly_avg['trust_score']))
            y_vals = [score_dict.get(m, None) for m in months]

            line_name = f"{inst.capitalize()} ({label})"
            color_to_use = color_map.get(real_val, "#000000")

            fig.add_trace(go.Scatter(
                x=months,
                y=y_vals,
                mode="lines+markers",
                name=line_name,
                line=dict(color=color_to_use),
                marker=dict(size=6),
                hovertemplate=(
                        f"<b>{line_name}</b><br>"
                        + "Month: %{x}<br>"
                        + "Trust Score: %{y:.2f}<extra></extra>"
                )
            ))

    fig.update_layout(
        title="Trust in Institutions by Demographic Subgroup",
        xaxis_title="Month",
        yaxis_title="Trust Score",
        yaxis=dict(range=[0, 5]),
        legend=dict(x=1.0, y=1.0, xanchor='left', yanchor='top'),
        margin=dict(l=60, r=60, t=80, b=80),
        height=450,
    )

    return fig


#######################################################################
# --- OVERALL TIME CHART (TOP) ----------------------------------------
#######################################################################
# def main_time_chart(data, months, selected_institutions):
#     top_colors = {
#         'bibi': '#1f77b4',
#         'tzahal': '#2ca02c',
#         'mishtara': '#d62728',
#         'memshala': '#ff7f0e'
#     }
#
#     fig = go.Figure()
#
#     inst_to_key = {
#         'bibi': data['bibi_scores'],
#         'tzahal': data['tzal_scores'],
#         'mishtara': data['mish_scores'],
#         'memshala': data['memshala_scores'],
#     }
#
#     for inst in selected_institutions:
#         y_vals = inst_to_key[inst]
#         fig.add_trace(go.Scatter(
#             x=months,
#             y=y_vals,
#             mode="lines+markers",
#             name=inst.capitalize(),
#             line=dict(color=top_colors.get(inst, "#000"), width=2),
#             marker=dict(size=6),
#             hovertemplate=(
#                     f"<b>{inst.capitalize()}</b><br>"
#                     + "Month: %{x}<br>"
#                     + "Trust Score: %{y:.2f}<extra></extra>"
#             )
#         ))
#
#     fig.update_layout(
#         title="Overall Trust in Institutions Over Time",
#         xaxis_title="Month",
#         yaxis_title="Trust Score",
#         yaxis=dict(range=[0, 5]),
#         legend=dict(x=1.0, y=1.0, xanchor='left', yanchor='top'),
#         margin=dict(l=60, r=60, t=80, b=80),
#         height=450,
#     )
#     return fig
def main_time_chart(data, months, institutions):
    """
    Visualize trust distribution for a single institution over time using pie charts.
    Each row contains up to 7 months.
    """
    import math

    colors = {
        1: '#d62728',  # Red for "low trust"
        2: '#ff7f0e',  # Orange
        3: '#2ca02c',  # Light green
        4: '#1f77b4',  # Dark green for "high trust"
    }

    inst_to_key = {
        'bibi': data['bibi_rows'],
        'tzahal': data['tzal_rows'],
        'mishtara': data['mish_rows'],
        'memshala': data['mems_rows'],
    }

    # Let the user select a single institution
    selected_institution = st.selectbox(
        "Choose an institution to display:",
        institutions,
        format_func=lambda x: {
            'bibi': 'Bibi (\u05e8\u05d0\u05e9 \u05de\u05de\u05e9\u05dc\u05d4)',
            'tzahal': 'Tzahal (\u05e6\u05d4\"\u05dc)',
            'mishtara': 'Mishtara (\u05de\u05e9\u05d8\u05e8\u05d4)',
            'memshala': 'Memshala (\u05de\u05de\u05e9\u05dc\u05d4)'
        }.get(x, x)
    )

    # Retrieve data for the selected institution
    df = inst_to_key[selected_institution]

    # Filter months with available data
    valid_months = []
    for month in months:
        if 'month_year' not in df.columns:
            continue
        monthly_data = df[df['month_year'].astype(str) == month]
        if not monthly_data.empty:
            valid_months.append(month)

    # Determine the number of rows needed for all pie charts
    pies_per_row = 7  # Maximum pies in one row
    rows_needed = math.ceil(len(valid_months) / pies_per_row)

    # Create specs dynamically
    specs = [
        [{'type': 'domain'} for _ in range(pies_per_row)]
        for _ in range(rows_needed)
    ]

    # Prepare subplots
    fig = sp.make_subplots(
        rows=rows_needed,
        cols=pies_per_row,
        specs=specs,
        subplot_titles=[f"{month}" for month in valid_months],
    )

    # Add pie charts for each month
    for idx, month in enumerate(valid_months):
        # Filter the data for the current month
        monthly_data = df[df['month_year'].astype(str) == month]

        # Calculate distribution for trust levels (1-4)
        trust_distribution = monthly_data[['a1', 'a2', 'a3', 'a4']].sum().values
        trust_labels = [f"Level {i}" for i in range(1, 5)]

        # Skip if no data available for this month
        if not trust_distribution.any():
            continue

        # Determine row and column position for the pie chart
        row = idx // pies_per_row + 1
        col = idx % pies_per_row + 1

        # Create a pie chart
        pie_chart = go.Pie(
            labels=trust_labels,
            values=trust_distribution,
            marker=dict(colors=[colors[i] for i in range(1, 5)]),
            name=f"{selected_institution.capitalize()} - {month}",
            hovertemplate=(
                f"{selected_institution.capitalize()}<br>"
                + "Level: %{label}<br>"
                + "Count: %{value}<extra></extra>"
            )
        )

        # Add pie chart to the subplot
        fig.add_trace(pie_chart, row=row, col=col)

    # Update layout
    fig.update_layout(
        title=f"Trust Distribution for {selected_institution.capitalize()} by Month",
        height=250 * rows_needed,  # Adjust height dynamically
        showlegend=False,
        margin=dict(t=50, l=20, r=20, b=20)
    )

    #st.plotly_chart(fig, use_container_width=True)
    return fig



def rocket_strikes_map():
    # Load data
    @st.cache_data
    def load_strike_data():
        gdf = gpd.read_file("hellel_data/Mechozot_all/Mechozot_all.shp")  # Replace with your path
        counties_data = pd.read_csv("hellel_data/CountiesData.csv")
        alarms_data = pd.read_csv("hellel_data/AlarmsData.csv")
        counties_data["Date"] = pd.to_datetime(counties_data["Date"], dayfirst=True).dt.date
        alarms_data["date"] = pd.to_datetime(alarms_data["date"], errors="coerce", dayfirst=True).dt.date
        alarms_data["time"] = pd.to_datetime(alarms_data["time"], errors="coerce").dt.time
        return gdf, counties_data, alarms_data

    gdf, counties_data, alarms_data = load_strike_data()

    # Sidebar: Time Period Selection
    st.sidebar.write("Time Period")
    start_date, end_date = st.sidebar.slider(
        "Select Date Range",
        value=(date(2023, 10, 1), date(2024, 12, 31)),
        format="YYYY-MM",
        key="time_range"
    )

    # Filter data based on date range
    filtered_counties = counties_data[(counties_data["Date"] >= start_date) & (counties_data["Date"] <= end_date)]
    filtered_alarms = alarms_data[(alarms_data["date"] >= filtered_counties["Date"].min() - pd.Timedelta(weeks=1)) &
                                  (alarms_data["date"] <= end_date)]

    # Calculate the percentage of the population that feels personal safety
    feel_safe = (
        filtered_counties.groupby("machoz").apply(
            lambda x: (x.iloc[:, 2:4].sum(axis=1).mean() * 100).round(3)
        ).reset_index(name="feel_safe_percentage")
    )

    # Calculate percentage per answer per county
    column_avgs = (
        filtered_counties.groupby("machoz").apply(
            lambda x: (x.iloc[:, 2:8].mean() * 100).round(decimals=3)
        ).reset_index()
    )

    curr_data = pd.merge(feel_safe, column_avgs, on="machoz")
    min_val = curr_data["feel_safe_percentage"].min()
    max_val = curr_data["feel_safe_percentage"].max()

    # Merge with GeoDataFrame
    merged_gdf = gdf.merge(curr_data, on="machoz", how="left").to_crs(epsg=4326)

    # Create the map
    choropleth = go.Choroplethmapbox(
        geojson=json.loads(merged_gdf.to_json()),
        locations=merged_gdf["machoz"],
        featureidkey="properties.machoz",
        z=merged_gdf["feel_safe_percentage"],
        zmin=min_val,
        zmax=max_val,
        colorscale="RdBu",
        colorbar=dict(title="Feel Safe (%)"),
        marker_opacity=0.9,
        hovertext=merged_gdf.apply(
            lambda row: f"{row['machoz']}<br>"
                        f"לא יודע: {row.get('לא יודע', 'N/A')}<br>"
                        f"נמוכה: {row.get('נמוכה', 'N/A')}<br>"
                        f"נמוכה מאוד: {row.get('נמוכה מאוד', 'N/A')}<br>"
                        f"בינונית: {row.get('בינונית', 'N/A')}<br>"
                        f"גבוהה: {row.get('גבוהה', 'N/A')}<br>"
                        f"גבוהה מאוד: {row.get('גבוהה מאוד', 'N/A')}",
            axis=1
        ),
        hoverinfo="text",
    )

    scatter = go.Scattermapbox(
        lat=filtered_alarms["outLat"],
        lon=filtered_alarms["outLong"],
        mode="markers",
        marker=go.scattermapbox.Marker(size=5, color="red"),
        text=filtered_alarms["data"],
        hoverinfo="text",
    )

    # Combine layers
    fig = go.Figure()
    fig.add_trace(choropleth)
    fig.add_trace(scatter)

    # Update Layout
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center={"lat": 31.5, "lon": 35},
            zoom=7,
        ),
        width=900,
        height=750,
        title="Rocket Strikes and Personal Safety Sentiments"
    )

    # Display the map
    st.plotly_chart(fig, use_container_width=True)


def significant_events_chart(_):
    st.info("This visualization is not implemented yet.")


def create_solidarity_dashboard():
    # File selection with predefined questions for each file
    file_options = {
        'סולידריות (Solidarity)': 'solidarity_data/solidarity.xlsx',
        'מצב חברתי (Social State)': 'solidarity_data/matzv_chvrati.xlsx',
        'משבר (Crisis)': 'solidarity_data/mashber.xlsx'
    }

    # Define response mappings for each question
    response_mappings = {
        'עד כמה את.ה מוטרד.ת או לא מוטרד.ת ממצבה החברתי של ישראל ביום שאחרי המלחמה דרג בסולם של 1-5, כאשר 5 = מוטרד '
        'מאד ו - 1 = לא מוטרד כלל': {
            'a1': 'לא מוטרד כלל',
            'a2': 'מעט מוטרד',
            'a3': 'מוטרד במידה בינונית',
            'a4': 'מוטרד',
            'a5': 'מוטרד מאד'
        },
        'עד כמה אתה אופטימי ביחס ליכולתה של החברה הישראלית להתאושש מהמשבר ולצמוח': {
            'a1': 'פסימי מאד',
            'a2': 'די פסימי',
            'a3': 'די אופטימי',
            'a4': 'אופטימי מאד'
        },
        'האם חל שינוי בתחושת הסולידריות בחברה הישראלית בעת הזו': {
            'a1': 'תחושת הסולידריות התחזקה מאד',
            'a2': 'תחושת הסולידריות די התחזקה',
            'a3': 'אין שינוי בתחושת הסולידריות',
            'a4': 'תחושת הסולידריות די פחתה',
            'a5': 'תחושת הסולידריות פחתה מאד'
        }
    }

    # Rest of the setup code remains the same
    predefined_questions = [
        'מגדר',
        'האם את.ה או בן משפחה בדרגה ראשונה שלך לוקח חלק בלחימה?',
        'האם את.ה או בן משפחה בדרגה ראשונה שלך מתגורר בעוטף עזה או בגבול הצפון?'
    ]

    selected_file = st.selectbox(
        'בחר סקר להצגה (Select survey to display):',
        list(file_options.keys())
    )

    viz_type = st.radio(
        "בחר סוג תצוגה (Select visualization type):",
        ["Bar Chart", "Line Plot"]
    )

    try:
        df = pd.read_excel(file_options[selected_file])
        df['date'] = pd.to_datetime(df['date'])
    except FileNotFoundError:
        st.error(f"File not found: {file_options[selected_file]}")
        return

    selected_question = st.selectbox(
        'בחר שאלה להצגה (Select question to display):',
        predefined_questions
    )

    question_data = df[df['subject'] == selected_question].copy()
    question_data = question_data[question_data['sub_subject'] != 'Total']

    full_question = df['q_full'].iloc[0] if not df.empty else None

    if viz_type == "Bar Chart":
        create_bar_chart(question_data, full_question, selected_question, response_mappings)
    else:
        create_line_plot(question_data, full_question, selected_question, response_mappings)


def create_bar_chart(question_data, full_question, selected_question, response_mappings):
    # Get numeric columns
    numeric_cols = [col for col in question_data.columns if col.startswith('a')]
    numeric_cols.sort()  # Sort normally first

    chart_data = (
        question_data.groupby('sub_subject', as_index=False)[numeric_cols]
        .mean()
    )

    fig = go.Figure()

    option1_color = '#082f49'  # Dark blue
    option2_color = '#f97316'  # Orange

    # Get categories and explicitly reverse them
    if full_question in response_mappings:
        categories = [response_mappings[full_question][col] for col in numeric_cols]
    else:
        categories = [f'Response {i}' for i in range(1, len(numeric_cols) + 1)]

    # Explicitly reverse both categories and numeric columns
    categories = categories[::-1]
    numeric_cols = numeric_cols[::-1]

    # Add bars with reversed values
    for idx, row in chart_data.iterrows():
        values = [row[col] * 100 for col in numeric_cols]
        fig.add_trace(go.Bar(
            name=row['sub_subject'],
            x=values,
            y=categories,
            orientation='h',
            marker_color=option1_color if idx == 0 else option2_color,
            hovertemplate="<b>%{y}</b><br>" +
                          row['sub_subject'] + ": %{x:.1f}%" +
                          "<extra></extra>"
        ))

    # Update layout
    fig.update_layout(
        title={
            'text': f'Survey Results by {selected_question}',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        barmode='group',
        height=600,
        showlegend=True,
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': -0.2,
            'xanchor': 'center',
            'x': 0.5
        },
        font=dict(size=14),
        margin=dict(t=100, b=100, l=300, r=50),
        paper_bgcolor='white',
        plot_bgcolor='rgba(248,249,250,0.5)',
        yaxis={'autorange': 'reversed'}  # This is the key change to invert the Y-axis
    )

    fig.update_xaxes(
        title_text='Percentage (%)',
        range=[0, 100],
        gridcolor='rgba(0,0,0,0.1)',
        showgrid=True,
        tickformat='.1f',
        zeroline=False
    )

    fig.update_yaxes(
        title_text='',
        gridcolor='rgba(0,0,0,0.1)',
        showgrid=False,
        automargin=True,
        tickfont=dict(size=14)
    )

    st.plotly_chart(fig, use_container_width=False, key="bar_chart")
def create_bar_chart(question_data, full_question, selected_question, response_mappings):
    # Restrict to numeric columns and calculate the mean
    numeric_cols = [col for col in question_data.columns if col.startswith('a')]
    chart_data = (
        question_data.groupby('sub_subject', as_index=False)[numeric_cols]
        .mean()
    )

    fig = go.Figure()

    # Define colors
    option1_color = '#082f49'  # Dark blue
    option2_color = '#f97316'  # Orange

    # Get response labels
    if full_question in response_mappings:
        categories = [response_mappings[full_question][f'a{i}']
                      for i in range(1, len(numeric_cols) + 1)]
    else:
        categories = [f'Response {i}' for i in range(1, len(numeric_cols) + 1)]

    # Adjust category order for specific questions
    if "מוטרד" in full_question:
        categories = ['מוטרד' ,'מעט מוטרד', 'מוטרד במידה בינונית','לא מוטרד כלל', 'מוטרד מאוד']
    elif "אופטימי" in full_question:
        categories = ['אופטימי מאוד', 'די אופטימי', 'די פסימי', 'פסימי מאוד']

    # Convert to categorical with explicit ordering
    categories = pd.Categorical(categories, categories=categories, ordered=True)

    # Reorder chart_data based on predefined category order
    sorted_chart_data = []
    for idx, row in chart_data.iterrows():
        values = [row[f'a{i}'] * 100 for i in range(1, len(numeric_cols) + 1)]
        sorted_chart_data.append(values)

    # Add bars with text labels
    for idx, row in chart_data.iterrows():
        values = sorted_chart_data[idx]
        fig.add_trace(go.Bar(
            name=row['sub_subject'],
            x=values,  # Ensure values are in correct order
            y=categories,  # Use ordered categories
            orientation='h',
            marker_color=option1_color if idx == 0 else option2_color,
            text=[f'{v:.1f}%' for v in values],  # Add percentage labels
            textposition='outside',  # Position labels at the end of bars
            textfont=dict(size=12),  # Set font size for labels
            hovertemplate="<b>%{y}</b><br>" +
                          row['sub_subject'] + ": %{x:.1f}%" +
                          "<extra></extra>"
        ))

    # Update layout
    fig.update_layout(
        title={
            'text': f'Survey Results by {selected_question}',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        barmode='group',
        height=600,
        showlegend=True,
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': -0.2,
            'xanchor': 'center',
            'x': 0.5
        },
        font=dict(size=14),
        margin=dict(t=100, b=100, l=300, r=100),  # Increased right margin for labels
        paper_bgcolor='white',
        plot_bgcolor='rgba(248,249,250,0.5)',
        uniformtext=dict(mode='hide', minsize=8)  # Ensure consistent label visibility
    )

    fig.update_xaxes(
        title_text='Percentage (%)',
        range=[0, 110],  # Increased range to accommodate labels
        gridcolor='rgba(0,0,0,0.1)',
        showgrid=True,
        tickformat='.1f',
        zeroline=False
    )

    fig.update_yaxes(
        title_text='',
        gridcolor='rgba(0,0,0,0.1)',
        showgrid=False,
        automargin=True,
        tickfont=dict(size=14)
    )

    # Only call st.plotly_chart here
    st.plotly_chart(fig, use_container_width=False, key="bar_chart")


def create_line_plot(question_data, full_question, selected_question, response_mappings):
    # Prepare data for time series
    numeric_cols = [col for col in question_data.columns if col.startswith('a')]

    fig = go.Figure()

    # Get response labels
    if full_question in response_mappings:
        response_labels = response_mappings[full_question]
    else:
        response_labels = {f'a{i}': f'Response {i}' for i in range(1, len(numeric_cols) + 1)}

    # Create a line for each response option
    colors = px.colors.qualitative.Set2
    for idx, sub_subject in enumerate(question_data['sub_subject'].unique()):
        sub_data = question_data[question_data['sub_subject'] == sub_subject].sort_values('date')

        for i, col in enumerate(numeric_cols):
            fig.add_trace(go.Scatter(
                x=sub_data['date'],
                y=sub_data[col] * 100,
                name=f"{sub_subject} - {response_labels[col]}",
                mode='lines+markers',
                line=dict(
                    color=colors[i % len(colors)],
                    dash='solid' if idx % 2 == 0 else 'dash'
                ),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>" +
                              "Value: %{y:.1f}%<br>" +
                              "<extra></extra>"
            ))

    # Dynamically set y-axis range based on data
    max_value = question_data[numeric_cols].max().max() * 100
    y_range_max = min(100, max_value + 10)  # Keep within 100% but add padding

    # Generate explicit tick values and labels for dates
    unique_dates = question_data['date'].dropna().sort_values().unique()
    tickvals = unique_dates
    ticktext = [date.strftime('%Y-%m-%d') for date in unique_dates]

    # Update layout specifically for line plot
    fig.update_layout(
        title={
            'text': f'Time Trends for {selected_question}',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        xaxis_title="Date",
        yaxis_title="Percentage (%)",
        height=600,
        showlegend=True,
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': -0.5,
            'xanchor': 'center',
            'x': 0.5
        },
        font=dict(size=14),
        margin=dict(t=100, b=150, l=100, r=50),
        paper_bgcolor='white',
        plot_bgcolor='rgba(248,249,250,0.5)'
    )

    fig.update_xaxes(
        gridcolor='rgba(0,0,0,0.1)',
        showgrid=True,
        tickvals=tickvals,
        ticktext=ticktext,
        tickangle=45  # Rotate labels for better readability
    )

    fig.update_yaxes(
        range=[0, y_range_max],
        gridcolor='rgba(0,0,0,0.1)',
        showgrid=True,
        tickformat='.1f'
    )

    st.plotly_chart(fig, use_container_width=False, key="line_plot")


def update_layout(fig, selected_question):
    fig.update_layout(
        title={
            'text': f'Survey Results by {selected_question}',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        barmode='group',
        height=600,
        showlegend=True,
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': -0.2,
            'xanchor': 'center',
            'x': 0.5
        },
        font=dict(size=14),
        margin=dict(t=100, b=100, l=300, r=50),
        paper_bgcolor='white',
        plot_bgcolor='rgba(248,249,250,0.5)'
    )

    fig.update_xaxes(
        title_text='Percentage (%)',
        range=[0, 100],
        gridcolor='rgba(0,0,0,0.1)',
        showgrid=True,
        tickformat='.1f',
        zeroline=False
    )

    fig.update_yaxes(
        title_text='',
        gridcolor='rgba(0,0,0,0.1)',
        showgrid=False,
        automargin=True,
        tickfont=dict(size=14)
    )

    st.plotly_chart(fig, use_container_width=False, key="layout")


##############################
# --- STREAMLIT APP LAYOUT ---
##############################
st.title("Israeli Sentiments Dashboard")
st.sidebar.title("Visualizations")

visualization = st.sidebar.radio(
    "Choose Visualization",
    [
        "Rocket Strikes and Sentiments",
        "Significant Events Analysis",
        "Trust in Institutions Over Time",
        "Solidarity in Israeli Society"
    ],
    label_visibility="visible"
)
institutions_data = load_data()
months = institutions_data["months"]

if visualization == "Rocket Strikes and Sentiments":
    st.header("Rocket Strikes and Northern Residents' Sentiments")
    rocket_strikes_map()

elif visualization == "Significant Events Analysis":
    st.header("Significant Events and Sentiments")
    significant_events_chart(None)

elif visualization == "Trust in Institutions Over Time":
    st.header("Trust in Institutions Over Time")

    inst_options = {
        "Prime Minister (Benjamin Netanyahu)": "bibi",
        "IDF (Israel Defense Forces)": "tzahal",
        "Police": "mishtara",
        "Government": "memshala",
    }

    chosen_insts = st.multiselect(
        "Select Institutions (top chart):",
        options=list(inst_options.keys()),
        default=list(inst_options.keys())
    )
    if not chosen_insts:
        st.warning("Please select at least one institution.")
        st.stop()

    selected_institutions_keys = [inst_options[i] for i in chosen_insts]

    # TOP CHART
    top_chart = main_time_chart(institutions_data, months, selected_institutions_keys)
    st.plotly_chart(top_chart, use_container_width=True)

    # BOTTOM DEMOGRAPHIC CHART
    st.subheader("Demographic Breakdown (Bottom Chart)")

    demo_choice = st.selectbox(
        "Choose one demographic dimension:",
        ["Gender (מגדר)", "District (מחוז)", "Religiousness (דתיות)", "Political stance", "Age (גיל)"]
    )

    if demo_choice == "Gender (מגדר)":
        sub_map = {"זכר": "זכר", "נקבה": "נקבה"}
    elif demo_choice == "District (מחוז)":
        sub_map = {
            "צפון": "צפון",
            "חיפה": "חיפה",
            "מרכז": "מרכז",
            "תל אביב": "תל אביב",
            "ירושלים": "ירושלים",
            "יהודה ושומרון": "יהודה ושומרון",
            "דרום": "דרום"
        }
    elif demo_choice == "Religiousness (דתיות)":
        sub_map = {
            "חילוני": "חילוני",
            "מסורתי": "מסורתי",
            "דתי": "דתי",
            "חרדי": "חרדי"
        }
    elif demo_choice == "Political stance":
        # If you want only the "פוליטית מקובץ" rows, you might set keep_only_politit_mekuvatz
        # inside prepare_monthly_data_cases or do a separate filter. Then map your final stances:
        sub_map = {
            "ימין": "ימין",
            "מרכז": "מרכז",
            "שמאל": "שמאל",
            "מסרב": "מסרב"
        }
    elif demo_choice == "Age (גיל)":
        sub_map = {
            "18-24": "18-24",
            "25-34": "25-34",
            "35-44": "35-44",
            "45-54": "45-54",
            "55-64": "55-64",
            "65-74": "65-74",
            "75+": "75+"
        }
    else:
        sub_map = {}

    bottom_fig = demo_chart(
        row_data=institutions_data,
        months=months,
        institutions=selected_institutions_keys,
        demo_category=demo_choice,  # or "sub_subject"
        demo_subgroups_map=sub_map
    )
    st.plotly_chart(bottom_fig, use_container_width=True)

elif visualization == "Solidarity in Israeli Society":
    st.header("Solidarity in Israeli Society")
    create_solidarity_dashboard()
