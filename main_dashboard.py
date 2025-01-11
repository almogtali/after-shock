import streamlit as st
import plotly.graph_objects as go
import pandas as pd

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
        bibi_data     = pd.read_excel(bibi_data_path)
        tzal_data     = pd.read_excel(tzal_data_path)
        mishtara_data = pd.read_excel(mishtara_data_path)
        memshala_data = pd.read_excel(memshala_data_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # 2) Prepare row-level data
    bibi_rows = prepare_monthly_data_cases(bibi_data,     keyword="ראש ממשלה", columns=["a1","a2","a3","a4"])
    tzal_rows = prepare_monthly_data_cases(tzal_data,     keyword="צהל",       columns=["a1","a2","a3","a4"])
    mish_rows = prepare_monthly_data_cases(mishtara_data, keyword="משטרה",     columns=["a1","a2","a3","a4"])
    mems_rows = prepare_monthly_data_cases(memshala_data, keyword="ממשלה",     columns=["a1","a2","a3","a4"])

    # 3) For overall monthly aggregates (top chart)
    bibi_monthly     = aggregate_monthly(bibi_rows)
    tzal_monthly     = aggregate_monthly(tzal_rows)
    mish_monthly     = aggregate_monthly(mish_rows)
    memshala_monthly = aggregate_monthly(mems_rows)

    # 4) Convert month_year -> string for union of months
    bibi_monthly['month_year_str']     = bibi_monthly['month_year'].astype(str)
    tzal_monthly['month_year_str']     = tzal_monthly['month_year'].astype(str)
    mish_monthly['month_year_str']     = mish_monthly['month_year'].astype(str)
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
    bibi_dict     = df_to_dict(bibi_monthly)
    tzal_dict     = df_to_dict(tzal_monthly)
    mish_dict     = df_to_dict(mish_monthly)
    memshala_dict = df_to_dict(memshala_monthly)

    # Build final numeric lists for top chart
    bibi_scores     = [bibi_dict.get(m, None) for m in all_months]
    tzal_scores     = [tzal_dict.get(m, None) for m in all_months]
    mish_scores     = [mish_dict.get(m, None) for m in all_months]
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
def main_time_chart(data, months, selected_institutions):
    top_colors = {
        'bibi': '#1f77b4',
        'tzahal': '#2ca02c',
        'mishtara': '#d62728',
        'memshala': '#ff7f0e'
    }

    fig = go.Figure()

    inst_to_key = {
        'bibi':      data['bibi_scores'],
        'tzahal':    data['tzal_scores'],
        'mishtara':  data['mish_scores'],
        'memshala':  data['memshala_scores'],
    }

    for inst in selected_institutions:
        y_vals = inst_to_key[inst]
        fig.add_trace(go.Scatter(
            x=months,
            y=y_vals,
            mode="lines+markers",
            name=inst.capitalize(),
            line=dict(color=top_colors.get(inst, "#000"), width=2),
            marker=dict(size=6),
            hovertemplate=(
                f"<b>{inst.capitalize()}</b><br>"
                + "Month: %{x}<br>"
                + "Trust Score: %{y:.2f}<extra></extra>"
            )
        ))

    fig.update_layout(
        title="Overall Trust in Institutions Over Time",
        xaxis_title="Month",
        yaxis_title="Trust Score",
        yaxis=dict(range=[0, 5]),
        legend=dict(x=1.0, y=1.0, xanchor='left', yanchor='top'),
        margin=dict(l=60, r=60, t=80, b=80),
        height=450,
    )
    return fig

#######################################################################
# --- PLACEHOLDERS FOR OTHER VISUALIZATIONS ---------------------------
#######################################################################
def rocket_strikes_map(_):
    st.info("This visualization is not implemented yet.")

def significant_events_chart(_):
    st.info("This visualization is not implemented yet.")

def solidarity_in_israeli_society(_):
    st.info("This visualization is not implemented yet.")

##############################
# --- STREAMLIT APP LAYOUT ---
##############################
st.title("Israeli Sentiments Dashboard")
st.sidebar.title("Visualizations")

visualization = st.sidebar.selectbox(
    "Choose Visualization",
    [
        "Rocket Strikes and Sentiments",
        "Significant Events Analysis",
        "Trust in Institutions Over Time",
        "Solidarity in Israeli Society"
    ]
)

institutions_data = load_data()
months = institutions_data["months"]

if visualization == "Rocket Strikes and Sentiments":
    st.header("Rocket Strikes and Northern Residents' Sentiments")
    rocket_strikes_map(None)

elif visualization == "Significant Events Analysis":
    st.header("Significant Events and Sentiments")
    significant_events_chart(None)

elif visualization == "Trust in Institutions Over Time":
    st.header("Trust in Institutions Over Time")

    inst_options = {
        "Bibi (ראש ממשלה)": "bibi",
        "Tzahal (צה\"ל)":    "tzahal",
        "Mishtara (משטרה)":  "mishtara",
        "Memshala (ממשלה)":  "memshala",
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
    solidarity_in_israeli_society(None)
