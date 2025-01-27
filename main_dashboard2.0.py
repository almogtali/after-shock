import numpy as np
from streamlit_plotly_events import plotly_events
import geopandas as gpd
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def create_trust_dashboard(bibi_data_path, tzal_data_path, mishtara_data_path, memshala_data_path):
    """
    Creates a complete trust dashboard component for Streamlit.

    Parameters:
    bibi_data_path (str): Path to Prime Minister data Excel file
    tzal_data_path (str): Path to IDF data Excel file
    mishtara_data_path (str): Path to Police data Excel file
    memshala_data_path (str): Path to Government data Excel file
    """

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

    def plot_scatter_chart():
        """Create the bar chart of average trust scores by institution."""
        scatter_data = []
        for inst, name in institutions.items():
            # Skip if that dict key doesn't exist
            if f"{inst}_scores" not in data:
                continue

            # Average across all months for each institution
            scores_dict = data[f"{inst}_scores"]
            if scores_dict:
                avg_trust = sum(scores_dict.values()) / len(scores_dict)
            else:
                avg_trust = 0

            scatter_data.append({
                "Institution": name,
                "Trust Score": avg_trust,
                "Key": inst
            })

        scatter_df = pd.DataFrame(scatter_data).sort_values(by="Trust Score", ascending=True)
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
            yaxis=dict(title="Trust Score", range=[0, 4]),
            showlegend=False,
            bargap=0.5
        )
        return fig

    def create_demographic_time_series(selected_inst_key):
        """Create and display the time-series chart by demographic dimension."""
        trust_scores = data.get(f"{selected_inst_key}_rows", pd.DataFrame())
        if trust_scores.empty:
            st.warning(f"No data available for {selected_inst_key}")
            return

        institution_name = institutions.get(selected_inst_key, "Unknown Institution")

        demo_mapping = {
            "District": {
                "North": {"hebrew": "צפון", "color": "rgb(0, 128, 255)"},
                "Haifa": {"hebrew": "חיפה", "color": "rgb(255, 140, 0)"},
                "Center": {"hebrew": "מרכז", "color": "rgb(50, 205, 50)"},
                "Tel Aviv": {"hebrew": "תל אביב", "color": "rgb(255, 0, 255)"},
                "Jerusalem": {"hebrew": "ירושלים", "color": "rgb(255, 215, 0)"},
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
                "75+": {"hebrew": "75+", "color": "rgb(139, 0, 0)"},
                "65-74": {"hebrew": "65-74", "color": "rgb(178, 34, 34)"},
                "55-64": {"hebrew": "55-64", "color": "rgb(205, 92, 92)"},
                "45-54": {"hebrew": "45-54", "color": "rgb(240, 128, 128)"},
                "35-44": {"hebrew": "35-44", "color": "rgb(250, 128, 114)"},
                "25-34": {"hebrew": "25-34", "color": "rgb(255, 160, 122)"},
                "18-24": {"hebrew": "18-24", "color": "rgb(255, 192, 203)"}
            }
        }

        # We create two columns: one for the radio button on the left, one for the chart on the right
        col_left, col_right = st.columns([0.2, 0.8])
        with col_left:
            demo_choice = st.radio("Choose a demographic dimension:", ["All"] + list(demo_mapping.keys()), index=0)

        with col_right:
            fig = go.Figure()
            if demo_choice == "All":
                # Overall average trust for this institution
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
                selected_map = demo_mapping[demo_choice]
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
                            connectgaps=True,
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
                        text=f"Trust Scores for {institution_name} by {demo_choice} Over Time",
                        showarrow=False,
                        font=dict(size=16, family="Arial", color="black"),
                        xanchor="center"
                    )
                ]
            )
            st.plotly_chart(fig, use_container_width=True)

        # st.plotly_chart(fig, use_container_width=True)

    # ---- MAIN DASHBOARD LAYOUT ----
    # st.title("Israeli Sentiments Dashboard")
    # st.subheader("Trust in Institutions Over Time")

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

    # Interactive event capture
    selected_point = plotly_events(plot_scatter_chart(), click_event=True)

    if selected_point:
        selected_inst = selected_point[0]["x"]
        selected_inst_key = next((key for key, val in institutions.items() if val == selected_inst), None)
        create_demographic_time_series(selected_inst_key)

# Generate a list of discrete months for the slider (Make it globally available)
start_date = pd.Timestamp(2023, 10, 1).date()
end_date = pd.Timestamp(2024, 11, 30).date()
months = pd.date_range(start=start_date, end=end_date, freq='MS').to_period('M')
def rocket_strikes_map():
    # New File Locations
    data_dir = "C:/Users/liamo/PycharmProjects/visu_project/hellel_data"
    surveyCSV = f"{data_dir}/AggCountiesData.csv"
    alarmsCSV = f"{data_dir}/AggAlarmsData.csv"
    countiesSHP = f"{data_dir}/Mechozot_all/Mechozot_all.shp"

    # Load the data using the new load_data function
    @st.cache_data
    def load_data(surveyCSV, alarmsCSV, countiesSHP):
        gdf = gpd.read_file(countiesSHP)
        data = pd.read_csv(surveyCSV)
        alarms = pd.read_csv(alarmsCSV)

        # Convert date columns to appropriate types
        data["Date"] = pd.to_datetime(data["Date"], errors='coerce').dt.date  # Ensure consistent parsing
        data["Year-Month"] = pd.to_datetime(data["Year-Month"], errors='coerce').dt.to_period('M')
        alarms["Year-Month"] = pd.to_datetime(alarms["Year-Month"], errors='coerce').dt.to_period('M')

        # Convert to English names
        mapping_dict = {
            'מרכז': "Center",
            "תל אביב": "Tel-Aviv",
            "יהודה ושומרון": "Judea and Samaria",
            "ירושלים": "Jerusalem",
            "דרום": "South",
            "צפון": "North",
            "חיפה": "Haifa"
        }
        gdf["machoz"] = gdf["machoz"].replace(mapping_dict)

        return gdf, data, alarms

    # Load data
    gdf, data, alarms = load_data(surveyCSV, alarmsCSV, countiesSHP)


    filtered_data = data[
        (data['Year-Month'] >= selected_range[0]) &
        (data['Year-Month'] <= selected_range[1])
        ]

    if not filtered_data.empty:  # Ensure there is data to avoid errors
        # Filter alarms data based on selected period
        filtered_alarms = alarms[
            (alarms["Year-Month"] >= selected_range[0]) &
            (alarms["Year-Month"] <= selected_range[1] + pd.offsets.MonthEnd())
            ]

        # Group alarms data and calculate sum for period
        grouped_alarms = filtered_alarms.groupby(['data', 'outLat', 'outLong'], as_index=False)['count'].sum().round(2)

        # Group survey data and calculate percentage per answer per county
        avg_by_machoz = (
            filtered_data.groupby("machoz")
            .apply(lambda x: x.iloc[:, 2:].mean().round(2))
            .reset_index()
            .rename(columns={"index": "machoz"})
        )

        # Assuming 'feel_safe' is one of the columns in avg_by_machoz
        # If it's calculated differently, adjust accordingly
        if 'feel_safe' not in avg_by_machoz.columns:
            # Example calculation if 'feel_safe' needs to be computed
            # Adjust the columns as per your actual data
            feel_safe = (
                filtered_data.groupby("machoz").apply(
                    lambda x: (x.iloc[:, 2:4].sum(axis=1).mean() * 100).round(2)
                ).reset_index(name="feel_safe")
            )
            avg_by_machoz = avg_by_machoz.merge(feel_safe, on="machoz", how="left")

        # Merge survey data with GeoDataFrame
        merged_gdf = gdf.merge(avg_by_machoz, on="machoz", how="left").to_crs(epsg=4326)

        # Create Choropleth Map
        choropleth = go.Choroplethmapbox(
            geojson=json.loads(merged_gdf.to_json()),
            locations=merged_gdf["machoz"],
            featureidkey="properties.machoz",
            z=merged_gdf["feel_safe"],
            zmin=merged_gdf["feel_safe"].min(),
            zmax=merged_gdf["feel_safe"].max(),
            colorscale="Blues",
            colorbar=dict(title="Feel Secure (%)"),
            marker=dict(opacity=0.7),
            hovertext=merged_gdf.apply(
                lambda row: (
                    f"<b>Sense of Personal Security in {row['machoz']} District</b><br>"
                    f"Don't know: {row.get('dont know', 'N/A')}%<br>"
                    f"Low: {row.get('low', 'N/A')}%<br>"
                    f"Very Low: {row.get('very low', 'N/A')}%<br>"
                    f"Medium: {row.get('medium', 'N/A')}%<br>"
                    f"High: {row.get('high', 'N/A')}%<br>"
                    f"Very High: {row.get('very high', 'N/A')}%"
                ),
                axis=1
            ),
            hoverinfo="text",
        )

        # Create Scatter Map for Alarms
        scatter = go.Scattermapbox(
            lat=grouped_alarms["outLat"],
            lon=grouped_alarms["outLong"],
            mode="markers",
            marker=go.scattermapbox.Marker(
                size=5,
                opacity=0.7,
                color=grouped_alarms["count"],
                colorscale=[
                    [0, '#FF002B'],
                    [0.5, '#81001C'],
                    [1, '#000000']
                ],
                cmin=max(grouped_alarms["count"].mean() - grouped_alarms["count"].std() * 2, 0),
                # Minimum value for color range
                cmax=grouped_alarms["count"].mean() + grouped_alarms["count"].std() * 2,
                # Maximum value for color range
                showscale=True,  # Show the color scale legend
                colorbar=dict(
                    title="Alarms Amount",
                    x=1.2,
                ),
            ),
            text=grouped_alarms.apply(lambda row: f"Data: {row['data']}<br>Count: {row['count']}", axis=1),
            hoverinfo="text",
        )

        # Initialize Figure
        fig = go.Figure()
        fig.add_trace(choropleth)

        if show_alarms:
            fig.add_trace(scatter)

        # Update Layout
        fig.update_layout(
            mapbox=dict(
                style="carto-positron",
                center={"lat": 31.45, "lon": 35},
                zoom=6.7,
            ),
            width=900,  # Adjusted width for better visibility
            height=750,  # Adjusted height for better visibility
            # title="Sense of Personal Security",
            margin={"r": 0, "t": 50, "l": 0, "b": 0}
        )

        # Display the map
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No data available for the selected time range. Please select a wider period.")
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import pandas as pd
import plotly.graph_objs as go

def create_solidarity_dashboard():
    """
    This function creates the Solidarity Dashboard with an English user interface.
    Users can select surveys, questions, and visualization types to view the data.
    """

    # 1. Mapping of English question labels to Hebrew question strings
    question_mapping = {
        "Are you or a first-degree family member involved in combat?":
            "האם את.ה או בן משפחה בדרגה ראשונה שלך לוקח חלק בלחימה?",
        "Are you or a first-degree family member residing near the Gaza envelope or northern border?":
            "האם את.ה או בן משפחה בדרגה ראשונה שלך מתגורר בעוטף עזה או בגבול הצפון?",
        "Gender": "מגדר"
    }

    # 2. File selection with predefined questions (English labels)
    file_options = {
        "Has there been a change in the sense of solidarity in Israeli society at this time?":
            "solidarity_data/solidarity.xlsx",
        "How concerned are you about Israel's social situation on the day after the war?":
            "solidarity_data/matzv_chvrati.xlsx",
        "How optimistic are you about Israeli society's ability to recover from the crisis and grow?":
            "solidarity_data/mashber.xlsx"
    }

    # 3. Response mappings (in Hebrew → displayed as English)
    response_mappings = {
        "עד כמה את.ה מוטרד.ת או לא מוטרד.ת ממצבה החברתי של ישראל ביום שאחרי המלחמה דרג בסולם של 1-5, כאשר 5 = מוטרד מאד ו - 1 = לא מוטרד כלל": {
            "a1": "Not concerned at all",
            "a2": "Slightly concerned",
            "a3": "Moderately concerned",
            "a4": "Concerned",
            "a5": "Very concerned"
        },
        "עד כמה אתה אופטימי ביחס ליכולתה של החברה הישראלית להתאושש מהמשבר ולצמוח": {
            "a4": "Very optimistic",
            "a3": "Somewhat optimistic",
            "a2": "Somewhat pessimistic",
            "a1": "Very pessimistic"
        },
        "האם חל שינוי בתחושת הסולידריות בחברה הישראלית בעת הזו": {
            "a1": "Solidarity has strengthened significantly",
            "a2": "Solidarity has somewhat strengthened",
            "a3": "No change in solidarity",
            "a4": "Solidarity has somewhat decreased",
            "a5": "Solidarity has significantly decreased"
        }
    }

    # 4. The original 'predefined_questions' list (English labels)
    predefined_questions = list(question_mapping.keys())

    # --------------------------------------------------------------------------
    # 5. Here is the NEW PART: concise statements to replace the question text
    # --------------------------------------------------------------------------
    concise_statements = {
        "Are you or a first-degree family member involved in combat?":
            "Involved in combat (self or first-degree family member)",
        "Are you or a first-degree family member residing near the Gaza envelope or northern border?":
            "Residing near Gaza/northern border (self or first-degree family member)",
        "Gender": "Gender"
    }

    # We'll build a reverse lookup to recover the original question from its short statement
    statement_to_question = {v: k for k, v in concise_statements.items()}

    # --------------------------------------------------------------------------
    # 6. Streamlit UI
    # --------------------------------------------------------------------------
    # File selection dropdown
    selected_file = st.selectbox(
        "Select Subject to Display:",
        list(file_options.keys())
    )

    # Two columns for layout
    col1, col2 = st.columns([0.8, 2.2])

    with col1:
        viz_type = st.radio(
            "Select Visualization Type:",
            ["Aggregated results", "Results over time"]
        )

    # Instead of displaying the original question, we display the concise statement
    with col2:
        selected_statement = st.selectbox(
            "Select segmentation to display:",
            [concise_statements[q] for q in predefined_questions]
        )

    # Convert the selected statement back to the original question key
    selected_question = statement_to_question.get(selected_statement, selected_statement)

    # --------------------------------------------------------------------------
    # 7. Find the Hebrew question string
    # --------------------------------------------------------------------------
    hebrew_question = question_mapping.get(selected_question)

    # 8. Load the data
    try:
        df = pd.read_excel(file_options[selected_file])
        df["date"] = pd.to_datetime(df["date"])
    except FileNotFoundError:
        st.error(f"The selected survey file '{selected_file}' was not found. Please check the file path.")
        return
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return

    # 9. Filter data for this Hebrew question
    if hebrew_question:
        question_data = df[df["subject"] == hebrew_question].copy()
        question_data = question_data[question_data["sub_subject"] != "Total"]
    else:
        st.error("Selected question mapping not found.")
        return

    full_question = df["q_full"].iloc[0] if not df.empty else None

    # 10. Choose which visualization to draw
    if viz_type == "Aggregated results":
        create_bar_chart(question_data, full_question, selected_question, response_mappings)
    else:
        create_line_plot(question_data, full_question, selected_question, response_mappings)


# --------------------------------------------------------------------------
# Short question forms for chart titles
# --------------------------------------------------------------------------
QUESTION_SHORT_FORMS = {
    "Are you or a first-degree family member involved in combat?": "Combat involvement (family)",
    "Are you or a first-degree family member residing near the Gaza envelope or northern border?": "Border residence (family)",
    "How concerned are you about Israel's social situation on the day after the war?": "Post-war concerns",
    "How optimistic are you about Israeli society's ability to recover from the crisis and grow?": "Recovery optimism",
    "Has there been a change in the sense of solidarity in Israeli society at this time?": "Solidarity change"
}

def create_bar_chart(question_data, full_question, selected_question, response_mappings):
    """
    Creates a horizontal bar chart where:
    - 'כן' is displayed as 'Involved in combat (self or first-degree family member)'
    - 'לא' is displayed as 'Not involved in combat (self or first-degree family member)'
    """

    # ✅ Define correct color mapping
    color_mapping = {
        "Involved in combat (self or first-degree family member)": '#654321',  # Dark brown
        "Not involved in combat (self or first-degree family member)": '#333333',  # Dark gray
        "Female": "#8B0000",  # Dark red
        "Male": "#00008B",  # Dark blue
        "Living in North/Gaza Envelope": "#006400",  # Dark green
        "Not Living in North/Gaza Envelope": "#4B0082",  # Dark purple
        "Unknown": "#999999"  # Fallback color
    }

    # Identify numeric columns (a1, a2, etc.)
    numeric_cols = [col for col in question_data.columns if col.startswith("a")]
    numeric_cols.sort()

    # Group data by 'sub_subject' and calculate the mean
    chart_data = (
        question_data.groupby("sub_subject", as_index=False)[numeric_cols]
        .mean()
    )

    fig = go.Figure()

    # Convert the 'aX' codes to English labels if available
    if full_question in response_mappings:
        categories = [response_mappings[full_question].get(col, col) for col in numeric_cols]
    else:
        categories = [f"Response {i}" for i in range(1, len(numeric_cols) + 1)]

    # Reverse categories if not about solidarity (example logic)
    if "solidarity" not in selected_question.lower():
        categories = list(reversed(categories))
        numeric_cols = list(reversed(numeric_cols))

    # ✅ Brute-force correct legend labels BEFORE assigning colors
    for _, row in chart_data.iterrows():
        sub_subject = row["sub_subject"]

        if selected_question == "Are you or a first-degree family member involved in combat?":
            if sub_subject.strip() == "כן":
                legend_name = "Involved in combat (self or first-degree family member)"
            elif sub_subject.strip() == "לא":
                legend_name = "Not involved in combat (self or first-degree family member)"
            else:
                legend_name = "Unknown"
        else:
            if sub_subject.strip() == "זכר":
                legend_name = "Male"
            elif sub_subject.strip() == "נקבה":
                legend_name = "Female"
            elif sub_subject.strip() == "כן":
                legend_name = "Living in North/Gaza Envelope"
            elif sub_subject.strip() == "לא":
                legend_name = "Not Living in North/Gaza Envelope"
            else:
                legend_name = "Unknown"  # Fallback for unmapped values

        # ✅ Now, fetch the correct color AFTER assigning correct legend name
        bar_color = color_mapping.get(legend_name, color_mapping["Unknown"])

        values = [row[col] * 100 for col in numeric_cols]
        text_values = [f"{v:.1f}%" for v in values]

        fig.add_trace(go.Bar(
            name=legend_name,
            x=values,
            y=categories,
            text=text_values,
            textposition="outside",
            textfont=dict(size=14),
            orientation="h",
            marker_color=bar_color,  # ✅ Color now correctly maps
            hovertemplate="<b>%{y}</b><br>" +
                          f"{legend_name}: " + "%{x:.1f}%" +
                          "<extra></extra>"
        ))

    short_question = QUESTION_SHORT_FORMS.get(selected_question, selected_question)
    title_text = f"{short_question} (aggregated)"

    fig.update_layout(
        title={
            "text": title_text,
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 24}
        },
        barmode="group",
        height=700,
        width=1200,
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.2,
            "xanchor": "center",
            "x": 0.5
        },
        font=dict(size=16),
        margin=dict(t=80, b=200, l=100, r=200),
        paper_bgcolor="white",
        plot_bgcolor="rgba(248,249,250,0.5)",
        yaxis={"autorange": "reversed"},
        xaxis=dict(
            range=[-5, 62],
            showgrid=True,
            dtick=10
        )
    )

    fig.update_yaxes(
        categoryorder="array",
        categoryarray=categories,
        tickangle=0,
        automargin=True
    )

    st.plotly_chart(fig, use_container_width=True, key="bar_chart")




def create_line_plot(question_data, full_question, selected_question, response_mappings):
    """
    Creates a line plot showing sums of specified columns (like a1+a2) over time.
    Brute force for combat question to ensure correct legend labels.
    """

    # Which columns to sum for each question in Hebrew
    aggregations = {
        'האם חל שינוי בתחושת הסולידריות בחברה הישראלית בעת הזו': {
            'name': 'Solidarity has strengthened',
            'columns': ['a1', 'a2']
        },
        'עד כמה אתה אופטימי ביחס ליכולתה של החברה הישראלית להתאושש מהמשבר ולצמוח': {
            'name': 'Optimistic',
            'columns': ['a3', 'a4']
        },
        'עד כמה את.ה מוטרד.ת או לא מוטרד.ת ממצבה החברתי של ישראל ביום שאחרי המלחמה דרג בסולם של 1-5, כאשר 5 = מוטרד מאד ו - 1 = לא מוטרד כלל': {
            'name': 'Concerned',
            'columns': ['a4', 'a5']
        }
    }

    # Minimal color mapping
    color_mapping = {
        "Involved in combat (self or first-degree family member)": '#654321',
        "Not involved in combat (self or first-degree family member)": '#333333',

        'Female': '#8B0000',
        'Male': '#00008B',

        'Living in North/Gaza Envelope': '#006400',
        'Not Living in North/Gaza Envelope': '#4B0082',

        'Unknown': '#999999'
    }

    fig = go.Figure()

    agg_config = aggregations.get(full_question)
    if agg_config:
        # Sum the relevant columns
        aggregated_data = question_data.copy()
        aggregated_data['agg_value'] = aggregated_data[agg_config['columns']].sum(axis=1)

        # For each sub_subject, draw a line over time
        for sub_subject in aggregated_data['sub_subject'].unique():
            sub_data = aggregated_data[aggregated_data['sub_subject'] == sub_subject].sort_values('date')

            # Brute force only for the combat question
            if selected_question == "Are you or a first-degree family member involved in combat?":
                if sub_subject == 'כן':
                    legend_name = "Involved in combat (self or first-degree family member)"
                else:
                    legend_name = "Not involved in combat (self or first-degree family member)"
            else:
                # Fallback for other questions
                if sub_subject in ('זכר', 'Male'):
                    legend_name = "Male"
                elif sub_subject in ('נקבה', 'Female'):
                    legend_name = "Female"
                elif sub_subject == 'כן':
                    legend_name = "Living in North/Gaza Envelope"
                elif sub_subject == 'לא':
                    legend_name = "Not Living in North/Gaza Envelope"
                else:
                    legend_name = "Unknown"

            color = color_mapping.get(legend_name, color_mapping['Unknown'])

            fig.add_trace(go.Scatter(
                x=sub_data['date'],
                y=sub_data['agg_value'] * 100,
                name=legend_name,
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(size=8),
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>" +
                    f"{agg_config['name']}: %{{y:.1f}}%<br>" +
                    "<extra></extra>"
                )
            ))

        # Title text
        short_question = response_mappings.get(selected_question, selected_question)
        title_text = f"{short_question} - Over Time"

        # Get Y range
        all_y_values = []
        for trace in fig.data:
            all_y_values.extend(trace.y)
        max_y = min(max(all_y_values) + 5, 100) if all_y_values else 100

        # Get X (date) range
        all_dates = []
        for trace in fig.data:
            all_dates.extend(trace.x)
        if all_dates:
            min_date, max_date = min(all_dates), max(all_dates)
        else:
            min_date, max_date = None, None

        fig.update_layout(
            title={
                'text': title_text,
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
                'y': -0.3,
                'xanchor': 'center',
                'x': 0.5,
                'font': {'size': 12}
            },
            font=dict(size=14),
            margin=dict(t=100, b=150, l=100, r=100),
            paper_bgcolor='white',
            plot_bgcolor='rgba(248,249,250,0.5)',
            yaxis=dict(
                range=[0, max_y],
                dtick=10,
                gridcolor='lightgray'
            ),
            xaxis=dict(
                gridcolor='lightgray',
                type='date',
                dtick='M1',
                tickformat='%b %Y',
                range=[min_date, max_date] if min_date else None
            )
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

        st.plotly_chart(fig, use_container_width=True, key="line_plot")
    else:
        st.error("Question configuration not found")


def update_layout(fig, selected_question):
    """
    (Optional) Helper to standardize a Plotly figure layout if desired.
    """
    fig.update_layout(
        title={
            'text': f'Survey Results: {selected_question}',
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

    st.plotly_chart(fig, use_container_width=True, key="layout")




def dashboard_overview():
    st.title("War of Iron Swords: Public Opinion Analytics Dashboard")

    # Overview Section
    st.header("Overview")
    st.write("""
    This dashboard visualizes monthly public opinion surveys related to the War of Iron Swords 
    to provide insights into the ongoing public sentiment along the conflict period.
    """)

    # Data Sources Section
    st.header("Data Sources")

    # Public Opinion Research
    st.subheader("Public Opinion Research")
    st.write("""
    - **Monthly surveys** conducted among the Israeli population by the **INSS** 
      ([Institute for National Security Studies](#)).
    - **Sample size**: ~580 respondents per survey.
    - **Demographics**: Jewish Israeli adults (18+).
    - **Sampling methodology**: Representative of the national adult population.
    - **Statistical validity**: ±3.5% margin of error at **95% confidence level**.
    """)

    # Alarms Data
    st.subheader("Alarms Data")
    st.write("""
    - **Real-time recordings** of alerts regarding **rocket launches** and **hostile aircraft intrusions**.
    - Data collected by scraping from **Pikud HaOref** ([Home Front Command](#)).
    """)

    # Purpose Section
    st.header("Purpose")
    st.write("""
    This dashboard serves as a tool for:
    - **Tracking** evolving public sentiment throughout the war.
    - **Analyzing** relations between security events and public opinion.
    - **Providing** **access** to war-related insights.
    """)

    # Focus Areas
    st.header("Focus Areas")

    # 1. Public Trust
    st.subheader("1. Public Trust In Institutions And Public Figures")
    st.write("Measuring confidence levels in public institutions throughout the war period.")

    # 2. Personal Security
    st.subheader("2. Sense of Personal Security")
    st.write("""
    Analyzing citizens' sense of personal security with regional breakdowns, 
    providing insights into how different areas experience and perceive threats.
    """)

    # 3. National Perspective
    st.subheader("3. Israel’s Social Outlook")
    st.write("""
    Evaluating collective attitudes toward:
    - **National unity and solidarity**.
    - **Current social conditions**.
    - **Recovery prospects and future growth potential post-crisis**.
    """)


def personal_security_text():
    """Displays an explanation of the personal security section in the dashboard."""
    st.subheader("Purpose")
    st.write(
        "Analyzing citizens' sense of personal security with regional breakdowns, "
        "providing insights into how different areas experience and perceive threats."
    )

    st.subheader("How To Use")
    st.write(
        "Filter period using the slider.\n\n"
        "The district’s colors represent the average percentage of people in the district who feel secure – "
        "the darker the color the more sense of security in this district. "
        "Hover over a district to see the average survey responses for this district in the selected period.\n\n"
        "It is optional to show alarms location, by checking the checkbox. "
        "The dots represent the sum of alarms per locality, "
        "the darker the color the higher number of alarms at this location. "
        "Hover over a dot to see the locality and the sum of alarms at this locality in the selected period."
    )


def public_trust_text():
    """
    Displays text describing the purpose and usage instructions for the Public Trust visualization.
    """
    st.markdown("""


    ### Purpose 
    Analyzing confidence levels in public institutions throughout the war period.

    ### How To Use
    - **Main Plot**: View the average trust levels during the war for each institution or public figure.
    - **Drill Down**: Click on a specific circle to see the trust levels throughout the war.
    - **Demographic Breakdown**: Choose a demographic dimension to see changes in trust over time within each subgroup.
    - **Scoring Range**: Trust scores range from 1 (lowest) to 4 (highest).
    - **Over Time view**: Click on a line in the plot or a category in the legend to hide it. Click again on the legend to bring it back. 
    """)


def text_for_solidarity():
    """
    Displays text describing the purpose and usage instructions
    for the Solidarity-related visualization.
    """
    st.markdown("""
    ### Purpose
    Evaluating collective attitudes toward:
    - National sense of solidarity  
    - Social situation the day after the war  
    - Recovery prospects and future potential growth post-crisis  

    ### How To Use
    1. **Choose a subject** you want to display.  
    2. **Choose a visualization type**: aggregated results or results over time.  
    3. **Choose a segmentation** to display the plots by.  
    4. Each subgroup is represented in a different color.  

    - **Aggregated Results**: Shows a bar plot with the average percentage for each answer within a given subgroup.  
    - **Results Over Time**: Displays how the highest score for each subgroup changes throughout the war.
    """)


##############################
# --- STREAMLIT APP LAYOUT ---
##############################

# st.sidebar.title("Visualizations")
st.set_page_config(
    page_title="My Dashboard",
    layout="wide"  # This makes the app use the full browser width
)

visualization = st.sidebar.radio(
    "Menu",
    [
        "Dashboard Overview",
        "Public Trust In Institutions And Public Figures",
        "Sense of Personal Security",
        "Israel’s Social Outlook"
    ],
    label_visibility="visible"
)
# institutions_data = load_data()
# months = institutions_data["months"]

# if visualization == "Sense of Personal Security":
#     st.header("Sense of Personal Security")
#
#     col3, col4 = st.columns([1, 1.5])
#
#     with col3:
#         personal_security_text()
#
#     with col4:
#         rocket_strikes_map()
if visualization == "Sense of Personal Security":
    st.header("Sense of Personal Security")

    col1, col2 = st.columns([1, 2])  # Left side for text & controls, Right side for map

    with col1:
        personal_security_text()

        # Move the time period selection here
        # st.write("Select Time Period")
        selected_range = st.select_slider(
            key="slider1",
            label="Select Period",
            options=list(months),  # ✅ Now `months` is defined
            value=(months[0], months[-1])  # Default to full range
        )

        # Move the checkbox here
        show_alarms = st.checkbox(label="Show Alarms", value=True)

    with col2:
        rocket_strikes_map()  # Map will now be positioned correctly


elif visualization == "Dashboard Overview":
    dashboard_overview()

if visualization == "Public Trust In Institutions And Public Figures":
    st.header("Public Trust In Institutions And Public Figures")

    col7, col8 = st.columns([1, 2])

    with col7:
        public_trust_text()
        # demo_choice = st.radio("Choose a demographic dimension:",
        #                        ["District", "Religiousness", "Political stance", "Age"])  # Move radio buttons here

    with col8:
        # Usage example:
        create_trust_dashboard(
            bibi_data_path="data_storage/bibi.xlsx",
            tzal_data_path="data_storage/tzal.xlsx",
            mishtara_data_path="data_storage/mishtra.xlsx",
            memshala_data_path="data_storage/memshla.xlsx"
        )

    # # Usage example:
    # create_trust_dashboard(
    #     bibi_data_path="data_storage/bibi.xlsx",
    #     tzal_data_path="data_storage/tzal.xlsx",
    #     mishtara_data_path="data_storage/mishtra.xlsx",
    #     memshala_data_path="data_storage/memshla.xlsx"
    # )


elif visualization == "Israel’s Social Outlook":
    st.title("Israel’s Social Outlook")

    col9, col10 = st.columns([1, 2])

    with col9:
        text_for_solidarity()

    with col10:
        # Usage example:
        create_solidarity_dashboard()

    # text_for_solidarity()
    # create_solidarity_dashboard()
