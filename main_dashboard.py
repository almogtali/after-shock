
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

    def plot_scatter_chart():
        scatter_data = []
        fixed_size = 60

        for inst, name in institutions.items():
            if f"{inst}_scores" not in data:
                continue

            avg_trust = sum(data[f"{inst}_scores"].values()) / len(data[f"{inst}_scores"]) if data[
                f"{inst}_scores"] else 0

            scatter_data.append({
                "Institution": name,
                "Trust Score": avg_trust,
                "Key": inst,
                "Size": fixed_size
            })

        scatter_df = pd.DataFrame(scatter_data)

        fig = go.Figure()
        for _, row in scatter_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row["Institution"]],
                y=[row["Trust Score"]],
                mode="markers+text",
                marker=dict(
                    size=fixed_size,
                    color=color_map[row["Key"]],
                    line=dict(width=2, color="white")
                ),
                text=f"{row['Trust Score']:.2f}",  # Only show trust score
                textposition="middle center",  # Center the text in the bubble
                hoverinfo="none"  # Remove hover information
            ))

        fig.update_layout(
            title="Trust Scores by Institution",
            yaxis=dict(title="Trust Score (1-4)", range=[1, 4]),
            xaxis=dict(title="Institution"),
            showlegend=False
        )
        return fig

    def create_demographic_time_series(selected_inst_key):
        trust_scores = data.get(f"{selected_inst_key}_rows", pd.DataFrame())

        if trust_scores.empty:
            st.warning(f"No data available for {selected_inst_key}")
            return None

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
                "Right": "ימין",
                "Center": "מרכז",
                "Left": "שמאל",
                "Refuses to Answer": "מסרב"
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

        demo_choice = st.radio("Choose a demographic dimension:", list(demo_mapping.keys()))



        fig = go.Figure()

        if demo_choice in ["Religiousness", "Age", "District"]:
            selected_map = demo_mapping[demo_choice]
            for eng_label, value in selected_map.items():
                sub_data = trust_scores[trust_scores["sub_subject"] == value["hebrew"]].copy()

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
        else:
            selected_map = {k: v for k, v in demo_mapping[demo_choice].items()}
            for eng_label, hebrew_label in selected_map.items():
                sub_data = trust_scores[trust_scores["sub_subject"] == hebrew_label].copy()

                if not sub_data.empty:
                    monthly_avg = sub_data.groupby("month_year")["trust_score"].mean().reset_index()
                    monthly_avg["month_year_str"] = monthly_avg["month_year"].astype(str)

                    fig.add_trace(go.Scatter(
                        x=monthly_avg["month_year_str"],
                        y=monthly_avg["trust_score"],
                        name=eng_label,
                        mode="lines+markers",
                        connectgaps=True,
                        line=dict(width=2),
                        marker=dict(size=8)
                    ))

        fig.update_layout(
            title=f"Trust Scores for {institution_name} Over Time by {demo_choice}",
            xaxis_title="Month",
            yaxis_title="Trust Score",
            yaxis_range=[1, 4],
            hovermode="x unified",
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="top",  # Anchoring to the top of the legend box
                y=-0.2,  # Moving the legend below the plot
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255, 255, 255, 0.8)"
            ),
            margin=dict(b=100)  # Adding space at the bottom for the legend
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

    # Generate a list of discrete months for the slider
    start_date = pd.Timestamp(2023, 10, 1).date()
    end_date = pd.Timestamp(2024, 11, 30).date()
    months = pd.date_range(start=start_date, end=end_date, freq='MS').to_period('M')

    # Sidebar: Time Period Selection using discrete months
    st.write("Select Time Period")
    selected_range = st.select_slider(
        key="slider1",
        label="Select Period",
        options=list(months),
        value=(months[0], months[-1])  # Default to full range
    )

    # Checkbox for showing alarms
    # show_alarms = st.sidebar.checkbox(label="Show Alarms", value=True)
    show_alarms = st.checkbox(label="Show Alarms", value=True)
    # Filter data based on the selected range
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
                cmin=max(grouped_alarms["count"].mean() - grouped_alarms["count"].std()*2, 0),  # Minimum value for color range
                cmax=grouped_alarms["count"].mean() + grouped_alarms["count"].std()*2,  # Maximum value for color range
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
            title="Sense of Personal Security",
            margin={"r":0,"t":50,"l":0,"b":0}
        )

        # Display the map
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No data available for the selected time range. Please select a wider period.")

def create_solidarity_dashboard():
    """
    This function creates the Solidarity Dashboard with an English user interface.
    Users can select surveys, questions, and visualization types to view the data.
    """
    # Mapping of English question labels to Hebrew question strings
    question_mapping = {
        'Are you or a first-degree family member involved in combat?': 'האם את.ה או בן משפחה בדרגה ראשונה שלך לוקח חלק בלחימה?',
        'Are you or a first-degree family member residing near the Gaza envelope or northern border?': 'האם את.ה או בן משפחה בדרגה ראשונה שלך מתגורר בעוטף עזה או בגבול הצפון?',
        'Gender': 'מגדר'
    }

    # File selection with predefined questions for each file (English labels)
    file_options = {
        'Has there been a change in the sense of solidarity in Israeli society at this time?': 'solidarity_data/solidarity.xlsx',
        'How concerned are you about Israel\'s social situation on the day after the war?': 'solidarity_data/matzv_chvrati.xlsx',
        'How optimistic are you about Israeli society\'s ability to recover from the crisis and grow?': 'solidarity_data/mashber.xlsx'
    }

    # Define response mappings for each question (remains in Hebrew)
    response_mappings = {
        'עד כמה את.ה מוטרד.ת או לא מוטרד.ת ממצבה החברתי של ישראל ביום שאחרי המלחמה דרג בסולם של 1-5, כאשר 5 = מוטרד מאד ו - 1 = לא מוטרד כלל': {
            'a1': 'Not concerned at all',
            'a2': 'Slightly concerned',
            'a3': 'Moderately concerned',
            'a4': 'Concerned',
            'a5': 'Very concerned'
        },
        'עד כמה אתה אופטימי ביחס ליכולתה של החברה הישראלית להתאושש מהמשבר ולצמוח': {
            'a1': 'Very pessimistic',
            'a2': 'Somewhat pessimistic',
            'a3': 'Somewhat optimistic',
            'a4': 'Very optimistic'
        },
        'האם חל שינוי בתחושת הסולידריות בחברה הישראלית בעת הזו': {
            'a1': 'Solidarity has strengthened significantly',
            'a2': 'Solidarity has somewhat strengthened',
            'a3': 'No change in solidarity',
            'a4': 'Solidarity has somewhat decreased',
            'a5': 'Solidarity has significantly decreased'
        }
    }

    # Predefined questions in English
    predefined_questions = list(question_mapping.keys())

    # Title of the Dashboard


    # File selection dropdown
    selected_file = st.selectbox(
        'Select Subject to Display:',
        list(file_options.keys())
    )

    # Create two columns for the controls
    col1, col2 = st.columns([0.8,2.2])

    # Place visualization type selector in the first column
    with col1:
        viz_type = st.radio(
            "Select Visualization Type:",
            ["Aggregated results", "Results over time"]
        )

    # Place segmentation selector in the second column
    with col2:
        selected_question = st.selectbox(
            'Select segmentation to display:',
            predefined_questions
        )

    # Get the Hebrew question string based on the selected English label
    hebrew_question = question_mapping.get(selected_question)

    # Data Loading with Error Handling
    try:
        df = pd.read_excel(file_options[selected_file])
        df['date'] = pd.to_datetime(df['date'])
    except FileNotFoundError:
        st.error(f"The selected survey file '{selected_file}' was not found. Please check the file path.")
        return
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return

    # Data Filtering
    if hebrew_question:
        question_data = df[df['subject'] == hebrew_question].copy()
        question_data = question_data[question_data['sub_subject'] != 'Total']
    else:
        st.error("Selected question mapping not found.")
        return

    full_question = df['q_full'].iloc[0] if not df.empty else None

    # Visualization Selection
    if viz_type == "Aggregated results":
        create_bar_chart(question_data, full_question, selected_question, response_mappings)
    else:
        create_line_plot(question_data, full_question, selected_question, response_mappings)

# Add this mapping at the top of your script
QUESTION_SHORT_FORMS = {
    'Are you or a first-degree family member involved in combat?': 'Combat involvement (family)',
    'Are you or a first-degree family member residing near the Gaza envelope or northern border?': 'Border residence (family)',
    'How concerned are you about Israel\'s social situation on the day after the war?': 'Post-war concerns',
    'How optimistic are you about Israeli society\'s ability to recover from the crisis and grow?': 'Recovery optimism',
    'Has there been a change in the sense of solidarity in Israeli society at this time?': 'Solidarity change'
}

def create_bar_chart(question_data, full_question, selected_question, response_mappings):
    """
    Creates a horizontal bar chart based on the selected question data with x-axis range capped at 70%.
    """
    # Identify numeric columns representing response options (e.g., a1, a2, ...)
    numeric_cols = [col for col in question_data.columns if col.startswith('a')]
    numeric_cols.sort()  # Ensure columns are sorted

    # Group data by 'sub_subject' and calculate the mean for each response option
    chart_data = (
        question_data.groupby('sub_subject', as_index=False)[numeric_cols]
        .mean()
    )

    fig = go.Figure()

    # Define colors for different groups
    option1_color = '#082f49'  # Dark blue
    option2_color = '#f97316'  # Orange

    # Map response codes to descriptive labels using response_mappings
    if full_question in response_mappings:
        categories = [response_mappings[full_question][col] for col in numeric_cols]
    else:
        categories = [f'Response {i}' for i in range(1, len(numeric_cols) + 1)]

    # Reverse the order of categories and numeric columns for better visualization
    categories = categories[::-1]
    numeric_cols = numeric_cols[::-1]

    # Add horizontal bars for each sub_subject
    for idx, row in chart_data.iterrows():
        values = [row[col] * 100 for col in numeric_cols]  # Convert to percentages
        text_values = [f"{v:.1f}%" for v in values]  # Format text with percentages

        fig.add_trace(go.Bar(
            name=row['sub_subject'],
            x=values,
            y=categories,
            text=text_values,  # Add percentage labels
            textposition='outside',  # Place labels outside bars
            textfont=dict(size=14),  # Increase label font size
            constraintext='none',  # Prevent text from being cut off
            cliponaxis=False,  # Prevent clipping of labels
            orientation='h',
            marker_color=option1_color if idx == 0 else option2_color,
            hovertemplate="<b>%{y}</b><br>" +
                          row['sub_subject'] + ": %{x:.1f}%" +
                          "<extra></extra>"
        ))

    # Get shorter version of the question for the title
    short_question = QUESTION_SHORT_FORMS.get(selected_question, selected_question)
    title_text = f"{short_question} aggregated"

    # Update layout with shorter title
    fig.update_layout(
        title={
            'text': title_text,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        barmode='group',
        height=700,
        width=1200,
        showlegend=True,
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': -0.2,
            'xanchor': 'center',
            'x': 0.5
        },
        font=dict(size=16),
        # Increase margins to accommodate labels
        margin=dict(t=80, b=200, l=100, r=200),
        paper_bgcolor='white',
        plot_bgcolor='rgba(248,249,250,0.5)',
        yaxis={'autorange': 'reversed'},
        # Modified x-axis range to cap at 70%
        xaxis=dict(
            range=[-5, 62],  # Changed from 105 to 70
            showgrid=True,
            dtick=10  # Add gridlines every 10%
        )
    )
    fig.update_yaxes(
        tickangle=-45,  # Negative angle tilts labels to the left
        automargin=True  # Let Plotly manage margins automatically
    )

    # Ensure full width in Streamlit
    st.plotly_chart(fig, use_container_width=True, key="bar_chart")


def create_line_plot(question_data, full_question, selected_question, response_mappings):
    """
    Creates a line plot showing specific aggregations for different questions.
    """
    # Define aggregation rules based on the question
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

    # Demographic pairs and their translations
    demographic_pairs = {
        'מגדר': {
            'נקבה': 'Female',
            'זכר': 'Male'
        },
        'גרים בצפון/עוטף': {
            'כן': 'Living in North/Gaza Envelope',
            'לא': 'Not Living in North/Gaza Envelope'
        },
        'משפחה של לוחמים': {
            'כן': 'Family of Combatants',
            'לא': 'Not Family of Combatants'
        }
    }

    # Color mapping with distinctive colors
    color_mapping = {
        'Female': '#8B0000',  # Dark red
        'Male': '#00008B',  # Dark blue
        'Living in North/Gaza Envelope': '#006400',  # Dark green
        'Not Living in North/Gaza Envelope': '#4B0082',  # Dark purple
        'Family of Combatants': '#654321',  # Dark brown
        'Not Family of Combatants': '#333333'  # Dark gray
    }

    fig = go.Figure()

    # Get aggregation rules for the current question
    agg_config = aggregations.get(full_question)
    if agg_config:
        aggregated_data = question_data.copy()

        # Calculate the sum of specified columns
        aggregated_data['agg_value'] = aggregated_data[agg_config['columns']].sum(axis=1)

        # Plot line for each demographic group
        for sub_subject in aggregated_data['sub_subject'].unique():
            sub_data = aggregated_data[aggregated_data['sub_subject'] == sub_subject].sort_values('date')

            # Find the correct English translation for the demographic group
            english_name = None
            for demo_group in demographic_pairs.values():
                if sub_subject in demo_group:
                    english_name = demo_group[sub_subject]
                    break

            if english_name:
                fig.add_trace(go.Scatter(
                    x=sub_data['date'],
                    y=sub_data['agg_value'] * 100,  # Convert to percentages
                    name=english_name,
                    mode='lines+markers',
                    line=dict(color=color_mapping[english_name], width=2),
                    marker=dict(size=8),
                    hovertemplate=(
                            "Date: %{x|%Y-%m-%d}<br>" +
                            f"{agg_config['name']}: %{{y:.1f}}%<br>" +
                            "<extra></extra>"
                    )
                ))

        # Get shorter version of the question for the title
        short_question = response_mappings.get(selected_question, selected_question)
        title_text = f"{short_question} - Over Time"

        # Get data range for y-axis
        y_values = []
        for trace in fig.data:
            y_values.extend(trace.y)
        min_y = max(min(y_values) - 5, 0)  # Don't go below 0
        max_y = min(max(y_values) + 5, 100)  # Don't exceed 100

        # Get data range for x-axis
        dates = []
        for trace in fig.data:
            dates.extend(trace.x)
        min_date = min(dates)
        max_date = max(dates)

        # Update layout with optimized ranges and more date ticks
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
                'orientation': 'h',  # Set legend to horizontal
                'yanchor': 'bottom',  # Anchor to the bottom
                'y': -0.3,  # Move the legend below the plot
                'xanchor': 'center',  # Center the legend
                'x': 0.5,  # Align to center horizontally
                'font': {'size': 12}
            },
            font=dict(size=14),
            margin=dict(t=100, b=150, l=100, r=100),  # Adjust bottom margin for legend space
            paper_bgcolor='white',
            plot_bgcolor='rgba(248,249,250,0.5)',
            yaxis=dict(
                range=[min_y, max_y],  # Dynamically set y-axis range
                dtick=10,
                gridcolor='lightgray'
            ),
            xaxis=dict(
                gridcolor='lightgray',
                type='date',
                dtick='M1',  # Show monthly ticks
                tickformat='%b %Y',  # Format as "Jan 2024"
                range=[min_date, max_date]  # Dynamically set x-axis range
            )
        )

        # Add light grid lines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

        # Render the plot
        st.plotly_chart(fig, use_container_width=True, key="line_plot")
    else:
        st.error("Question configuration not found")



def update_layout(fig, selected_question):
    """
    Optional helper function to update the layout of a Plotly figure.
    This function standardizes the layout settings across different chart types.
    """
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

    # Customize X-axis
    fig.update_xaxes(
        title_text='Percentage (%)',
        range=[0, 100],
        gridcolor='rgba(0,0,0,0.1)',
        showgrid=True,
        tickformat='.1f',
        zeroline=False
    )

    # Customize Y-axis
    fig.update_yaxes(
        title_text='',
        gridcolor='rgba(0,0,0,0.1)',
        showgrid=False,
        automargin=True,
        tickfont=dict(size=14)
    )

    # Render the updated figure in Streamlit
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
    st.subheader("1. Public Trust")
    st.write("Measuring confidence levels in public institutions throughout the war period.")

    # 2. Personal Security
    st.subheader("2. Personal Security")
    st.write("""
    Analyzing citizens' sense of personal security with regional breakdowns, 
    providing insights into how different areas experience and perceive threats.
    """)

    # 3. National Perspective
    st.subheader("3. National Perspective")
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
    ### Public Trust

    **Purpose**  
    Analyzing confidence levels in public institutions throughout the war period.

    **How To Use**  
    - **Main Plot**: View the average trust levels during the war for each institution or public figure.
    - **Drill Down**: Click on a specific circle to see the trust levels throughout the war.
    - **Demographic Breakdown**: Choose a demographic dimension to see changes in trust over time within each subgroup.
    - **Scoring Range**: Trust scores range from 1 (lowest) to 4 (highest).
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
    layout="wide"       # This makes the app use the full browser width
)

visualization = st.sidebar.radio(
    "Menu",
    [
        "Dashboard Overview",
        "Sense of Personal Security",
        "Trust in Institutions Over Time",
        "Israel’s Social Outlook"
    ],
    label_visibility="visible"
)
# institutions_data = load_data()
# months = institutions_data["months"]

if visualization == "Sense of Personal Security":
    st.header("Sense of Personal Security")

    col3, col4 = st.columns([1, 1.5])

    with col3:
        personal_security_text()

    with col4:
        rocket_strikes_map()

elif visualization == "Dashboard Overview":
    dashboard_overview()



if visualization == "Trust in Institutions Over Time":
    st.header("Trust in institutions and public figures during the War of Iron Swords")

    col7, col8 = st.columns([1, 2])

    with col7:
        public_trust_text()

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
